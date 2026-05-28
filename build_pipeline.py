"""End-to-end pipeline: load Online Retail II, build repurchase model, save plots + metrics."""
import json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             precision_recall_curve, brier_score_loss,
                             confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings('ignore')

HERE = Path(__file__).parent
ART  = HERE / 'artifacts'
ART.mkdir(exist_ok=True)
plt.style.use('ggplot')
sns.set_palette('husl')

# ---------- 1. Load ----------
PKL = HERE / 'online_retail.pkl'
XLSX = HERE / 'online_retail_II.xlsx'
if PKL.exists():
    df = pd.read_pickle(PKL)
else:
    xl = pd.ExcelFile(XLSX)
    df = pd.concat([pd.read_excel(XLSX, sheet_name=s) for s in xl.sheet_names], ignore_index=True)
    df.to_pickle(PKL)
df.columns = [c.strip() for c in df.columns]
df.rename(columns={'Customer ID':'CustomerID'}, inplace=True)
print(f'Loaded {len(df):,} rows')

# ---------- 2. Quality + cleaning ----------
quality = {
    'rows_raw': int(len(df)),
    'cols': int(df.shape[1]),
    'date_min': str(df['InvoiceDate'].min()),
    'date_max': str(df['InvoiceDate'].max()),
    'nulls_pct': {c: float(df[c].isna().mean()*100) for c in df.columns},
    'duplicates': int(df.duplicated().sum()),
}

df['InvoiceStr'] = df['Invoice'].astype(str)
df['IsCancel']   = df['InvoiceStr'].str.startswith('C')
df['StockStr']   = df['StockCode'].astype(str).str.upper()
BAD_CODES = {'POST','DOT','M','BANK CHARGES','AMAZONFEE','ADJUST','D','CRUK','PADS','C2','B','S','TEST001','TEST002','GIFT','SAMPLES','m'}
df['IsService'] = df['StockStr'].isin(BAD_CODES) | df['StockStr'].str.contains(r'^[A-Z]+$', regex=True, na=False)

filt = df[
    (~df['IsCancel']) &
    (~df['IsService']) &
    (df['Quantity'] > 0) &
    (df['Price']    > 0) &
    (df['CustomerID'].notna())
].copy()
filt['Revenue'] = filt['Quantity'] * filt['Price']
# remove gross outliers (top 0.5% revenue per line)
cap = filt['Revenue'].quantile(0.995)
filt = filt[filt['Revenue'] <= cap]
filt['CustomerID'] = filt['CustomerID'].astype(int)
print(f'After clean: {len(filt):,} rows | {filt["CustomerID"].nunique():,} customers')

quality['rows_clean']      = int(len(filt))
quality['customers_clean'] = int(filt['CustomerID'].nunique())
quality['revenue_cap']     = float(cap)

# ---------- 3. Cutoff + target ----------
DATE_END    = filt['InvoiceDate'].max()
CUTOFF      = pd.Timestamp('2011-06-09')   # 6 months hold-out
LABEL_END   = CUTOFF + pd.Timedelta(days=180)

train = filt[filt['InvoiceDate'] <  CUTOFF].copy()
label = filt[(filt['InvoiceDate'] >= CUTOFF) & (filt['InvoiceDate'] <= LABEL_END)].copy()
print(f'Cutoff {CUTOFF.date()} | train tx={len(train):,} | label tx={len(label):,}')

repeaters = set(label['CustomerID'].unique())
custs = train['CustomerID'].unique()
y_df = pd.DataFrame({'CustomerID': custs})
y_df['Repurchase'] = y_df['CustomerID'].isin(repeaters).astype(int)

# ---------- 4. Features ----------
ref = CUTOFF
g = train.groupby('CustomerID')
feat = pd.DataFrame({
    'Recency':      (ref - g['InvoiceDate'].max()).dt.days,
    'Tenure':       (ref - g['InvoiceDate'].min()).dt.days,
    'Frequency':    g['Invoice'].nunique(),
    'NLines':       g.size(),
    'NProducts':    g['StockCode'].nunique(),
    'Monetary':     g['Revenue'].sum(),
    'AvgTicket':    g['Revenue'].sum() / g['Invoice'].nunique(),
    'AvgLine':      g['Revenue'].mean(),
    'AvgQty':       g['Quantity'].mean(),
    'StdRevenue':   g['Revenue'].std().fillna(0),
    'TopCountry':   g['Country'].agg(lambda s: s.mode().iat[0] if len(s) else 'Unknown'),
}).reset_index()
feat['IsUK']        = (feat['TopCountry'] == 'United Kingdom').astype(int)
feat['BuysPerMonth']= feat['Frequency'] / (feat['Tenure'].clip(lower=1) / 30)
feat = feat.merge(y_df, on='CustomerID', how='inner')

print(f'Customers labelled: {len(feat):,} | repeat rate: {feat["Repurchase"].mean():.3f}')

# ---------- 5. Plots: EDA ----------
# 5a monthly revenue
mr = filt.set_index('InvoiceDate')['Revenue'].resample('ME').sum()/1000
fig, ax = plt.subplots(figsize=(9,4))
mr.plot(ax=ax, marker='o', color='#2E86AB')
ax.axvline(CUTOFF, color='red', ls='--', label=f'Cutoff {CUTOFF.date()}')
ax.set_ylabel('Revenue (£k)'); ax.set_title('Monthly revenue + cutoff'); ax.legend()
fig.tight_layout(); fig.savefig(ART/'eda_monthly_revenue.png', dpi=130); plt.close(fig)

# 5b top countries
top_c = filt.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10)/1000
fig, ax = plt.subplots(figsize=(8,4))
top_c.plot(kind='barh', ax=ax, color='#A23B72')
ax.invert_yaxis(); ax.set_xlabel('Revenue (£k)'); ax.set_title('Top 10 countries by revenue')
fig.tight_layout(); fig.savefig(ART/'eda_top_countries.png', dpi=130); plt.close(fig)

# 5c label balance
fig, ax = plt.subplots(figsize=(5,4))
feat['Repurchase'].value_counts().rename({0:'No recompra',1:'Recompra'}).plot(
    kind='bar', ax=ax, color=['#D7263D','#1B998B'])
ax.set_title(f'Balance del target (rate={feat["Repurchase"].mean():.1%})')
ax.set_ylabel('Clientes'); plt.xticks(rotation=0)
fig.tight_layout(); fig.savefig(ART/'eda_target_balance.png', dpi=130); plt.close(fig)

# 5d RFM distributions
fig, axes = plt.subplots(1,3, figsize=(12,3.5))
for ax, col, title in zip(axes, ['Recency','Frequency','Monetary'],
                          ['Recencia (días)','Frecuencia (facturas)','Monetario (£)']):
    data = np.log1p(feat[col]) if col=='Monetary' else feat[col]
    ax.hist(data, bins=40, color='#2E86AB', alpha=.8)
    ax.set_title(title); ax.set_xlabel('')
fig.suptitle('Distribución RFM por cliente'); fig.tight_layout()
fig.savefig(ART/'eda_rfm.png', dpi=130); plt.close(fig)

# ---------- 6. Modelling ----------
FEATCOLS = ['Recency','Tenure','Frequency','NLines','NProducts','Monetary',
            'AvgTicket','AvgLine','AvgQty','StdRevenue','BuysPerMonth','IsUK']
X = feat[FEATCOLS].values
y = feat['Repurchase'].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

results = {}
def evaluate(name, proba):
    auc = roc_auc_score(yte, proba)
    ap  = average_precision_score(yte, proba)
    brier = brier_score_loss(yte, proba)
    results[name] = {'AUC':auc, 'AP':ap, 'Brier':brier}
    print(f'  {name:12s}  AUC={auc:.4f}  AP={ap:.4f}  Brier={brier:.4f}')
    return auc, proba

# baseline LR
lr = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(max_iter=2000, C=1.0))])
lr.fit(Xtr, ytr); evaluate('LogReg', lr.predict_proba(Xte)[:,1])

# Random Forest
rf = RandomForestClassifier(n_estimators=400, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(Xtr, ytr); evaluate('RandomForest', rf.predict_proba(Xte)[:,1])

# XGBoost
xgb_clf = xgb.XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9, eval_metric='auc',
                            tree_method='hist', random_state=42)
xgb_clf.fit(Xtr, ytr); evaluate('XGBoost', xgb_clf.predict_proba(Xte)[:,1])

# LightGBM
lgb_clf = lgb.LGBMClassifier(n_estimators=600, max_depth=-1, num_leaves=63,
                             learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                             random_state=42, verbose=-1)
lgb_clf.fit(Xtr, ytr); evaluate('LightGBM', lgb_clf.predict_proba(Xte)[:,1])

best_name = max(results, key=lambda k: results[k]['AUC'])
print(f'Best baseline: {best_name}')

# ---------- 7. Tuning best (lgb assumed strong; tune via RandomizedSearchCV) ----------
print('Tuning LightGBM…')
param_dist = {
    'n_estimators':[300, 500, 800],
    'num_leaves':[31, 63, 127],
    'learning_rate':[0.02, 0.05, 0.1],
    'subsample':[0.7, 0.85, 1.0],
    'colsample_bytree':[0.7, 0.85, 1.0],
    'min_child_samples':[10, 20, 50],
    'reg_alpha':[0.0, 0.1, 1.0],
    'reg_lambda':[0.0, 0.1, 1.0],
}
rs = RandomizedSearchCV(lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=4),
                        param_dist, n_iter=12, cv=3, scoring='roc_auc',
                        n_jobs=2, random_state=42)
rs.fit(Xtr, ytr)
tuned = rs.best_estimator_
proba_best = tuned.predict_proba(Xte)[:,1]
auc_best, _ = evaluate('LightGBM_tuned', proba_best)
results['best_params'] = rs.best_params_

# Pick true best
candidates = {'LogReg':lr,'RandomForest':rf,'XGBoost':xgb_clf,'LightGBM_tuned':tuned}
best_name = max(candidates, key=lambda k: results[k]['AUC'])
best_model = candidates[best_name]
proba_best = best_model.predict_proba(Xte)[:,1]
print(f'>>> FINAL model: {best_name} AUC={results[best_name]["AUC"]:.4f}')
results['final_model'] = best_name

# ---------- 8. Diagnostic plots ----------
# ROC curves (all models)
fig, ax = plt.subplots(figsize=(6,5))
for model, name in [(lr,'LogReg'),(rf,'RandomForest'),(xgb_clf,'XGBoost'),(tuned,'LightGBM_tuned')]:
    p = model.predict_proba(Xte)[:,1]
    fpr, tpr, _ = roc_curve(yte, p)
    auc = roc_auc_score(yte, p)
    ax.plot(fpr, tpr, label=f'{name} AUC={auc:.3f}')
ax.plot([0,1],[0,1], 'k--', alpha=.4)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('Curvas ROC – test'); ax.legend()
fig.tight_layout(); fig.savefig(ART/'model_roc.png', dpi=130); plt.close(fig)

# Calibration
fig, ax = plt.subplots(figsize=(6,5))
prob_true, prob_pred = calibration_curve(yte, proba_best, n_bins=10)
ax.plot(prob_pred, prob_true, marker='o', label=best_name)
ax.plot([0,1],[0,1], 'k--', alpha=.4)
ax.set_xlabel('Prob. predicha'); ax.set_ylabel('Prob. observada')
ax.set_title('Calibración – modelo final'); ax.legend()
fig.tight_layout(); fig.savefig(ART/'model_calibration.png', dpi=130); plt.close(fig)

# Feature importance from final model (RF or LGB both have feature_importances_)
imp = pd.Series(best_model.feature_importances_, index=FEATCOLS).sort_values()
fig, ax = plt.subplots(figsize=(7,4.5))
imp.plot(kind='barh', ax=ax, color='#1B998B')
ax.set_title(f'Importancia variables – {best_name}'); ax.set_xlabel('importance')
fig.tight_layout(); fig.savefig(ART/'model_importance.png', dpi=130); plt.close(fig)

# Decile gains (business view)
df_te = pd.DataFrame({'p':proba_best, 'y':yte})
df_te['decile'] = pd.qcut(df_te['p'], 10, labels=False, duplicates='drop')
gains = df_te.groupby('decile')['y'].mean().sort_index(ascending=False)*100
fig, ax = plt.subplots(figsize=(7,4))
gains.plot(kind='bar', ax=ax, color='#F18F01')
ax.set_xlabel('Decil (10=más alto score)'); ax.set_ylabel('% que recompra')
ax.set_title('Tasa real de recompra por decil de score')
fig.tight_layout(); fig.savefig(ART/'model_deciles.png', dpi=130); plt.close(fig)

# ---------- 9. Save metrics ----------
results['feature_importance'] = imp.sort_values(ascending=False).round(3).to_dict()
results['quality'] = quality
results['cutoff'] = str(CUTOFF.date())
results['label_window_end'] = str(LABEL_END.date())
results['n_customers'] = int(len(feat))
results['repurchase_rate'] = float(feat['Repurchase'].mean())
results['top_decile_rate'] = float(gains.iloc[0]/100)
results['bottom_decile_rate'] = float(gains.iloc[-1]/100)

with open(ART/'metrics.json','w') as f:
    json.dump(results, f, indent=2, default=str)
print('Done. Artifacts in', ART)
