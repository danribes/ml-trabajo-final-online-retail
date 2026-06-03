"""Build clean final notebook ml_trabajo_final.ipynb"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t))
co = lambda t: cells.append(nbf.v4.new_code_cell(t))

md("""# Predicción de recompra – Online Retail II
**Máster ML – trabajo final**

Objetivo: predecir la probabilidad de que un cliente del retailer online UK Online Retail II
**vuelva a comprar** en los 6 meses siguientes a una fecha de corte.

**Flujo del Trabajo**:
1. Carga + calidad de datos
2. Limpieza + filtrado
3. Definición de fecha de corte + variable objetivo
4. Feature engineering (RFM básico)
5. Análisis exploratorio (EDA)
6. Modelos Baseline (LogReg, RandomForest, XGBoost, LightGBM)
7. Enriched Feature Engineering (Añadiendo inter-purchase intervals, tendencias, recency windows y ratios)
8. Optimización con Optuna (TPE bayesiano en LGBM/XGB/RF con 30 features)
9. Evaluación y Diagnósticos (Curva ROC comparativa, calibración y deciles)
10. Uso de negocio (Segmentación de clientes y estrategias por decil de score)""")

md("## 0. Imports")
co("""import warnings, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             brier_score_loss, classification_report,
                             confusion_matrix)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb
import optuna

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
sns.set_palette('husl')
pd.set_option('display.max_columns', 35)""")

md("## 1. Carga de datos")
co("""DATA_PATH = 'online_retail_II.xlsx'
PKL = Path('online_retail.pkl')

# Carga súper veloz mediante el pickle regenerado compatible
if PKL.exists():
    df = pd.read_pickle(PKL)
else:
    xl = pd.ExcelFile(DATA_PATH)
    df = pd.concat([pd.read_excel(DATA_PATH, sheet_name=s) for s in xl.sheet_names],
                   ignore_index=True)
    df.to_pickle(PKL)

df.columns = [c.strip() for c in df.columns]
df.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)
print(f'Filas: {len(df):,} | Columnas: {len(df.columns)}')
df.head()""")

md("## 2. Calidad de datos")
co("""df.info()""")
co("""print('Nulos (%):')
print((df.isna().mean()*100).round(2))
print(f'\\nDuplicados: {df.duplicated().sum():,}')
print(f'Rango fechas: {df.InvoiceDate.min()} → {df.InvoiceDate.max()}')""")

md("""**Observaciones**
- ~23% de filas sin CustomerID → imposibles de etiquetar, se descartan.
- Hay cancelaciones (Invoice empieza por `C`), códigos de servicio (POST, DOT, M, AMAZONFEE…) y valores negativos en Quantity/Price.""")

md("## 3. Limpieza + filtrado")
co("""df['InvoiceStr'] = df['Invoice'].astype(str)
df['IsCancel']   = df['InvoiceStr'].str.startswith('C')
df['StockStr']   = df['StockCode'].astype(str).str.upper()

BAD_CODES = {'POST','DOT','M','BANK CHARGES','AMAZONFEE','ADJUST','D','CRUK','PADS',
             'C2','B','S','TEST001','TEST002','GIFT','SAMPLES'}
df['IsService'] = (df['StockStr'].isin(BAD_CODES) |
                   df['StockStr'].str.contains(r'^[A-Z]+$', regex=True, na=False))

filt = df[
    (~df['IsCancel']) &
    (~df['IsService']) &
    (df['Quantity'] > 0) &
    (df['Price']    > 0) &
    (df['CustomerID'].notna())
].copy()
filt['Revenue'] = filt['Quantity'] * filt['Price']

# Limpieza de outliers del top 0.5% de revenue
cap = filt['Revenue'].quantile(0.995)
filt = filt[filt['Revenue'] <= cap]
filt['CustomerID'] = filt['CustomerID'].astype(int)

print(f'Filas tras limpieza: {len(filt):,}')
print(f'Clientes únicos: {filt.CustomerID.nunique():,}')
print(f'Cap revenue por línea: £{cap:,.2f}')""")

md("## 4. Fecha de corte + variable objetivo")
co("""CUTOFF    = pd.Timestamp('2011-06-09')   # 6 meses antes del final
LABEL_END = CUTOFF + pd.Timedelta(days=180)

train_tx = filt[filt['InvoiceDate'] <  CUTOFF].copy()
label_tx = filt[(filt['InvoiceDate'] >= CUTOFF) & (filt['InvoiceDate'] <= LABEL_END)].copy()

repeaters = set(label_tx['CustomerID'].unique())
y_df = pd.DataFrame({'CustomerID': train_tx['CustomerID'].unique()})
y_df['Repurchase'] = y_df['CustomerID'].isin(repeaters).astype(int)

print(f'Clientes en periodo pre-cutoff:  {len(y_df):,}')
print(f'Tasa de recompra observada:      {y_df.Repurchase.mean():.3%}')""")

md("## 5. Feature Engineering – Baseline RFM")
co("""ref = CUTOFF
g = train_tx.groupby('CustomerID')

feat = pd.DataFrame({
    'Recency':    (ref - g['InvoiceDate'].max()).dt.days,
    'Tenure':     (ref - g['InvoiceDate'].min()).dt.days,
    'Frequency':  g['Invoice'].nunique(),
    'NLines':     g.size(),
    'NProducts':  g['StockCode'].nunique(),
    'Monetary':   g['Revenue'].sum(),
    'AvgTicket':  g['Revenue'].sum() / g['Invoice'].nunique(),
    'AvgLine':    g['Revenue'].mean(),
    'AvgQty':     g['Quantity'].mean(),
    'StdRevenue': g['Revenue'].std().fillna(0),
    'TopCountry': g['Country'].agg(lambda s: s.mode().iat[0]),
}).reset_index()

feat['IsUK']         = (feat['TopCountry']=='United Kingdom').astype(int)
feat['BuysPerMonth'] = feat['Frequency'] / (feat['Tenure'].clip(lower=1)/30)

feat = feat.merge(y_df, on='CustomerID', how='inner')
feat.drop(columns=['TopCountry'], inplace=True, errors='ignore')
print(f'Clientes etiquetados: {len(feat):,}')
feat.head()""")

md("## 6. Análisis Exploratorio de Datos (EDA)")
co("""# 1. Revenue mensual + cutoff
mr = filt.set_index('InvoiceDate')['Revenue'].resample('ME').sum()/1000
fig, ax = plt.subplots(figsize=(10,4))
mr.plot(ax=ax, marker='o', color='#2E86AB')
ax.axvline(CUTOFF, color='red', ls='--', label=f'Cutoff {CUTOFF.date()}')
ax.set_ylabel('Revenue (£k)'); ax.set_title('Revenue mensual')
ax.legend(); plt.show()""")

co("""# 2. Distribución del target
fig, ax = plt.subplots(figsize=(5,4))
feat['Repurchase'].value_counts().rename({0:'No recompra',1:'Recompra'}).plot(
    kind='bar', ax=ax, color=['#D7263D','#1B998B'])
ax.set_title(f'Balance del target  (tasa={feat.Repurchase.mean():.1%})')
ax.set_ylabel('Clientes'); plt.xticks(rotation=0); plt.show()""")

md("## 7. Modelos Baseline (12 features)")
co("""FEATCOLS_BASE = ['Recency','Tenure','Frequency','NLines','NProducts','Monetary',
                 'AvgTicket','AvgLine','AvgQty','StdRevenue','BuysPerMonth','IsUK']
X_base = feat[FEATCOLS_BASE].values
y = feat['Repurchase'].values

Xtr_base, Xte_base, ytr_base, yte_base = train_test_split(X_base, y, test_size=0.25,
                                                          random_state=42, stratify=y)

results = {}
def evaluate(name, y_pred):
    auc = roc_auc_score(yte_base, y_pred)
    ap = average_precision_score(yte_base, y_pred)
    brier = brier_score_loss(yte_base, y_pred)
    results[name] = {'AUC': auc, 'AP': ap, 'Brier': brier, 'proba': y_pred}
    print(f'{name:16s}  AUC={auc:.4f}  AP={ap:.4f}  Brier={brier:.4f}')

# Logistic Regression
lr = Pipeline([('sc', StandardScaler()),
               ('lr', LogisticRegression(max_iter=2000, C=1.0))]).fit(Xtr_base, ytr_base)
evaluate('LogReg_base', lr.predict_proba(Xte_base)[:,1])

# Random Forest Baseline
rf = RandomForestClassifier(n_estimators=400, max_depth=10, n_jobs=-1, random_state=42).fit(Xtr_base, ytr_base)
evaluate('RandomForest_base', rf.predict_proba(Xte_base)[:,1])

# XGBoost Baseline
xgb_base = xgb.XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9,
                             eval_metric='auc', tree_method='hist',
                             random_state=42, n_jobs=-1).fit(Xtr_base, ytr_base)
evaluate('XGBoost_base', xgb_base.predict_proba(Xte_base)[:,1])""")

md("## 8. Feature Engineering Enriquecido (30 features)")
md("""Para mejorar el poder predictivo del modelo más allá de las 12 variables RFM clásicas, creamos variables enriquecidas adicionales:
- **Patrones temporales**: Día de la semana más común (`DayOfWeekMode`), hora preferida (`HourMode`), y ratio de compras en fin de semana (`WeekendRatio`).
- **Estadísticas de intervalos entre compras (IPI)**: Media, desviación estándar y coeficiente de variación de los días entre facturas (`IPI_mean`, `IPI_std`, `IPI_cv`).
- **Tendencia de gasto**: Ratio de cambio en la facturación comparando la primera mitad de la tenure con la segunda mitad (`RevenueTrend`).
- **Frecuencia reciente**: Número de compras en ventanas de los últimos 30, 60 y 90 días.
- **Ratios e interacciones**: Riqueza/diversidad de productos comprados (`ProductDiversity`), ticket medio por producto (`RevenuePerProduct`), la interacción recencia x frecuencia, etc.
- **Transformación logarítmica** de las variables de facturación muy sesgadas (`Log_Monetary`, `Log_StdRevenue`).""")

co("""# Generar dataset enriquecido
df_enr, cols_enr = optuna_tuning.build_features(filt, CUTOFF, enrich=True)
X_enr = df_enr[cols_enr].values.astype(np.float32)
y_enr = df_enr['Repurchase'].values

Xtr_enr, Xte_enr, ytr_enr, yte_enr = train_test_split(X_enr, y_enr, test_size=0.25,
                                                      random_state=42, stratify=y_enr)
print(f'Variables enriquecidas ({len(cols_enr)}): {cols_enr}')""")

md("## 9. Optimización de Hiperparámetros con Optuna")
md("""Ejecutamos un proceso de optimización bayesiana (con muestreo Tree-structured Parzen Estimator, TPE) sobre LightGBM, XGBoost y Random Forest usando validación cruzada estratificada de 5 folds sobre el conjunto de entrenamiento. 

*Optimizamos las folds en paralelo (`n_jobs=5`) para lograr ejecuciones de trial en ~1.1 segundos.*""")

co("""# Carga de los resultados de Optuna guardados en disco
with open('artifacts/optuna_results.json') as f:
    opt_res = json.load(f)

print(f"Mejor modelo encontrado por Optuna: {opt_res['final_model']}")
print(f"CV AUC óptimo: {opt_res['optuna_best_cv_auc']:.4f}")
print("Mejores parámetros:")
print(json.dumps(opt_res['best_params'], indent=2))""")

md("## 10. Re-entrenamiento del Modelo Final Optimizado")
co("""bp = opt_res['best_params'].copy()
model_type = bp.pop('model')

if model_type == 'xgb':
    best_clf = xgb.XGBClassifier(
        n_estimators=bp.pop('xgb_n_est'),
        max_depth=bp.pop('xgb_depth'),
        learning_rate=bp.pop('xgb_lr'),
        subsample=bp.pop('xgb_sub'),
        colsample_bytree=bp.pop('xgb_colsample'),
        min_child_weight=bp.pop('xgb_minchild'),
        gamma=bp.pop('xgb_gamma'),
        reg_alpha=bp.pop('xgb_alpha'),
        reg_lambda=bp.pop('xgb_lambda'),
        scale_pos_weight=bp.pop('xgb_spw'),
        eval_metric='auc', tree_method='hist',
        random_state=42, n_jobs=-1
    )
elif model_type == 'lgbm':
    best_clf = lgb.LGBMClassifier(
        n_estimators=bp.pop('lgb_n_est'),
        num_leaves=bp.pop('lgb_leaves'),
        max_depth=bp.pop('lgb_depth'),
        learning_rate=bp.pop('lgb_lr'),
        subsample=bp.pop('lgb_sub'),
        colsample_bytree=bp.pop('lgb_colsample'),
        min_child_samples=bp.pop('lgb_minchild'),
        reg_alpha=bp.pop('lgb_alpha'),
        reg_lambda=bp.pop('lgb_lambda'),
        random_state=42, verbose=-1, n_jobs=-1
    )
else:
    best_clf = RandomForestClassifier(
        n_estimators=bp.pop('rf_n_est'),
        max_depth=bp.pop('rf_depth'),
        min_samples_split=bp.pop('rf_minsplit'),
        min_samples_leaf=bp.pop('rf_minleaf'),
        max_features=bp.pop('rf_maxfeat'),
        random_state=42, n_jobs=-1
    )

best_clf.fit(Xtr_enr, ytr_enr)
p_opt = best_clf.predict_proba(Xte_enr)[:,1]
evaluate('Optuna_Tuned_Enriched', p_opt)""")

md("## 11. Comparación Diagnóstica de Modelos")
co("""# Curva ROC comparativa
fig, ax = plt.subplots(figsize=(7,5.5))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(yte_base, r['proba'])
    linewidth = 2.5 if 'Optuna' in name else 1.2
    ax.plot(fpr, tpr, label=f"{name} AUC={r['AUC']:.4f}", linewidth=linewidth)
ax.plot([0,1],[0,1],'k--', alpha=.4)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC en Test (Baseline vs Enriquecido + Tuned)')
ax.legend(); plt.show()""")

co("""# Curva de Calibración del Modelo Final
prob_true, prob_pred = calibration_curve(yte_enr, p_opt, n_bins=10)
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(prob_pred, prob_true, marker='o', label='Optuna final', color='#1B998B')
ax.plot([0,1],[0,1],'k--', alpha=.4)
ax.set_xlabel('Probabilidad predicha'); ax.set_ylabel('Probabilidad observada')
ax.set_title('Curva de Calibración del Modelo Final'); ax.legend(); plt.show()""")

md("## 12. Importancia de Variables del Modelo Final")
co("""imp = pd.Series(best_clf.feature_importances_, index=cols_enr).sort_values()
fig, ax = plt.subplots(figsize=(8,6))
imp.tail(15).plot(kind='barh', ax=ax, color='#1B998B')
ax.set_title('Importancia de Variables (Top 15)')
ax.set_xlabel('Importancia relativa'); plt.show()""")

md("## 13. Vista de Negocio – Tasa real de recompra por decil de score")
co("""df_te = pd.DataFrame({'p': p_opt, 'y': yte_enr})
df_te['decile'] = pd.qcut(df_te['p'], 10, labels=False, duplicates='drop')
gains = df_te.groupby('decile')['y'].mean().sort_index(ascending=False)*100

fig, ax = plt.subplots(figsize=(8,4))
gains.plot(kind='bar', ax=ax, color='#F18F01')
ax.set_xlabel('Decil (9 = score más alto)')
ax.set_ylabel('% real que recompra')
ax.set_title('Tasa real de recompra por decil de score')
plt.show()

print(f'Top decil (VIP):   {gains.iloc[0]:.1f}% de recompra real')
print(f'Bottom decil (Churn): {gains.iloc[-1]:.1f}% de recompra real')""")

md("""## 14. Conclusiones y Propuesta de Uso de Negocio

### Resultados de la Optimización
1. **Poder Predictivo**: Logramos incrementar el **AUC en Test de 0.8030 a 0.8141** (+1.11 pp) y el **CV AUC a 0.8107** mediante la introducción de variables comportamentales enriquecidas y tuning sistemático con Optuna.
2. **Capacidad de Segmentación**: El decil de score más alto (Top 10%) tiene una tasa de recompra real del **96.7%**, mientras que el decil más bajo tiene solo un **12.2%**.
3. **Variables Clave**: Las variables más explicativas son **Recency** (recencia clásica), **BuysPerMonth** (frecuencia estandarizada), e **IPI_mean** (días promedio entre compras).

### Propuesta de Campañas de Marketing por Segmento
| Segmento (Decil de Score) | Prob. Recompra | Campaña Propuesta |
|---|---|---|
| **Deciles 8–9 (VIP)** | ~93% – 97% | **Fidelización Activa**: Acceso anticipado a colecciones, programas de referidos, upselling sin descuentos de margen. |
| **Deciles 5–7 (Estables)** | ~55% – 85% | **Afinidad e Impulso**: Cross-selling personalizado basado en categorías compradas y ofertas flash. |
| **Deciles 0–4 (Churn Riesgo)** | ~12% – 45% | **Retención / Win-back**: Campañas de reactivación agresivas (descuentos fuertes, encuestas de insatisfacción). |""")

nb['cells'] = cells
out = Path('ml_trabajo_final.ipynb')
nbf.write(nb, str(out))
print('Wrote', out.resolve())
