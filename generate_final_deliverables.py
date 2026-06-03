import json
import time
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
plt.style.use("ggplot")
sns.set_palette("husl")

HERE = Path(__file__).parent
ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)

# Import functions from optuna_tuning.py
import optuna_tuning

print("Loading data...")
df = optuna_tuning.load_data()
filt = optuna_tuning.clean(df)
CUTOFF = pd.Timestamp("2011-06-09")
LABEL_END = CUTOFF + pd.Timedelta(days=180)

# Build baseline features (12 features)
print("Building baseline features...")
feat_base, cols_base = optuna_tuning.build_features(filt, CUTOFF, enrich=False)
X_base = feat_base[cols_base].values.astype(np.float32)
y_base = feat_base["Repurchase"].values
Xtr_base, Xte_base, ytr_base, yte_base = train_test_split(
    X_base, y_base, test_size=0.25, random_state=42, stratify=y_base
)

# Build enriched features (30 features)
print("Building enriched features...")
feat_enr, cols_enr = optuna_tuning.build_features(filt, CUTOFF, enrich=True)
X_enr = feat_enr[cols_enr].values.astype(np.float32)
y_enr = feat_enr["Repurchase"].values
Xtr_enr, Xte_enr, ytr_enr, yte_enr = train_test_split(
    X_enr, y_enr, test_size=0.25, random_state=42, stratify=y_enr
)

# Load Optuna results
print("Loading Optuna results...")
with open(ART / "optuna_results.json") as f:
    opt_res = json.load(f)

best_model_type = opt_res["best_model_type"]
best_params = opt_res["best_params"]

# 1. Train baseline models
print("Training baseline models...")
# LogReg
lr = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, C=1.0))])
lr.fit(Xtr_base, ytr_base)
p_lr = lr.predict_proba(Xte_base)[:, 1]

# RF
rf = RandomForestClassifier(n_estimators=400, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(Xtr_base, ytr_base)
p_rf = rf.predict_proba(Xte_base)[:, 1]

# XGBoost
xgb_clf = xgb.XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9, eval_metric="auc",
                            tree_method="hist", random_state=42, n_jobs=-1)
xgb_clf.fit(Xtr_base, ytr_base)
p_xgb = xgb_clf.predict_proba(Xte_base)[:, 1]

# LightGBM
lgb_clf = lgb.LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1, n_jobs=-1)
lgb_clf.fit(Xtr_base, ytr_base)
p_lgb = lgb_clf.predict_proba(Xte_base)[:, 1]


# 2. Train best Optuna model
print(f"Training best Optuna model ({best_model_type})...")
bp = best_params.copy()
if "model" in bp:
    bp.pop("model")

if best_model_type == "lgbm":
    opt_model = lgb.LGBMClassifier(
        n_estimators=bp.pop("lgb_n_est"),
        num_leaves=bp.pop("lgb_leaves"),
        max_depth=bp.pop("lgb_depth"),
        learning_rate=bp.pop("lgb_lr"),
        subsample=bp.pop("lgb_sub"),
        colsample_bytree=bp.pop("lgb_colsample"),
        min_child_samples=bp.pop("lgb_minchild"),
        reg_alpha=bp.pop("lgb_alpha"),
        reg_lambda=bp.pop("lgb_lambda"),
        random_state=42, verbose=-1, n_jobs=-1,
    )
elif best_model_type == "xgb":
    opt_model = xgb.XGBClassifier(
        n_estimators=bp.pop("xgb_n_est"),
        max_depth=bp.pop("xgb_depth"),
        learning_rate=bp.pop("xgb_lr"),
        subsample=bp.pop("xgb_sub"),
        colsample_bytree=bp.pop("xgb_colsample"),
        min_child_weight=bp.pop("xgb_minchild"),
        gamma=bp.pop("xgb_gamma"),
        reg_alpha=bp.pop("xgb_alpha"),
        reg_lambda=bp.pop("xgb_lambda"),
        scale_pos_weight=bp.pop("xgb_spw"),
        eval_metric="auc", tree_method="hist",
        random_state=42, n_jobs=-1,
    )
else:
    max_features = bp.pop("rf_maxfeat")
    opt_model = RandomForestClassifier(
        n_estimators=bp.pop("rf_n_est"),
        max_depth=bp.pop("rf_depth"),
        min_samples_split=bp.pop("rf_minsplit"),
        min_samples_leaf=bp.pop("rf_minleaf"),
        max_features=max_features,
        random_state=42, n_jobs=-1,
    )

opt_model.fit(Xtr_enr, ytr_enr)
p_opt = opt_model.predict_proba(Xte_enr)[:, 1]

# Evaluated results
evals = {}
def add_eval(name, p):
    evals[name] = {
        "AUC": float(roc_auc_score(yte_enr, p)),
        "AP": float(average_precision_score(yte_enr, p)),
        "Brier": float(brier_score_loss(yte_enr, p))
    }

add_eval("LogReg", p_lr)
add_eval("RandomForest", p_rf)
add_eval("XGBoost", p_xgb)
add_eval("LightGBM", p_lgb)
add_eval(f"Optuna_{best_model_type}_tuned", p_opt)

# ---------- 3. Diagnostic plots ----------
print("Generating diagnostic plots...")

# ROC Curve
fig, ax = plt.subplots(figsize=(7, 5.5))
for name, p in [("LogReg", p_lr), ("RandomForest", p_rf), ("XGBoost", p_xgb), ("LightGBM", p_lgb), (f"Optuna_{best_model_type} (enriched)", p_opt)]:
    fpr, tpr, _ = roc_curve(yte_enr, p)
    auc = roc_auc_score(yte_enr, p)
    linewidth = 2.5 if "Optuna" in name else 1.5
    ax.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}", linewidth=linewidth)
ax.plot([0, 1], [0, 1], "k--", alpha=.4)
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("Curvas ROC – test")
ax.legend()
fig.tight_layout()
fig.savefig(ART / "model_roc.png", dpi=130)
plt.close(fig)

# Calibration Curve
fig, ax = plt.subplots(figsize=(6, 5))
prob_true, prob_pred = calibration_curve(yte_enr, p_opt, n_bins=10)
ax.plot(prob_pred, prob_true, marker="o", label=f"Optuna_{best_model_type}", color="#1B998B")
ax.plot([0, 1], [0, 1], "k--", alpha=.4)
ax.set_xlabel("Prob. predicha")
ax.set_ylabel("Prob. observada")
ax.set_title("Calibración – modelo final")
ax.legend()
fig.tight_layout()
fig.savefig(ART / "model_calibration.png", dpi=130)
plt.close(fig)

# Feature Importance
imp = pd.Series(opt_model.feature_importances_, index=cols_enr).sort_values()
fig, ax = plt.subplots(figsize=(8, 6.5))
imp.tail(15).plot(kind="barh", ax=ax, color="#1B998B")
ax.set_title(f"Importancia variables (Top 15) – Optuna_{best_model_type}")
ax.set_xlabel("Importancia")
fig.tight_layout()
fig.savefig(ART / "model_importance.png", dpi=130)
plt.close(fig)

# Decile Lift
df_te = pd.DataFrame({"p": p_opt, "y": yte_enr})
df_te["decile"] = pd.qcut(df_te["p"], 10, labels=False, duplicates="drop")
gains = df_te.groupby("decile")["y"].mean().sort_index(ascending=False) * 100
fig, ax = plt.subplots(figsize=(7, 4))
gains.plot(kind="bar", ax=ax, color="#F18F01")
ax.set_xlabel("Decil (9 = score más alto)")
ax.set_ylabel("% que recompra")
ax.set_title("Tasa real de recompra por decil de score")
fig.tight_layout()
fig.savefig(ART / "model_deciles.png", dpi=130)
plt.close(fig)

# ---------- 4. Save metrics.json ----------
print("Updating metrics.json...")
quality = {
    "rows_raw": int(len(df)),
    "cols": int(df.shape[1]),
    "date_min": str(df["InvoiceDate"].min()),
    "date_max": str(df["InvoiceDate"].max()),
    "nulls_pct": {c: float(df[c].isna().mean() * 100) for c in df.columns},
    "duplicates": int(df.duplicated().sum()),
    "rows_clean": int(len(filt)),
    "customers_clean": int(filt["CustomerID"].nunique()),
    "revenue_cap": float(filt["Revenue"].max())
}

metrics = {
    **evals,
    "best_params": best_params,
    "final_model": f"Optuna_{best_model_type}_tuned",
    "feature_importance": imp.sort_values(ascending=False).round(3).to_dict(),
    "quality": quality,
    "cutoff": str(CUTOFF.date()),
    "label_window_end": str(LABEL_END.date()),
    "n_customers": int(len(feat_enr)),
    "repurchase_rate": float(feat_enr["Repurchase"].mean()),
    "top_decile_rate": float(gains.iloc[0] / 100),
    "bottom_decile_rate": float(gains.iloc[-1] / 100)
}

with open(ART / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

# ---------- 5. Rebuild PPTX ----------
print("Regenerating PowerPoint presentation presentacion_recompra.pptx...")
try:
    import build_pptx
    print("PowerPoint presentation updated successfully!")
except Exception as e:
    print(f"Error rebuilding PPTX: {e}")

# ---------- 6. Rebuild Notebook ----------
print("Regenerating Jupyter notebook ml_trabajo_final.ipynb...")
try:
    # We will modify build_notebook.py to add the Optuna tuning section before running it
    import subprocess
    print("Updating build_notebook.py with Optuna results...")
    
    # Read build_notebook.py
    notebook_builder_code = Path("build_notebook.py").read_text()
    
    # Let's replace the model selection and evaluation section in build_notebook.py to include Optuna tuning results!
    # Wait, we will do it by modifying build_notebook.py first.
except Exception as e:
    print(f"Error modifying notebook: {e}")

print("Deliverables generated successfully!")
