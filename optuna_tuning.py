"""Optuna hyperparameter tuning for the repurchase model.

Adds enriched features on top of the existing RFM set and runs a proper
Bayesian search (TPE) across LightGBM, XGBoost, and RandomForest.

Usage:
    python optuna_tuning.py                  # 200 trials, enriched features
    python optuna_tuning.py --n-trials 500   # more trials
    python optuna_tuning.py --baseline-only   # existing 12 features, no enrichment
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = Path(__file__).parent
ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)


# ── 1. Data loading (identical to build_pipeline.py) ──────────────────────

def load_data():
    PKL = HERE / "online_retail.pkl"
    XLSX = HERE / "online_retail_II.xlsx"
    try:
        if PKL.exists():
            df = pd.read_pickle(PKL)
            return df
    except Exception:
        print("      (pickle incompatible, loading from xlsx — this takes ~30s)")
    xl = pd.ExcelFile(XLSX)
    df = pd.concat([pd.read_excel(XLSX, sheet_name=s) for s in xl.sheet_names],
                   ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={"Customer ID": "CustomerID"}, inplace=True)
    return df


def clean(df):
    df["InvoiceStr"] = df["Invoice"].astype(str)
    df["IsCancel"] = df["InvoiceStr"].str.startswith("C")
    df["StockStr"] = df["StockCode"].astype(str).str.upper()
    BAD_CODES = {
        "POST", "DOT", "M", "BANK CHARGES", "AMAZONFEE", "ADJUST", "D",
        "CRUK", "PADS", "C2", "B", "S", "TEST001", "TEST002", "GIFT",
        "SAMPLES", "m",
    }
    df["IsService"] = (
        df["StockStr"].isin(BAD_CODES)
        | df["StockStr"].str.contains(r"^[A-Z]+$", regex=True, na=False)
    )
    filt = df[
        (~df["IsCancel"])
        & (~df["IsService"])
        & (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & (df["CustomerID"].notna())
    ].copy()
    filt["Revenue"] = filt["Quantity"] * filt["Price"]
    cap = filt["Revenue"].quantile(0.995)
    filt = filt[filt["Revenue"] <= cap]
    filt["CustomerID"] = filt["CustomerID"].astype(int)
    return filt


# ── 2. Feature engineering ────────────────────────────────────────────────

def build_features(filt, cutoff, *, enrich=True):
    """Build customer-level features from transactions before cutoff.

    If enrich=True, adds ~10 new features on top of the baseline 12.
    """
    LABEL_END = cutoff + pd.Timedelta(days=180)
    train_tx = filt[filt["InvoiceDate"] < cutoff].copy()
    label_tx = filt[
        (filt["InvoiceDate"] >= cutoff) & (filt["InvoiceDate"] <= LABEL_END)
    ]

    repeaters = set(label_tx["CustomerID"].unique())
    custs = train_tx["CustomerID"].unique()
    y_df = pd.DataFrame({"CustomerID": custs})
    y_df["Repurchase"] = y_df["CustomerID"].isin(repeaters).astype(int)

    ref = cutoff
    g = train_tx.groupby("CustomerID")

    # ── Baseline 12 features (same as build_pipeline.py) ──
    feat = pd.DataFrame({
        "Recency":      (ref - g["InvoiceDate"].max()).dt.days,
        "Tenure":       (ref - g["InvoiceDate"].min()).dt.days,
        "Frequency":    g["Invoice"].nunique(),
        "NLines":       g.size(),
        "NProducts":    g["StockCode"].nunique(),
        "Monetary":     g["Revenue"].sum(),
        "AvgTicket":    g["Revenue"].sum() / g["Invoice"].nunique(),
        "AvgLine":      g["Revenue"].mean(),
        "AvgQty":       g["Quantity"].mean(),
        "StdRevenue":   g["Revenue"].std().fillna(0),
        "TopCountry":   g["Country"].agg(
            lambda s: s.mode().iat[0] if len(s) else "Unknown"
        ),
    }).reset_index()
    feat["IsUK"] = (feat["TopCountry"] == "United Kingdom").astype(int)
    feat["BuysPerMonth"] = feat["Frequency"] / (feat["Tenure"].clip(lower=1) / 30)

    if enrich:
        # ── Enriched features ──

        # Time-based patterns
        feat["DayOfWeekMode"] = g.apply(
            lambda x: x["InvoiceDate"].dt.dayofweek.mode().iat[0]
            if len(x) else 0
        ).values
        feat["HourMode"] = g.apply(
            lambda x: x["InvoiceDate"].dt.hour.mode().iat[0]
            if len(x) else 12
        ).values
        feat["WeekendRatio"] = g.apply(
            lambda x: (x["InvoiceDate"].dt.dayofweek >= 5).mean()
        ).values

        # Inter-purchase interval stats
        def _ipi_stats(grp):
            dates = grp["InvoiceDate"].sort_values().drop_duplicates()
            if len(dates) < 2:
                return pd.Series({"IPI_mean": 999, "IPI_std": 0, "IPI_cv": 0})
            diffs = dates.diff().dropna().dt.days
            mn = diffs.mean()
            sd = diffs.std() if len(diffs) > 1 else 0
            return pd.Series({
                "IPI_mean": mn,
                "IPI_std": sd,
                "IPI_cv": sd / mn if mn > 0 else 0,
            })

        ipi = g.apply(_ipi_stats).reset_index()
        feat = feat.merge(ipi, on="CustomerID", how="left")

        # Trend: is the customer buying more or less recently?
        # Split tenure into first-half vs second-half revenue
        def _trend(grp):
            if len(grp) < 2:
                return 0
            mid = grp["InvoiceDate"].min() + (grp["InvoiceDate"].max()
                                               - grp["InvoiceDate"].min()) / 2
            r1 = grp.loc[grp["InvoiceDate"] <= mid, "Revenue"].sum()
            r2 = grp.loc[grp["InvoiceDate"] > mid, "Revenue"].sum()
            total = r1 + r2
            return (r2 - r1) / total if total > 0 else 0

        feat["RevenueTrend"] = g.apply(_trend).values

        # Recency buckets (last 30/60/90 days purchase count)
        for window_days in [30, 60, 90]:
            window_start = ref - pd.Timedelta(days=window_days)
            recent = train_tx[train_tx["InvoiceDate"] >= window_start]
            rc = recent.groupby("CustomerID")["Invoice"].nunique().rename(
                f"Freq_last{window_days}d"
            )
            feat = feat.merge(rc, on="CustomerID", how="left")
            feat[f"Freq_last{window_days}d"] = feat[f"Freq_last{window_days}d"].fillna(0)

        # Product diversity ratio
        feat["ProductDiversity"] = feat["NProducts"] / feat["Frequency"].clip(lower=1)

        # Revenue per product
        feat["RevenuePerProduct"] = feat["Monetary"] / feat["NProducts"].clip(lower=1)

        # Interactions
        feat["Recency_x_Freq"] = feat["Recency"] * feat["Frequency"]
        feat["Monetary_per_Tenure"] = feat["Monetary"] / feat["Tenure"].clip(lower=1)

        # Country count (multi-country customers)
        feat["NCountries"] = g["Country"].nunique().values

        # Log transforms for skewed features
        for col in ["Monetary", "StdRevenue", "AvgTicket"]:
            feat[f"Log_{col}"] = np.log1p(feat[col])

    feat = feat.merge(y_df, on="CustomerID", how="inner")
    feat.drop(columns=["TopCountry"], inplace=True)

    feature_cols = [c for c in feat.columns if c not in ("CustomerID", "Repurchase")]
    return feat, feature_cols


def save_progress_callback(study, trial):
    # Get best trial details
    best_trial = study.best_trial
    progress = {
        "current_trial": trial.number + 1,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": best_trial.number,
        "elapsed_seconds": round(time.time() - study.user_attrs.get("start_time", time.time()), 1),
        "last_trial_value": trial.value,
        "last_trial_params": trial.params,
        "last_trial_model": trial.params.get("model", "unknown")
    }
    out_path = Path(__file__).parent / "artifacts" / "optuna_progress.json"
    try:
        with open(out_path, "w") as f:
            json.dump(progress, f, indent=2, default=str)
    except Exception as e:
        pass


# ── 3. Optuna objective ──────────────────────────────────────────────────

def create_objective(X, y, cv_folds=5):
    """Return an Optuna objective that picks a model family + hyperparameters."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def objective(trial):
        model_type = trial.suggest_categorical("model", ["lgbm", "xgb", "rf"])

        if model_type == "lgbm":
            params = {
                "n_estimators": trial.suggest_int("lgb_n_est", 100, 800),
                "num_leaves": trial.suggest_int("lgb_leaves", 15, 255),
                "max_depth": trial.suggest_int("lgb_depth", 3, 12),
                "learning_rate": trial.suggest_float("lgb_lr", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("lgb_sub", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("lgb_colsample", 0.4, 1.0),
                "min_child_samples": trial.suggest_int("lgb_minchild", 5, 100),
                "reg_alpha": trial.suggest_float("lgb_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("lgb_lambda", 1e-8, 10.0, log=True),
                "random_state": 42,
                "verbose": -1,
                "n_jobs": 1,
            }
            clf = lgb.LGBMClassifier(**params)

        elif model_type == "xgb":
            params = {
                "n_estimators": trial.suggest_int("xgb_n_est", 100, 800),
                "max_depth": trial.suggest_int("xgb_depth", 3, 10),
                "learning_rate": trial.suggest_float("xgb_lr", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("xgb_sub", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("xgb_minchild", 1, 50),
                "gamma": trial.suggest_float("xgb_gamma", 1e-8, 5.0, log=True),
                "reg_alpha": trial.suggest_float("xgb_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("xgb_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("xgb_spw", 0.5, 2.0),
                "eval_metric": "auc",
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 1,
            }
            clf = xgb.XGBClassifier(**params)

        else:  # rf
            params = {
                "n_estimators": trial.suggest_int("rf_n_est", 100, 500),
                "max_depth": trial.suggest_int("rf_depth", 5, 25),
                "min_samples_split": trial.suggest_int("rf_minsplit", 2, 30),
                "min_samples_leaf": trial.suggest_int("rf_minleaf", 1, 20),
                "max_features": trial.suggest_categorical(
                    "rf_maxfeat", ["sqrt", "log2", 0.5, 0.7]
                ),
                "random_state": 42,
                "n_jobs": 1,
            }
            clf = RandomForestClassifier(**params)

        # parallelize folds (n_jobs=cv_folds) to run fold evaluations in parallel.
        scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc", n_jobs=cv_folds)
        return scores.mean()

    return objective


# ── 4. Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for repurchase model")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of Optuna trials (default: 100)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Use only the original 12 features, no enrichment")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Optuna Hyperparameter Tuning — Repurchase Model")
    print("=" * 60)

    # Load and prepare data
    print("\n[1/5] Loading data...")
    df = load_data()
    filt = clean(df)
    CUTOFF = pd.Timestamp("2011-06-09")
    enrich = not args.baseline_only
    print(f"      Features: {'enriched (baseline + new)' if enrich else 'baseline only (12)'}")

    print("[2/5] Building features...")
    t0 = time.time()
    feat, feature_cols = build_features(filt, CUTOFF, enrich=enrich)
    print(f"      {len(feature_cols)} features | {len(feat)} customers | "
      f"repurchase rate {feat['Repurchase'].mean():.1%} | "
      f"built in {time.time()-t0:.1f}s")
    print(f"      Features: {feature_cols}")

    X = feat[feature_cols].values.astype(np.float32)
    y = feat["Repurchase"].values

    # Hold-out test set (same split as build_pipeline.py)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ── Optuna study ──
    print(f"\n[3/5] Running Optuna ({args.n_trials} trials, {args.cv_folds}-fold CV)...")
    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.set_user_attr("start_time", t0)
    objective = create_objective(Xtr, ytr, cv_folds=args.cv_folds)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=True,
        callbacks=[save_progress_callback]
    )
    elapsed = time.time() - t0

    print(f"\n      Done in {elapsed/60:.1f} min")
    print(f"      Best CV AUC: {study.best_value:.4f}")
    print(f"      Best params: {study.best_params}")

    # ── Retrain best model on full train set, evaluate on hold-out test ──
    print("\n[4/5] Retraining best model on full train set...")
    bp = study.best_params
    model_type = bp.pop("model")

    if model_type == "lgbm":
        best_clf = lgb.LGBMClassifier(
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
    elif model_type == "xgb":
        best_clf = xgb.XGBClassifier(
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
        best_clf = RandomForestClassifier(
            n_estimators=bp.pop("rf_n_est"),
            max_depth=bp.pop("rf_depth"),
            min_samples_split=bp.pop("rf_minsplit"),
            min_samples_leaf=bp.pop("rf_minleaf"),
            max_features=max_features,
            random_state=42, n_jobs=-1,
        )

    best_clf.fit(Xtr, ytr)
    proba = best_clf.predict_proba(Xte)[:, 1]
    test_auc = roc_auc_score(yte, proba)

    # ── Also try a stacking ensemble ──
    print("[4b]  Trying stacking ensemble...")
    stack_estimators = [
        ("lgbm", lgb.LGBMClassifier(
            n_estimators=500, num_leaves=63, learning_rate=0.05,
            random_state=42, verbose=-1, n_jobs=-1)),
        ("xgb", xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            eval_metric="auc", tree_method="hist",
            random_state=42, n_jobs=-1)),
        ("rf", RandomForestClassifier(
            n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)),
    ]
    stack = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, passthrough=False, n_jobs=1,
    )
    stack.fit(Xtr, ytr)
    proba_stack = stack.predict_proba(Xte)[:, 1]
    stack_auc = roc_auc_score(yte, proba_stack)

    # ── Results ──
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n  Previous best (RF, 12 features):     AUC = 0.8030")
    print(f"  Optuna best ({model_type}, CV):         AUC = {study.best_value:.4f}")
    print(f"  Optuna best ({model_type}, test):       AUC = {test_auc:.4f}")
    print(f"  Stacking ensemble (test):              AUC = {stack_auc:.4f}")
    print(f"  Improvement:                           +{max(test_auc, stack_auc) - 0.8030:.4f}")

    final_auc = max(test_auc, stack_auc)
    final_name = f"Optuna_{model_type}" if test_auc >= stack_auc else "Stacking"
    print(f"\n  >>> FINAL: {final_name}  AUC = {final_auc:.4f}")

    # ── Save results ──
    print("\n[5/5] Saving results...")
    results = {
        "previous_best_auc": 0.8030,
        "optuna_best_cv_auc": study.best_value,
        "optuna_best_test_auc": test_auc,
        "stacking_test_auc": stack_auc,
        "final_model": final_name,
        "final_auc": final_auc,
        "improvement": final_auc - 0.8030,
        "best_model_type": model_type,
        "best_params": study.best_params,
        "n_trials": args.n_trials,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "enriched": enrich,
        "elapsed_minutes": round(elapsed / 60, 1),
        "top_10_trials": [
            {"number": t.number, "value": round(t.value, 4), "params": t.params}
            for t in sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]
        ],
    }
    out_path = ART / "optuna_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"      Saved to {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
