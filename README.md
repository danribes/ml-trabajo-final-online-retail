# Predicción de recompra – Online Retail II

Trabajo final del módulo de Machine Learning. Construye un modelo de probabilidad de recompra a 6 meses para un retailer online UK y propone una segmentación accionable.

## Entregables

| Archivo | Descripción |
|---|---|
| `ml_trabajo_final.ipynb` | Notebook ejecutado de extremo a extremo (EDA → limpieza → features → modelos → tuning → diagnóstico → conclusiones). |
| `presentacion_recompra.pptx` | Presentación de 5 slides para negocio. |
| `presentacion/` | Deck técnico (Marp → `deck.pdf` tracked; `deck.pptx` / `deck.html` / `deck.marp.md` en `.gitignore`). |
| `build_pipeline.py` | Script reproducible que regenera todos los artefactos. |
| `build_notebook.py` | Genera el notebook a partir del pipeline. |
| `build_pptx.py` | Genera la pptx desde la plantilla. |
| `generate_final_deliverables.py` | Entrena todos los modelos base y genera artefactos finales. |
| `optuna_tuning.py` | Búsqueda bayesiana (TPE, 100 trials) sobre XGBoost/LightGBM/RF con features enriquecidas. Acepta `--n-trials` y `--baseline-only`. |
| `rebuild_pickle.py` | Reconstruye `online_retail.pkl` desde el Excel original (~30 s). |
| `artifacts/` | PNGs de EDA + modelo + `metrics.json` + `optuna_results.json`. |

## Dataset

Online Retail II (UCI ML Repository), ~1.07M transacciones de un retailer online UK entre 2009-12 y 2011-12. Descargar `online_retail_II.xlsx` desde:

https://archive.ics.uci.edu/dataset/502/online+retail+ii

Colocar en la raíz del repo. No se sube por tamaño (38 MB) y licencia.

## Reproducir

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl xgboost lightgbm optuna jupyter nbformat python-pptx

# (Opcional) reconstruir pickle desde Excel si no existe online_retail.pkl
python rebuild_pickle.py

# Entrenar modelos base + artefactos
python generate_final_deliverables.py

# Búsqueda bayesiana (100 trials por defecto, ~2 min)
python optuna_tuning.py

# Regenerar notebook y presentaciones
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace ml_trabajo_final.ipynb
python build_pptx.py            # regenera presentacion_recompra.pptx
```

## Decisiones de diseño

- **Cutoff:** 2011-06-09 (6 meses antes del fin del dataset).
- **Ventana de label:** 180 días post-cutoff.
- **Target:** 1 si el cliente vuelve a comprar al menos una vez en la ventana.
- **Exclusiones:** filas sin `CustomerID` (~23 %), cancelaciones (`Invoice` "C\*"), códigos de servicio (POST, DOT, AMAZONFEE, …), `Quantity ≤ 0`, `Price ≤ 0`, top 0.5 % revenue por línea (outliers).
- **Features (30):** RFM clásico + intensidad (`BuysPerMonth`, `Monetary_per_Tenure`, `StdRevenue`, `AvgQty`, `NProducts`, `Tenure`, `IsUK`) + recencia escalonada (`Freq_last30d/60d/90d`) + Inter-Purchase Interval (`IPI_mean`, `IPI_std`, `IPI_cv`) + tendencia de revenue (`RevenueTrend`) + diversidad de producto (`ProductDiversity`, `NCountries`, `RevenuePerProduct`) + interacción (`Recency_x_Freq`) + log-transforms (`Log_Monetary`, `Log_StdRevenue`, `Log_AvgTicket`) + patrones temporales (`DayOfWeekMode`, `HourMode`, `WeekendRatio`).

## Resultados

| | AUC | AP | Brier |
|---|---:|---:|---:|
| Logistic Regression | 0.796 | 0.820 | 0.184 |
| Random Forest | 0.804 | 0.834 | 0.179 |
| XGBoost | 0.790 | 0.823 | 0.194 |
| LightGBM | 0.778 | 0.812 | 0.228 |
| **XGBoost tuned – Optuna (final)** | **0.814** | **0.843** | **0.176** |

- Tuning: Optuna TPE, 100 trials, XGBoost depth-3, lr ≈ 0.009, 607 estimadores.
- Mejora vs. baseline RF: +1.1 pp AUC, +0.9 pp AP, −0.003 Brier.
- Top decil de score → **97.6 %** recompran (vs. 51.6 % base).
- Bottom decil → **11.4 %** recompran.
- Variables más explicativas (importancia XGB): `BuysPerMonth`, `Monetary_per_Tenure`, `Recency`, `Freq_last90d`, `Frequency`.

## Uso de negocio

| Decil score | Acción |
|---|---|
| 9–10 | Fidelización (programa VIP, acceso prioritario, NPS). |
| 5–8 | Cross-sell / cupones personalizados. |
| 0–4 | Retención: win-back con descuento agresivo + encuesta de churn. |
