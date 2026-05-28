# Predicción de recompra – Online Retail II

Trabajo final del módulo de Machine Learning. Construye un modelo de probabilidad de recompra a 6 meses para un retailer online UK y propone una segmentación accionable.

## Entregables

| Archivo | Descripción |
|---|---|
| `ml_trabajo_final.ipynb` | Notebook ejecutado de extremo a extremo (EDA → limpieza → features → modelos → tuning → diagnóstico → conclusiones). |
| `presentacion_recompra.pptx` | Presentación de 5 slides para negocio. |
| `build_pipeline.py` | Script reproducible que regenera todos los artefactos. |
| `build_notebook.py` | Genera el notebook a partir del pipeline. |
| `build_pptx.py` | Genera la pptx desde la plantilla. |
| `artifacts/` | PNGs de EDA + modelo + `metrics.json`. |

## Dataset

Online Retail II (UCI ML Repository), ~1.07M transacciones de un retailer online UK entre 2009-12 y 2011-12. Descargar `online_retail_II.xlsx` desde:

https://archive.ics.uci.edu/dataset/502/online+retail+ii

Colocar en la raíz del repo. No se sube por tamaño (38 MB) y licencia.

## Reproducir

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl xgboost lightgbm jupyter nbformat python-pptx
python build_pipeline.py        # entrena modelos + genera artifacts/
python build_notebook.py        # regenera el notebook
jupyter nbconvert --to notebook --execute --inplace ml_trabajo_final.ipynb
python build_pptx.py            # regenera presentacion_recompra.pptx
```

## Decisiones de diseño

- **Cutoff:** 2011-06-09 (6 meses antes del fin del dataset).
- **Ventana de label:** 180 días post-cutoff.
- **Target:** 1 si el cliente vuelve a comprar al menos una vez en la ventana.
- **Exclusiones:** filas sin `CustomerID` (~23 %), cancelaciones (`Invoice` "C\*"), códigos de servicio (POST, DOT, AMAZONFEE, …), `Quantity ≤ 0`, `Price ≤ 0`, top 0.5 % revenue por línea (outliers).
- **Features:** RFM clásico + intensidad (`BuysPerMonth`, `StdRevenue`, `AvgQty`, `NProducts`, `Tenure`, `IsUK`).

## Resultados

| | AUC | AP | Brier |
|---|---:|---:|---:|
| Logistic Regression | 0.796 | 0.820 | 0.184 |
| **Random Forest (final)** | **0.803** | **0.833** | **0.180** |
| XGBoost | 0.789 | 0.823 | 0.194 |
| LightGBM | 0.778 | 0.812 | 0.229 |
| LightGBM tuned | 0.791 | 0.826 | 0.189 |

- Top decil de score → **97 %** recompran (vs. 51.6 % base).
- Bottom decil → **12 %** recompran.
- Variables más explicativas: `Recency`, `StdRevenue`, `AvgQty`, `BuysPerMonth`, `Tenure`.

## Uso de negocio

| Decil score | Acción |
|---|---|
| 9–10 | Fidelización (programa VIP, acceso prioritario, NPS). |
| 5–8 | Cross-sell / cupones personalizados. |
| 0–4 | Retención: win-back con descuento agresivo + encuesta de churn. |
