"""
Genera el diagrama de flujo del pipeline en estilo Mermaid usando Graphviz.
Exporta: pipeline_flowchart.png
"""
import graphviz

MERMAID_SOURCE = """
flowchart TD
    A([🗂 product_sales.csv\\n227 K filas]) --> B
    G([🗺 geo.csv\\n10 K filas])             --> B

    B[/① LOAD DATA\\nload_data()/]           --> C

    C[② PREPROCESS\\npreprocess + _assign_city\\n─────────────────\\nCast tipos · outlier 9999\\nJoin rango zipcode · 94.9% match\\n177 K filas limpias]  --> D

    D[③ FEATURE ENGINEERING\\nfeature_engineering\\n─────────────────\\n2 875 combos · 17 features\\nSerie semanal · lags 1-4w\\nrolls · trend · cv · zero_rate\\nTarget: log1p weekly_sales] --> E

    E{--skip-tuning?} -->|No| F
    E                 -->|Sí | H

    F[④ TUNE HYPERPARAMETERS\\ntune_hyperparameters\\n─────────────────\\nOptuna TPE · N_TRIALS=100\\nWalk-forward single split\\nObjetivo: minimizar WMAPE\\nEspacio: 9 hiperparámetros] --> H

    H[⑤ TRAIN\\ntrain_model\\n─────────────────\\nLGBMRegressor global\\n100% datos de entrenamiento\\nLog feature importances] --> I

    I[⑥ EVALUATE\\nevaluate\\n─────────────────\\nÚltimas VALIDATION_WEEKS semanas\\nPredicción directa con modelo\\nMétricas por tier:\\nsparse · medium · dense\\nMAE · MAPE · WMAPE · Bias · Accuracy] --> J

    J[⑦ PREDICT\\npredict + _build_prediction_features\\n─────────────────\\nForecast: 2024-08-08 · 09 · 10\\nVentana 7 días por fecha\\n≥3 sem → LightGBM + dow_index\\n<3 sem → fallback roll_2w] --> K

    K([📄 submission.csv\\n8 625 filas])
"""

PALETTE = {
    "csv":    ("#1B6E9B", "white"),   # azul — archivos entrada
    "load":   ("#1A3A5C", "white"),   # azul oscuro — ingest
    "prep":   ("#1A3A5C", "white"),   # azul oscuro — preprocess
    "feat":   ("#1B6E4F", "white"),   # verde — feature eng
    "tune":   ("#7B3F00", "white"),   # marrón — tuning
    "train":  ("#7B3F00", "white"),   # marrón — train
    "eval":   ("#4A235A", "white"),   # púrpura — evaluación
    "pred":   ("#145A32", "white"),   # verde oscuro — predict
    "out":    ("#145A32", "white"),   # verde oscuro — output
    "branch": ("#555555", "white"),   # gris — decisión
}

dot = graphviz.Digraph(
    "pipeline_flowchart",
    format="png",
    graph_attr={
        "rankdir":   "TB",
        "splines":   "ortho",
        "nodesep":   "0.5",
        "ranksep":   "0.65",
        "bgcolor":   "#F7F9FC",
        "fontname":  "Helvetica",
        "pad":       "0.4",
        "dpi":       "180",
    },
    node_attr={
        "fontname":  "Helvetica",
        "fontsize":  "12",
        "style":     "filled,rounded",
        "margin":    "0.3,0.18",
    },
    edge_attr={
        "fontname":  "Helvetica",
        "fontsize":  "11",
        "color":     "#444444",
        "arrowsize": "0.8",
    },
)

# ── Inputs ────────────────────────────────────────────────────────────────────
for nid, lbl in [
    ("sales_csv", "product_sales.csv\n227 K filas"),
    ("geo_csv",   "geo.csv\n10 K filas"),
]:
    dot.node(nid, lbl,
             shape="cylinder",
             fillcolor="#2E7D9B", fontcolor="white",
             width="1.8", height="0.7")

# ── Etapas principales ────────────────────────────────────────────────────────
stages = [
    ("load",   "① LOAD DATA\nload_data()",
     "box",      "#1A3A5C"),
    ("prep",   "② PREPROCESS\npreprocess()  +  _assign_city()\n──────────────────────────\n"
               "Cast tipos  ·  outlier 9999\n"
               "Join rango zipcode  ·  94.9% match\n"
               "177 K filas limpias",
     "box",      "#1A3A5C"),
    ("feat",   "③ FEATURE ENGINEERING\nfeature_engineering()\n──────────────────────────\n"
               "2 875 combos  ·  17 features\n"
               "Lags 1-4w  ·  rolls 2,4w\n"
               "trend  ·  cv  ·  zero_rate  ·  max  ·  growth\n"
               "Target: log(1 + weekly_sales)",
     "box",      "#1B6E4F"),
    ("skip",   "--skip-tuning?",
     "diamond",  "#555555"),
    ("tune",   "④ TUNE HYPERPARAMETERS\ntune_hyperparameters()\n──────────────────────────\n"
               "Optuna TPE  ·  N_TRIALS = 100\n"
               "Walk-forward single split\n"
               "Objetivo: minimizar WMAPE\n"
               "Espacio: 9 hiperparámetros",
     "box",      "#7B3F00"),
    ("train",  "⑤ TRAIN\ntrain_model()\n──────────────────────────\n"
               "LGBMRegressor global\n"
               "100% datos de entrenamiento\n"
               "Log feature importances (top 8)",
     "box",      "#7B3F00"),
    ("eval",   "⑥ EVALUATE\nevaluate()\n──────────────────────────\n"
               "Últimas VALIDATION_WEEKS semanas\n"
               "Predicción directa con modelo entrenado\n"
               "Métricas por tier (sparse · medium · dense):\n"
               "MAE  ·  MAPE  ·  WMAPE  ·  Bias  ·  Accuracy",
     "box",      "#4A235A"),
    ("pred",   "⑦ PREDICT\npredict()  +  _build_prediction_features()\n──────────────────────────\n"
               "Forecast: 2024-08-08  ·  09  ·  10\n"
               "Ventana de 7 días por fecha\n"
               "≥ MIN_WEEKS → LightGBM  +  dow_index\n"
               "< MIN_WEEKS → fallback roll_2w / lag_1w",
     "box",      "#145A32"),
]

shape_map = {"box": "box", "diamond": "diamond"}

for sid, label, shape, color in stages:
    dot.node(
        sid, label,
        shape=shape_map[shape],
        fillcolor=color,
        fontcolor="white",
        width="4.5" if shape == "box" else "2.0",
    )

# ── Output ────────────────────────────────────────────────────────────────────
dot.node("out_csv", "submission.csv\n8 625 filas",
         shape="cylinder",
         fillcolor="#145A32", fontcolor="white",
         width="1.8", height="0.7")

# ── Edges ─────────────────────────────────────────────────────────────────────
dot.edge("sales_csv", "load")
dot.edge("geo_csv",   "load")
dot.edge("load",  "prep")
dot.edge("prep",  "feat")
dot.edge("feat",  "skip")
dot.edge("skip",  "tune",  label="  No",  color="#7B3F00", fontcolor="#7B3F00")
dot.edge("skip",  "train", label="  Sí\n(default params)",
         color="#888888", fontcolor="#888888",
         style="dashed")
dot.edge("tune",  "train")
dot.edge("train", "eval")
dot.edge("eval",  "pred")
dot.edge("pred",  "out_csv")

# Agrupar inputs en el mismo rango vertical
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("sales_csv")
    s.node("geo_csv")

out_path = dot.render("pipeline_flowchart", cleanup=True)
print(f"Guardado: {out_path}")
