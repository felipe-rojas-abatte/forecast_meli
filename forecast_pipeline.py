"""
forecast_pipeline.py
═══════════════════════════════════════════════════════════════════════════════
Pipeline completo de forecast de demanda semanal por product_id × city.
MercadoLibre — Technical Assessment

Arquitectura
------------
    1. load_data             → carga de archivos CSV
    2. preprocess            → limpieza, tipos, join geográfico
    3. feature_engineering   → serie semanal + features para LightGBM
    4. tune_hyperparameters  → Optuna con walk-forward objective
    5. train_model           → LightGBM global (un solo modelo para todos los combos)
    6. evaluate              → MAE/MAPE por tier de densidad vs. baseline
    7. predict               → genera submission.csv con las 3 fechas de forecast

Supuestos del modelo
--------------------
  - El valor 9999 en 'sales' es un outlier del sistema fuente, no una venta real.
  - El join geográfico se hace por rango: s_zipcode ≤ zipcode ≤ e_zipcode, por país.
  - El target de forecast es la SUMA de ventas de 7 días a partir de cada fecha.
  - Se entrena UN modelo global sobre todos los combos (maneja cold start via features
    de contexto: n_weeks, days_observed, city_freq, etc.).
  - Combos con ≥ MIN_WEEKS_FOR_LGBM semanas históricas → LightGBM.
  - Combos con < MIN_WEEKS_FOR_LGBM semanas → fallback: media ponderada reciente.
  - El target se modela en escala log(1+x) para comprimir la distribución heavy-tail.
  - Validación: walk-forward sobre las últimas VALIDATION_WEEKS semanas del dataset.

Uso
---
  python forecast_pipeline.py                          # pipeline completo con Optuna
  python forecast_pipeline.py --skip-tuning            # omite Optuna, params por defecto
  python forecast_pipeline.py --trials 50              # más trials de Optuna
  python forecast_pipeline.py --data-dir mi/ruta/data  # ruta personalizada

Requisitos
----------
  pip install lightgbm optuna scikit-learn pandas numpy
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR           = Path("data")
OUTPUT_DIR         = Path("output")

SENTINEL_VALUE     = 9999        # valor outlier a eliminar de 'sales'
LAG_WEEKS          = [1, 2, 3, 4]   # semanas de lag usadas como features
ROLLING_WINDOWS    = [2, 4]      # ventanas de promedio móvil (semanas)
N_OPTUNA_TRIALS    = 100          # número de trials en búsqueda bayesiana
VALIDATION_WEEKS   = 2           # semanas reservadas para walk-forward validation
MIN_WEEKS_FOR_LGBM = 3           # umbral mínimo de historia para usar LightGBM


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA INGESTOR
# ═════════════════════════════════════════════════════════════════════════════

def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga product_sales.csv y geo.csv desde data_dir.

    No aplica ninguna transformación: devuelve los DataFrames tal como
    fueron leídos del disco para que el preprocesamiento sea explícito y
    rastreable.

    Returns
    -------
    sales : DataFrame con columnas [country, product_id, date, zipcode, sales]
    geo   : DataFrame con columnas [country, s_zipcode, e_zipcode, city]
    """
    log.info("═══ [1] DATA INGESTOR ═══════════════════════════════════════════")
    sales = pd.read_csv(data_dir / "product_sales.csv")
    geo   = pd.read_csv(data_dir / "geo.csv")

    log.info(f"  product_sales : {len(sales):>10,} filas  |  columnas: {list(sales.columns)}")
    log.info(f"  geo           : {len(geo):>10,} filas  |  columnas: {list(geo.columns)}")
    return sales, geo


# ═════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESS
# ═════════════════════════════════════════════════════════════════════════════

def preprocess(sales: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza y consolidación en una única tabla con columna 'city'.

    Pasos
    -----
    1. Normalización de tipos de datos:
         - 'sales'   → numérico (errores → NaN)
         - 'date'    → datetime
         - 'zipcode' → entero nullable (Int64)
         - s/e_zipcode en geo → Int64 para garantizar el join por rango
    2. Eliminación del outlier 9999 y nulos en columnas críticas
    3. Join geográfico via _assign_city():
         asigna la ciudad cuyo rango [s_zipcode, e_zipcode] contiene el zipcode
    4. Descarte de registros sin ciudad asignada (cobertura incompleta de geo.csv)

    Returns
    -------
    df_clean : DataFrame con columnas originales + 'city'
    """
    log.info("═══ [2] PREPROCESS ══════════════════════════════════════════════")
    sales = sales.copy()
    geo   = geo.copy()

    # ── Tipos de datos ─────────────────────────────────────────────────────────
    sales["sales"]   = pd.to_numeric(sales["sales"], errors="coerce")
    sales["date"]    = pd.to_datetime(sales["date"])
    sales["zipcode"] = pd.to_numeric(sales["zipcode"], errors="coerce").astype("Int64")
    geo["s_zipcode"] = pd.to_numeric(geo["s_zipcode"], errors="coerce").astype("Int64")
    geo["e_zipcode"] = pd.to_numeric(geo["e_zipcode"], errors="coerce").astype("Int64")

    # ── Eliminación del outlier y nulos ──────────────────────────────────────
    n_orig = len(sales)
    sales  = sales[sales["sales"] != SENTINEL_VALUE]
    sales  = sales.dropna(subset=["sales", "date", "zipcode"])
    log.info(f"  Eliminados {n_orig - len(sales):,} registros (outlier {SENTINEL_VALUE} + nulos)")

    # ── Join geográfico ────────────────────────────────────────────────────────
    sales = _assign_city(sales, geo)

    n_match = sales["city"].notna().sum()
    pct     = n_match / len(sales) * 100
    log.info(f"  Ciudad asignada : {n_match:,} / {len(sales):,}  ({pct:.1f}%)")

    # Descartar registros fuera de la cobertura de geo.csv
    sales = sales[sales["city"].notna()].copy()

    log.info(f"  Dataset limpio  : {len(sales):,} filas")
    return sales.reset_index(drop=True)


def _assign_city(sales_df: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join eficiente por rango de zipcode, segmentado por país.

    Para cada país, itera sobre las filas de geo y aplica una máscara booleana
    s_zipcode ≤ zipcode ≤ e_zipcode. El primer rango que coincide gana
    (los siguientes no sobreescriben porque ya tiene ciudad asignada).
    """
    parts = []
    for ctry, grp in sales_df.groupby("country"):
        g     = grp.copy()
        g["city"] = pd.NA
        geo_c = geo_df[geo_df["country"] == ctry]

        for _, geo_row in geo_c.iterrows():
            mask = (
                g["city"].isna()
                & (g["zipcode"] >= geo_row["s_zipcode"])
                & (g["zipcode"] <= geo_row["e_zipcode"])
            )
            g.loc[mask, "city"] = geo_row["city"]

        parts.append(g)

    return pd.concat(parts, ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def feature_engineering(df: pd.DataFrame) -> tuple:
    """
    Transforma el dataset diario en una vista semanal y construye las features
    para el modelo LightGBM global.

    Estrategia de features
    ----------------------
    Lag features (capturan inercia y tendencia):
      lag_1w, lag_2w, lag_3w, lag_4w  → ventas de 1, 2, 3 y 4 semanas atrás

    Promedio móvil (nivel base del combo):
      roll_2w, roll_4w        → media de las últimas 2 y 4 semanas

    Madurez del combo (maneja cold start):
      n_weeks                 → semanas de historia disponibles (índice cronológico)
      days_observed           → días únicos con ventas (densidad real del historial)

    Encodings categóricos:
      country_enc             → label encoding del país
      city_freq               → frecuencia relativa de la ciudad (robusto a ciudades raras)

    Escala:
      log_lag_1w              → log(1 + lag_1w), comprime la cola derecha de ventas

    Target
    ------
      log(1 + weekly_sales)  → se invierte con expm1() al predecir

    Returns
    -------
    weekly       : DataFrame semanal completo (product_id × city × week_start)
    train_df     : Subconjunto de weekly con lag_1w != NaN (filas entrenables)
    X_train      : Features de entrenamiento (DataFrame)
    y_train      : Target log(1 + weekly_sales) (Series)
    feature_cols : Lista con los nombres de las features
    dow_index    : Dict {(country, day_name): seasonal_index} del EDA
    """
    log.info("═══ [3] FEATURE ENGINEERING ═════════════════════════════════════")
    df = df.copy()

    # ── Alinear al inicio de la semana (lunes) ─────────────────────────────────
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")

    # ── Serie semanal por combo (product_id × city) ────────────────────────────
    weekly = (
        df.groupby(["product_id", "city", "country", "week_start"])["sales"]
        .sum()
        .reset_index(name="weekly_sales")
        .sort_values(["product_id", "city", "week_start"])
        .reset_index(drop=True)
    )

    combo_grp = weekly.groupby(["product_id", "city"])

    # ── Lags semanales ─────────────────────────────────────────────────────────
    for lag in LAG_WEEKS:
        weekly[f"lag_{lag}w"] = combo_grp["weekly_sales"].shift(lag)

    # ── Promedios móviles (shift(1) para no filtrar el target actual) ──────────
    for w in ROLLING_WINDOWS:
        weekly[f"roll_{w}w"] = combo_grp["weekly_sales"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )

    # ── Madurez del combo ──────────────────────────────────────────────────────
    weekly["n_weeks"] = combo_grp.cumcount() + 1

    days_obs = (
        df.groupby(["product_id", "city"])["date"]
        .nunique()
        .reset_index(name="days_observed")
    )
    weekly = weekly.merge(days_obs, on=["product_id", "city"], how="left")

    # ── Índice de estacionalidad del día de la semana (del EDA) ───────────────
    # Nota: en el set de entrenamiento week_start es siempre lunes, por lo que
    # este feature es constante en training (LightGBM lo ignorará) pero útil
    # en predicción, donde los 3 forecast dates pueden caer en cualquier día.
    dow_index = _compute_dow_index(df)
    weekly["dow_index"] = weekly.apply(
        lambda r: dow_index.get((r["country"], r["week_start"].day_name()), 1.0),
        axis=1,
    )

    # ── Encodings categóricos ──────────────────────────────────────────────────
    weekly["country_enc"] = weekly["country"].astype("category").cat.codes

    city_freq = (
        weekly.groupby("city")["product_id"]
        .count()
        .div(len(weekly))
        .rename("city_freq")
        .reset_index()
    )
    weekly = weekly.merge(city_freq, on="city", how="left")

    # ── Transformación log del primer lag ──────────────────────────────────────
    weekly["log_lag_1w"] = np.log1p(weekly["lag_1w"].fillna(0))

    # ── Features de alto impacto para series cortas ───────────────────────────
    # Tendencia: ¿el combo está creciendo o cayendo?
    weekly["trend_4w"] = (weekly["lag_1w"] - weekly["lag_4w"]) / 3
    # Variabilidad de la demanda (coeficiente de variación sobre 4 semanas)
    weekly["cv_4w"] = (
        weekly[["lag_1w", "lag_2w", "lag_3w", "lag_4w"]].std(axis=1)
        / (weekly["roll_4w"] + 1)
    )
    # Intermitencia: % de semanas recientes sin ventas (clave para sparse)
    weekly["zero_rate"] = combo_grp["weekly_sales"].transform(
        lambda x: (x.shift(1) == 0).rolling(4, min_periods=1).mean()
    )
    # Spike reciente: máximo de las últimas 4 semanas
    weekly["max_4w"] = combo_grp["weekly_sales"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).max()
    )
    # Momentum reciente: tasa de crecimiento semanal relativo
    weekly["growth_rate"] = (weekly["lag_1w"] - weekly["lag_2w"]) / (weekly["lag_2w"] + 1)

    # ── Dataset de entrenamiento: requiere al menos lag_1w ────────────────────
    feature_cols = (
        [f"lag_{l}w" for l in LAG_WEEKS]
        + [f"roll_{w}w" for w in ROLLING_WINDOWS]
        + ["n_weeks", "days_observed",
           "country_enc", "city_freq", "log_lag_1w",
           "trend_4w", "cv_4w", "zero_rate", "max_4w", "growth_rate"]
        # dow_index NO se incluye como feature: en training week_start es siempre
        # lunes y el modelo no puede aprender variación por día de semana.
        # El efecto estacional se aplica como multiplicador post-modelo en predict().
    )

    train_df = weekly.dropna(subset=["lag_1w"]).copy()
    X_train  = train_df[feature_cols]
    y_train  = np.log1p(train_df["weekly_sales"])

    log.info(f"  Combos únicos (product×city) : {weekly.groupby(['product_id','city']).ngroups:,}")
    log.info(f"  Semanas × combos (total)     : {len(weekly):,}")
    log.info(f"  Filas de entrenamiento       : {len(X_train):,}")
    log.info(f"  Features ({len(feature_cols)})          : {feature_cols}")

    return weekly, train_df, X_train, y_train, feature_cols, dow_index


def _compute_dow_index(df: pd.DataFrame) -> dict:
    """
    Calcula el índice de estacionalidad semanal por (país, día de semana).

    Índice = promedio_ventas_día / promedio_diario_del_país
    Valor > 1.0 → ese día tiene más demanda que el promedio semanal.
    """
    daily = (
        df.groupby(["country", "date"])["sales"]
        .sum()
        .reset_index()
    )
    daily["day_name"] = pd.to_datetime(daily["date"]).dt.day_name()

    avg_dow = (
        daily.groupby(["country", "day_name"])["sales"]
        .mean()
        .reset_index(name="avg")
    )
    avg_dow["weekly_mean"] = avg_dow.groupby("country")["avg"].transform("mean")
    avg_dow["index"]       = avg_dow["avg"] / avg_dow["weekly_mean"]

    return {
        (row["country"], row["day_name"]): row["index"]
        for _, row in avg_dow.iterrows()
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. TUNE HYPERPARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

def tune_hyperparameters(
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = N_OPTUNA_TRIALS,
) -> dict:
    """
    Búsqueda bayesiana de hiperparámetros para LightGBM con Optuna.

    Estrategia de validación
    ------------------------
    Walk-forward temporal: se entrena sobre todas las semanas excepto las
    últimas VALIDATION_WEEKS, y se evalúa sobre esas semanas finales.
    Esto replica exactamente el escenario de producción (predecir el futuro
    usando solo el pasado).

    Espacio de búsqueda
    -------------------
    num_leaves, max_depth, learning_rate, n_estimators,
    min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda

    Objetivo: minimizar WMAPE en el set de validación.

    Returns
    -------
    best_params : dict con los hiperparámetros óptimos
    """
    log.info("═══ [4] TUNE HYPERPARAMETERS ════════════════════════════════════")

    weeks     = sorted(train_df["week_start"].unique())
    val_set   = set(weeks[-VALIDATION_WEEKS:])
    tr_mask   = ~train_df["week_start"].isin(val_set)
    val_mask  = train_df["week_start"].isin(val_set)

    X_tr, y_tr = X_train[tr_mask],  y_train[tr_mask]
    X_val      = X_train[val_mask]
    y_val_true = np.expm1(y_train[val_mask])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "regression_l1",
            "metric":            "mae",
            "verbosity":         -1,
            "n_jobs":            -1,
            "random_state":      42,
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        }
        mdl   = lgb.LGBMRegressor(**params)
        mdl.fit(X_tr, y_tr)
        preds = np.maximum(np.expm1(mdl.predict(X_val)), 0)
        return _wmape(y_val_true, preds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params | {
        "objective": "regression_l1",
        "metric":    "mae",
        "verbosity": -1,
        "n_jobs":    -1,
        "random_state": 42,
    }
    log.info(f"  Mejor WMAPE val : {study.best_value:.4f}")
    log.info(f"  Mejores params : {study.best_params}")
    return best_params


# ═════════════════════════════════════════════════════════════════════════════
# 5. TRAIN
# ═════════════════════════════════════════════════════════════════════════════

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> lgb.LGBMRegressor:
    """
    Entrena el modelo LightGBM global sobre TODO el set de entrenamiento
    con los hiperparámetros optimizados.

    Modelo global vs. modelos por combo
    ------------------------------------
    Se entrena un único modelo con todos los combos product_id×city como
    filas de entrenamiento. Esto permite que el modelo aprenda patrones
    transversales entre combos (e.g., comportamiento típico de un producto
    nuevo en una ciudad grande), resolviendo naturalmente el cold start.

    Returns
    -------
    model : LGBMRegressor entrenado, target en escala log(1 + weekly_sales)
    """
    log.info("═══ [5] TRAIN ════════════════════════════════════════════════════")
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    importances = (
        pd.Series(model.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
    )
    log.info("  Feature importances (top 8):")
    for feat, imp in importances.head(8).items():
        log.info(f"    {feat:<20}  {imp:>6.0f}")

    return model


# ═════════════════════════════════════════════════════════════════════════════
# 6. METRICS
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: lgb.LGBMRegressor,
) -> pd.DataFrame:
    """
    Evaluación sobre las últimas VALIDATION_WEEKS semanas del dataset.
    Compara LightGBM vs. Baseline (media ponderada reciente) por tier de densidad.

    Tiers de densidad (por días observados del combo):
        sparse  :  ≤  7 días  → cold start, alta incertidumbre
        medium  :  8–30 días  → historia moderada
        dense   :  > 30 días  → suficiente historia para series temporales

    Métricas reportadas
    -------------------
    MAE           : Error absoluto medio (en unidades de ventas semanales)
    MAPE          : Error porcentual absoluto medio (excluye combos con venta < 1)
    Bias          : Sesgo relativo = Σ(pred − real) / Σ(real)
                      > 0 → modelo sobreestima  |  < 0 → modelo subestima
    Accuracy         : 1 − WMAPE  (accuracy de supply chain basada en WMAPE)
    delta_mape_%     : mejora de LightGBM sobre baseline en MAPE (positivo = mejor)
    delta_accuracy_pp: mejora de LightGBM sobre baseline en Accuracy (puntos porcentuales)

    Returns
    -------
    metrics_df : DataFrame con métricas por tier + fila ALL
    """
    log.info("═══ [6] METRICS ══════════════════════════════════════════════════")

    weeks    = sorted(train_df["week_start"].unique())
    val_set  = set(weeks[-VALIDATION_WEEKS:])
    val_mask = train_df["week_start"].isin(val_set)

    val_df = train_df[val_mask].copy()
    val_df["y_true"]    = np.expm1(y_train[val_mask].values)
    val_df["lgbm_pred"] = np.maximum(np.expm1(model.predict(X_train[val_mask])), 0)
    val_df["base_pred"] = np.maximum(
        val_df["roll_2w"].fillna(val_df["lag_1w"]).fillna(0).values, 0
    )

    # Clasificación por tier de densidad
    val_df["tier"] = pd.cut(
        val_df["days_observed"],
        bins=[0, 7, 30, float("inf")],
        labels=["sparse (≤7d)", "medium (8-30d)", "dense (>30d)"],
    )

    groups = [("ALL", val_df)] + [
        (str(t), g) for t, g in val_df.groupby("tier", observed=True)
    ]
    rows = []
    for tier_name, grp in groups:
        lgbm_bias  = _bias(grp["y_true"], grp["lgbm_pred"])
        base_bias  = _bias(grp["y_true"], grp["base_pred"])
        lgbm_wmape = _wmape(grp["y_true"], grp["lgbm_pred"])
        base_wmape = _wmape(grp["y_true"], grp["base_pred"])
        rows.append({
            "tier":           tier_name,
            "n_filas":        len(grp),
            "n_combos":       grp[["product_id", "city"]].drop_duplicates().__len__(),
            "lgbm_mae":       round(mean_absolute_error(grp["y_true"], grp["lgbm_pred"]), 3),
            "base_mae":       round(mean_absolute_error(grp["y_true"], grp["base_pred"]), 3),
            "lgbm_mape":      round(_mape(grp["y_true"], grp["lgbm_pred"]), 4),
            "base_mape":      round(_mape(grp["y_true"], grp["base_pred"]), 4),
            "lgbm_wmape":     round(lgbm_wmape, 4),
            "base_wmape":     round(base_wmape, 4),
            "lgbm_bias":      round(lgbm_bias, 4),
            "base_bias":      round(base_bias, 4),
            "lgbm_accuracy":  round(1 - abs(lgbm_wmape), 4),
            "base_accuracy":  round(1 - abs(base_wmape), 4),
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df["delta_mape_%"] = (
        (metrics_df["base_mape"] - metrics_df["lgbm_mape"])
        / metrics_df["base_mape"] * 100
    ).round(1)
    metrics_df["delta_wmape_%"] = (
        (metrics_df["base_wmape"] - metrics_df["lgbm_wmape"])
        / metrics_df["base_wmape"] * 100
    ).round(1)
    metrics_df["delta_accuracy_pp"] = (
        (metrics_df["lgbm_accuracy"] - metrics_df["base_accuracy"]) * 100
    ).round(1)

    log.info("\n" + metrics_df.to_string(index=False))
    return metrics_df


def _mape(y_true, y_pred, eps: float = 1.0) -> float:
    """
    MAPE robusto: excluye filas donde y_true < eps para evitar división por cero
    (valores de venta muy bajos distorsionan el porcentaje de error).
    """
    yt   = np.asarray(y_true, dtype=float)
    yp   = np.asarray(y_pred, dtype=float)
    mask = yt >= eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(yt[mask] - yp[mask]) / yt[mask]))


def _wmape(y_true, y_pred, eps: float = 1.0) -> float:
    """
    WMAPE (Weighted MAPE): pondera el error absoluto por el volumen real.
    Fórmula: Σ|pred − real| / Σ real

    Ventaja sobre MAPE: los combos de bajo volumen (1-3 uds.) no distorsionan
    el promedio — cada error se pondera por su contribución al volumen total.
    Con distribuciones heavy-tail (muchos combos esporádicos), WMAPE es más
    estable e interpretable que MAPE simple.
    """
    yt   = np.asarray(y_true, dtype=float)
    yp   = np.asarray(y_pred, dtype=float)
    mask = yt >= eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.abs(yt[mask] - yp[mask]).sum() / yt[mask].sum())


def _bias(y_true, y_pred, eps: float = 1.0) -> float:
    """
    Sesgo relativo del forecast (Bias).

    Definición estándar en supply chain / demand planning:
        Bias = Σ(pred − real) / Σ(real)

    Interpretación:
        > 0  →  el modelo sobreestima sistemáticamente (over-forecast)
        < 0  →  el modelo subestima sistemáticamente (under-forecast)
        = 0  →  modelo sin sesgo

    Nota: se excluyen filas con y_true < eps para evitar que ventas cercanas a
    cero dominen el denominador y distorsionen el sesgo agregado.
    """
    yt   = np.asarray(y_true, dtype=float)
    yp   = np.asarray(y_pred, dtype=float)
    mask = yt >= eps
    if mask.sum() == 0:
        return float("nan")
    return float((yp[mask] - yt[mask]).sum() / yt[mask].sum())


# ═════════════════════════════════════════════════════════════════════════════
# 7. PREDICT
# ═════════════════════════════════════════════════════════════════════════════

def predict(
    df_clean: pd.DataFrame,
    weekly: pd.DataFrame,
    model: lgb.LGBMRegressor,
    feature_cols: list[str],
    dow_index: dict,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Genera submission.csv con las 3 fechas de forecast.

    Semántica del forecast
    ----------------------
    Para cada forecast_date en [last_date+1, last_date+2, last_date+3]:
        submission[forecast_date] = suma proyectada de ventas del período
                                    [forecast_date, forecast_date + 6 días]

    Estrategia de predicción por tier
    -----------------------------------
    ≥ MIN_WEEKS_FOR_LGBM semanas  →  LightGBM global (predicción en escala log)
    < MIN_WEEKS_FOR_LGBM semanas  →  Fallback: roll_2w (o lag_1w si nulo)

    El índice de estacionalidad (dow_index) se aplica como multiplicador
    post-modelo según el día de la semana del forecast_date, capturando
    el efecto semanal identificado en el EDA.

    Returns
    -------
    submission : DataFrame [product_id, date, city, sales]
                 (guardado también en output_dir/submission.csv)
    """
    log.info("═══ [7] PREDICT ══════════════════════════════════════════════════")

    last_date      = df_clean["date"].max()
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 4)]
    log.info(f"  Última fecha del dataset : {last_date.date()}")
    log.info(f"  Fechas de forecast       : {[d.date() for d in forecast_dates]}")

    # Construir features de predicción desde el historial semanal real
    pred_base = _build_prediction_features(weekly)

    all_rows = []
    for fd in forecast_dates:
        pf = pred_base.copy()

        # Actualizar features dependientes de la fecha de forecast
        pf["dow_index"]  = pf["country"].map(
            lambda c: dow_index.get((c, fd.day_name()), 1.0)
        )
        pf["log_lag_1w"] = np.log1p(pf["lag_1w"].fillna(0))

        X_pred    = pf[feature_cols].fillna(0)
        lgbm_mask = pf["n_weeks"] >= MIN_WEEKS_FOR_LGBM
        preds     = np.zeros(len(pf))

        # LightGBM para combos con historia suficiente
        if lgbm_mask.sum() > 0:
            raw = np.expm1(model.predict(X_pred[lgbm_mask]))
            # Ajuste estacional post-modelo (dow_index como multiplicador)
            seasonal = pf.loc[lgbm_mask, "dow_index"].values
            preds[lgbm_mask.values] = np.maximum(raw * seasonal, 0)

        # Fallback media ponderada para combos con poca historia
        if (~lgbm_mask).sum() > 0:
            base = (
                pf.loc[~lgbm_mask, "roll_2w"]
                .fillna(pf.loc[~lgbm_mask, "lag_1w"])
                .fillna(0)
            )
            seasonal = pf.loc[~lgbm_mask, "dow_index"].values
            preds[~lgbm_mask.values] = np.maximum(base.values * seasonal, 0)

        pf["sales"] = np.round(preds, 4)
        pf["date"]  = fd
        all_rows.append(pf[["product_id", "city", "date", "sales"]])

    submission = (
        pd.concat(all_rows, ignore_index=True)
        [["product_id", "date", "city", "sales"]]
        .sort_values(["product_id", "city", "date"])
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "submission.csv"
    submission.to_csv(out_path, index=False)

    log.info(f"  Combos con predicción : {submission[['product_id','city']].drop_duplicates().__len__():,}")
    log.info(f"  Filas en submission   : {len(submission):,}")
    log.info(f"  Guardado en           : {out_path}")
    return submission


def _build_prediction_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el vector de features para cada combo para la semana de forecast.

    Toma directamente las últimas N ventas semanales de cada combo para
    construir lags y rolling means correctos (lag_1w = última semana conocida,
    lag_2w = penúltima, etc.), evitando el desplazamiento de índices que ocurre
    si se reusan los lag columns del set de entrenamiento.
    """
    weekly_sorted = weekly.sort_values(["product_id", "city", "week_start"])
    records = []

    for (pid, city), grp in weekly_sorted.groupby(["product_id", "city"]):
        sales_h = grp["weekly_sales"].values   # orden cronológico, más antiguo primero
        n       = len(sales_h)

        # Pre-cómputo de valores base para features derivadas
        lag_1w_val  = sales_h[-1] if n >= 1 else np.nan
        lag_4w_val  = sales_h[-4] if n >= 4 else np.nan
        roll_4w_val = float(np.mean(sales_h[-4:])) if n >= 1 else np.nan
        recent_4    = sales_h[-min(4, n):]
        lag_quad    = np.array([
            sales_h[-1] if n >= 1 else np.nan,
            sales_h[-2] if n >= 2 else np.nan,
            sales_h[-3] if n >= 3 else np.nan,
            sales_h[-4] if n >= 4 else np.nan,
        ], dtype=float)
        cv_4w_val   = (
            float(np.nanstd(lag_quad, ddof=1) / (roll_4w_val + 1))
            if not np.isnan(roll_4w_val) else 0.0
        )

        rec = {
            "product_id":    pid,
            "city":          city,
            "country":       grp["country"].iloc[-1],
            # Lags desde el pasado más reciente
            "lag_1w":        lag_1w_val,
            "lag_2w":        sales_h[-2]           if n >= 2 else np.nan,
            "lag_3w":        sales_h[-3]           if n >= 3 else np.nan,
            "lag_4w":        lag_4w_val,
            # Promedios móviles
            "roll_2w":       np.mean(sales_h[-2:]) if n >= 1 else np.nan,
            "roll_4w":       roll_4w_val,
            # Madurez
            "n_weeks":       n,
            "days_observed": grp["days_observed"].iloc[-1],
            # Encodings (se llenan desde el weekly df ya procesado)
            "country_enc":   grp["country_enc"].iloc[-1] if "country_enc" in grp.columns else 0,
            "city_freq":     grp["city_freq"].iloc[-1]   if "city_freq"   in grp.columns else 0.0,
            # Estos dos se actualizan en el loop de forecast_dates
            "dow_index":     1.0,
            "log_lag_1w":    np.log1p(lag_1w_val if not np.isnan(lag_1w_val) else 0),
            # Features de alto impacto para series cortas
            "trend_4w":      float((lag_1w_val - lag_4w_val) / 3)
                             if (not np.isnan(lag_1w_val) and not np.isnan(lag_4w_val))
                             else 0.0,
            "cv_4w":         cv_4w_val,
            "zero_rate":     float((recent_4 == 0).mean()),
            "max_4w":        float(np.max(recent_4)),
            "growth_rate":   float((lag_1w_val - sales_h[-2]) / (sales_h[-2] + 1))
                             if (not np.isnan(lag_1w_val) and n >= 2)
                             else 0.0,
        }
        records.append(rec)

    return pd.DataFrame(records)


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    data_dir:        Path = DATA_DIR,
    output_dir:      Path = OUTPUT_DIR,
    n_optuna_trials: int  = N_OPTUNA_TRIALS,
    skip_tuning:     bool = False,
) -> dict:
    """
    Orquesta el pipeline completo de punta a punta.

    Args
    ----
    data_dir        : carpeta con product_sales.csv y geo.csv
    output_dir      : carpeta donde se guarda submission.csv
    n_optuna_trials : número de trials en la búsqueda de hiperparámetros
    skip_tuning     : si True, omite Optuna y usa parámetros por defecto
                      (útil para prototipado rápido)

    Returns
    -------
    dict con claves: model, metrics, submission, best_params, feature_cols
    """
    # 1. Ingest
    sales, geo = load_data(data_dir)

    # 2. Preprocess
    df_clean = preprocess(sales, geo)

    # 3. Feature Engineering
    weekly, train_df, X_train, y_train, feature_cols, dow_index = (
        feature_engineering(df_clean)
    )

    # 4. Tune (o params por defecto)
    if skip_tuning:
        log.info("═══ [4] TUNE HYPERPARAMETERS — omitido, usando defaults ═════════")
        best_params = _default_params()
    else:
        best_params = tune_hyperparameters(train_df, X_train, y_train, n_optuna_trials)

    # 5. Train
    model = train_model(X_train, y_train, best_params)

    # 6. Metrics
    metrics = evaluate(train_df, X_train, y_train, model)

    # 7. Predict
    submission = predict(df_clean, weekly, model, feature_cols, dow_index, output_dir)

    log.info("═══ PIPELINE COMPLETADO ══════════════════════════════════════════")
    return {
        "model":        model,
        "metrics":      metrics,
        "submission":   submission,
        "best_params":  best_params,
        "feature_cols": feature_cols,
    }


def _default_params() -> dict:
    """Hiperparámetros de referencia (sin tuning Optuna)."""
    return {
        "objective":         "regression_l1",
        "metric":            "mae",
        "verbosity":         -1,
        "n_jobs":            -1,
        "random_state":      42,
        "num_leaves":        63,
        "max_depth":         7,
        "learning_rate":     0.05,
        "n_estimators":      300,
        "min_child_samples": 20,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forecast de demanda semanal por product_id × city — MercadoLibre"
    )
    parser.add_argument("--data-dir",    default="data",   help="Directorio de datos CSV")
    parser.add_argument("--output-dir",  default="output", help="Directorio de salida")
    parser.add_argument("--trials",      default=N_OPTUNA_TRIALS, type=int,
                        help=f"Número de trials Optuna (default: {N_OPTUNA_TRIALS})")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Omitir Optuna y usar hiperparámetros por defecto")
    args = parser.parse_args()

    results = run_pipeline(
        data_dir        = Path(args.data_dir),
        output_dir      = Path(args.output_dir),
        n_optuna_trials = args.trials,
        skip_tuning     = args.skip_tuning,
    )

    print("\n" + "═" * 60)
    print("SUBMISSION — primeras 15 filas")
    print("═" * 60)
    print(results["submission"].head(15).to_string(index=False))

    print("\n" + "═" * 60)
    print("MÉTRICAS POR TIER DE DENSIDAD")
    print("═" * 60)
    print(results["metrics"].to_string(index=False))
