# Forecast de Demanda Semanal — MercadoLibre Technical Assessment

Pipeline de machine learning para pronosticar la demanda semanal de productos por ciudad, desarrollado como parte del proceso de evaluación técnica de MercadoLibre.

---

## Descripción del proyecto

MercadoLibre necesita pronosticar la demanda de productos con la mayor precisión posible para gestionar el stock correcto en cada ciudad y garantizar entregas a tiempo.

**Objetivo:** generar un forecast de demanda semanal a nivel `product_id × city` para las 3 fechas siguientes a la última fecha disponible en el dataset.

**Semántica del forecast:** el forecast para el día N representa la suma proyectada de ventas del período `[N, N+6]` (ventana de 7 días).

**Período de entrenamiento:** 2024-06-01 → 2024-08-07 (68 días, ~9.7 semanas)

**Fechas de predicción:** 2024-08-08, 2024-08-09, 2024-08-10

### Enfoque de modelado

Se entrena un **modelo LightGBM global** sobre todos los combos `product_id × city` simultáneamente. Esto permite que el modelo aprenda patrones transversales entre combos y resuelve naturalmente el problema de cold start mediante features de contexto (`n_weeks`, `days_observed`, `city_freq`).

Los combos con historia insuficiente (< `MIN_WEEKS_FOR_LGBM` semanas) usan un fallback de media ponderada reciente (`roll_2w / lag_1w`).

---

## Arquitectura del pipeline

![Arquitectura del Pipeline](pipeline_architecture.png)

El pipeline consta de 7 etapas secuenciales:

| Etapa | Función | Descripción |
|---|---|---|
| ① | `load_data()` | Carga de CSVs sin transformación |
| ② | `preprocess()` + `_assign_city()` | Limpieza, tipos, join geográfico por rango de zipcode |
| ③ | `feature_engineering()` | Serie semanal + 17 features para LightGBM |
| ④ | `tune_hyperparameters()` | Optuna TPE, walk-forward, objetivo WMAPE *(opcional)* |
| ⑤ | `train_model()` | Entrenamiento LightGBM global sobre 100% de los datos |
| ⑥ | `evaluate()` | Walk-forward múltiple + early stopping, métricas por tier |
| ⑦ | `predict()` | Genera `submission.csv` para las 3 fechas de forecast |

---

## Estructura del proyecto

```
forecast_meli/
│
├── data/                          # Datos de entrada (solo lectura)
│   ├── product_sales.csv          # Ventas históricas (227 355 filas)
│   └── geo.csv                    # Rangos de zipcode → ciudad (10 805 filas)
│
├── output/                        # Resultados generados por el pipeline
│   └── submission.csv             # Forecast final (8 625 filas)
│
├── forecast_pipeline.py           # Pipeline principal end-to-end
├── EDA_forecast_meli.ipynb        # Análisis exploratorio de datos (EDA)
├── pipeline_architecture.png      # Diagrama de arquitectura del pipeline
│
├── requirements.txt               # Dependencias Python del proyecto
├── setup_env.sh                   # Script de instalación del entorno conda
│
├── _draw_architecture.py          # Script auxiliar: genera el diagrama PNG
├── instructions.md                # Enunciado original del assessment
└── resumen.txt                    # Resumen del problema y datos
```

---

## Descripción de archivos principales

### `forecast_pipeline.py`
Pipeline completo end-to-end. Contiene las 7 funciones principales más helpers internos.

**Features del modelo (17 en total):**

| Grupo | Features | Propósito |
|---|---|---|
| Lags | `lag_1w`, `lag_2w`, `lag_3w`, `lag_4w` | Inercia y memoria de corto plazo |
| Rolling | `roll_2w`, `roll_4w` | Nivel base del combo |
| Tendencia | `trend_4w`, `growth_rate` | Pendiente y momentum reciente |
| Variabilidad | `cv_4w` | Coeficiente de variación de la demanda |
| Intermitencia | `zero_rate`, `max_4w` | Captura demanda esporádica (sparse) |
| Madurez | `n_weeks`, `days_observed` | Cold start handling |
| Categorías | `country_enc`, `city_freq` | Encodings robustos |
| Estacionalidad | `dow_index`, `log_lag_1w` | Día de semana + escala log |

**Métricas de evaluación:**

| Métrica | Definición | Uso |
|---|---|---|
| MAE | Error absoluto medio | Error en unidades absolutas |
| MAPE | Error % absoluto medio (excluye ventas < 1) | Error relativo por combo |
| WMAPE | Σ\|pred−real\| / Σreal | Métrica principal, ponderada por volumen |
| Bias | Σ(pred−real) / Σreal | Sesgo sistemático del modelo |
| Accuracy | 1 − \|WMAPE\| | Accuracy de supply chain basada en WMAPE |

**Resultados con `VALIDATION_WEEKS = 2` (walk-forward múltiple):**

```
tier           n_filas  n_combos  lgbm_wmape  base_wmape  lgbm_accuracy  delta_wmape_%  delta_accuracy_pp
ALL               2158      1402      0.5888      0.6979         0.4112          15.6%          +10.9 pp
sparse (≤7d)       397       335      0.5346      0.7687         0.4654          30.5%          +23.4 pp
medium (8-30d)    1009       673      0.6150      0.8925         0.3850          31.1%          +27.8 pp
dense (>30d)       752       394      0.5857      0.6547         0.4143          10.5%           +6.9 pp
```

**CLI:**
```bash
python forecast_pipeline.py                     # pipeline completo con Optuna
python forecast_pipeline.py --skip-tuning       # omite Optuna, params por defecto
python forecast_pipeline.py --trials 50         # más trials de Optuna
python forecast_pipeline.py --data-dir mi/ruta  # ruta personalizada
```

---

### `EDA_forecast_meli.ipynb`
Notebook de análisis exploratorio. Contiene:
- Análisis de la distribución de ventas (heavy-tail, outlier 9999)
- Cobertura del join geográfico por país
- Estacionalidad intra-semanal (índice `dow_index`)
- Análisis del cold start y distribución de historia por combo
- Distribución de combos por tier de densidad
- Correlaciones entre features y target

### `data/product_sales.csv`
Ventas históricas de productos.

| Columna | Tipo | Descripción |
|---|---|---|
| `country` | string | País de la venta |
| `product_id` | string | ID del producto |
| `date` | date | Fecha de la venta |
| `zipcode` | int | Código postal desde donde se sirve |
| `sales` | float | Número de ventas (9999 = outlier del sistema) |

### `data/geo.csv`
Mapeo geográfico de rangos de zipcode a ciudad.

| Columna | Tipo | Descripción |
|---|---|---|
| `country` | string | País |
| `s_zipcode` | int | Inicio del rango de zipcode |
| `e_zipcode` | int | Fin del rango de zipcode |
| `city` | string | Ciudad que sirve ese rango |

### `output/submission.csv`
Forecast generado por el pipeline.

| Columna | Descripción |
|---|---|
| `product_id` | ID del producto |
| `date` | Fecha de inicio del período de forecast |
| `city` | Ciudad desde donde se sirve el producto |
| `sales` | Ventas semanales proyectadas para [date, date+6 días] |

---

## Instalación del entorno

### Opción 1 — Script automático (recomendado)

```bash
bash setup_env.sh
```

El script:
1. Crea el entorno conda `forecast_meli` con Python 3.11
2. Instala todas las dependencias desde `requirements.txt`
3. Registra el kernel de Jupyter como *"Python (forecast_meli)"*

Para recrear el entorno desde cero:
```bash
bash setup_env.sh --reinstall
```

### Opción 2 — Manual con pip

```bash
# Crear entorno (opcional pero recomendado)
conda create -n forecast_meli python=3.11 -y
conda activate forecast_meli

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias (`requirements.txt`)

```
lightgbm>=4.6.0
optuna>=4.9.0
scikit-learn>=1.9.0
pandas>=3.0.3
numpy>=2.4.6
```

Versiones probadas con **Python 3.11** en Linux x86-64.

---

## Cómo ejecutar el pipeline

### Prerequisitos

```bash
conda activate forecast_meli
cd forecast_meli/
```

Los archivos de datos deben estar en `data/product_sales.csv` y `data/geo.csv`.

### Ejecución rápida (sin tuning)

```bash
python forecast_pipeline.py --skip-tuning
```

Tiempo estimado: ~1 minuto. Usa hiperparámetros por defecto.

### Ejecución completa (con Optuna)

```bash
python forecast_pipeline.py --trials 100
```

Tiempo estimado: 15-30 minutos dependiendo del hardware. Optimiza hiperparámetros con 100 trials bayesianos.

### Opciones disponibles

| Flag | Default | Descripción |
|---|---|---|
| `--skip-tuning` | `False` | Omite Optuna, usa parámetros por defecto |
| `--trials N` | `30` | Número de trials de Optuna |
| `--data-dir PATH` | `data/` | Carpeta con los CSVs de entrada |
| `--output-dir PATH` | `output/` | Carpeta donde se guarda `submission.csv` |

### Output esperado

```
output/
└── submission.csv    # 8 625 filas: 2 875 combos × 3 fechas de forecast
```

Las métricas de evaluación se imprimen en consola al finalizar:

```
MÉTRICAS POR TIER DE DENSIDAD
tier           n_filas  n_combos  lgbm_wmape  ...  delta_accuracy_pp
ALL               2158      1402      0.5888  ...          +10.9 pp
sparse (≤7d)       397       335      0.5346  ...          +23.4 pp
medium (8-30d)    1009       673      0.6150  ...          +27.8 pp
dense (>30d)       752       394      0.5857  ...           +6.9 pp
```

---

## Supuestos del modelo

1. **Outlier 9999**: el valor `sales = 9999` es un artefacto del sistema fuente, no una venta real → se elimina en el preprocesamiento.
2. **Join geográfico**: cada `zipcode` cae dentro de un rango `[s_zipcode, e_zipcode]` en `geo.csv`, que determina la ciudad desde la que se sirve el producto.
3. **Target**: se modela `log(1 + weekly_sales)` para comprimir la distribución heavy-tail. Se invierte con `expm1()` al predecir.
4. **Modelo global**: un único LightGBM entrena sobre todos los combos. Las features de contexto (`n_weeks`, `days_observed`) resuelven el cold start sin necesidad de modelos separados.
5. **Validación walk-forward múltiple**: `VALIDATION_WEEKS` folds independientes, cada uno entrenando solo en el pasado respecto a su semana de validación.

---

## Decisiones de diseño clave

### ¿Por qué LightGBM global en lugar de modelos por combo?

Con 2 875 combos y ~9 semanas de historia, entrenar un modelo individual por combo daría como máximo 5 puntos de entrenamiento por modelo. Un modelo global aprovecha los patrones transversales entre combos (e.g., un producto nuevo en una ciudad grande se comporta similar a otro producto nuevo en otra ciudad grande) y mejora significativamente el rendimiento en cold start.

### ¿Por qué WMAPE como métrica principal?

La distribución de ventas es heavy-tail: muchos combos con 1-3 unidades/semana y pocos con 30+. El MAPE simple da el mismo peso a un combo de 1 unidad que a uno de 1000, distorsionando la optimización. WMAPE pondera por volumen, alineando la métrica con el impacto real en inventario.

### ¿Por qué VALIDATION_WEEKS = 2?

Con 1 semana de validación, `n_filas = n_combos` (cada combo aparece exactamente 1 vez), lo que produce estimadores muy inestables. El bias puede cambiar de signo al mover la ventana 1 semana. Con 2 semanas se obtiene un estimador estadísticamente más robusto sin sacrificar demasiado training data.
