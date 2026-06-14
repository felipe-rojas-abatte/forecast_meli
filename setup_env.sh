#!/usr/bin/env bash
# =============================================================================
# setup_env.sh
# Crea el entorno conda forecast_meli e instala todas las dependencias
# necesarias para ejecutar forecast_pipeline.py
#
# Uso:
#   bash setup_env.sh              # crea entorno nuevo + instala
#   bash setup_env.sh --reinstall  # elimina el entorno existente y recrea
# =============================================================================

set -euo pipefail

ENV_NAME="forecast_meli"
PYTHON_VERSION="3.11"
REQUIREMENTS="requirements.txt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colores para output ───────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn] ${NC} $*"; }
err()  { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ── Verificar que conda está disponible ──────────────────────────────────────
if ! command -v conda &> /dev/null; then
    err "conda no encontrado. Instala Anaconda o Miniconda primero."
fi

# ── Verificar que requirements.txt existe ────────────────────────────────────
if [[ ! -f "$SCRIPT_DIR/$REQUIREMENTS" ]]; then
    err "No se encontró $REQUIREMENTS en $SCRIPT_DIR"
fi

# ── Opción --reinstall: eliminar entorno existente ───────────────────────────
if [[ "${1:-}" == "--reinstall" ]]; then
    warn "--reinstall: eliminando entorno existente '$ENV_NAME'..."
    conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
    log "Entorno eliminado."
fi

# ── Crear entorno si no existe ────────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    log "El entorno '$ENV_NAME' ya existe. Saltando creación."
    log "Usa --reinstall para recrearlo desde cero."
else
    log "Creando entorno conda '$ENV_NAME' con Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    log "Entorno creado."
fi

# ── Instalar dependencias desde requirements.txt ─────────────────────────────
log "Instalando dependencias desde $REQUIREMENTS..."
conda run -n "$ENV_NAME" pip install -r "$SCRIPT_DIR/$REQUIREMENTS"

# ── Registrar kernel de Jupyter (opcional) ───────────────────────────────────
log "Registrando kernel de Jupyter como 'Python ($ENV_NAME)'..."
conda run -n "$ENV_NAME" pip install ipykernel --quiet
conda run -n "$ENV_NAME" python -m ipykernel install \
    --user \
    --name "$ENV_NAME" \
    --display-name "Python ($ENV_NAME)" \
    2>/dev/null || warn "No se pudo registrar el kernel de Jupyter (no es crítico)."

# ── Verificación final ────────────────────────────────────────────────────────
log "Verificando instalación..."
conda run -n "$ENV_NAME" python - << 'PYEOF'
import importlib, sys

packages = {
    "lightgbm":  "lightgbm",
    "optuna":    "optuna",
    "sklearn":   "scikit-learn",
    "pandas":    "pandas",
    "numpy":     "numpy",
}

ok = True
for mod, pkg in packages.items():
    try:
        m = importlib.import_module(mod)
        print(f"  ✓  {pkg:<15}  {m.__version__}")
    except ImportError:
        print(f"  ✗  {pkg:<15}  NO INSTALADO")
        ok = False

sys.exit(0 if ok else 1)
PYEOF

# ── Resumen ───────────────────────────────────────────────────────────────────
echo ""
log "════════════════════════════════════════════════════"
log " Instalación completada."
log " Para ejecutar el pipeline:"
echo ""
echo "   conda activate $ENV_NAME"
echo "   cd $SCRIPT_DIR"
echo "   python forecast_pipeline.py --skip-tuning   # rápido"
echo "   python forecast_pipeline.py --trials 50     # con tuning"
echo ""
log "════════════════════════════════════════════════════"
