#!/bin/bash
# ============================================================
# Dark Scaffold Cosmology - HIGH-RESOLUTION Simulation Suite
# ============================================================
# Runs all simulations with upscaled parameters.
# ============================================================

set -e

# ── Corsair drive base path ──────────────────────────────────
CORSAIR_BASE="/Volumes/Corsair_Lab/Home/Documents/Cosmology"

# Assuming the script is in the project root
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -d "${CORSAIR_BASE}" ]; then
    echo "ERROR: Corsair drive not mounted at ${CORSAIR_BASE}"
    exit 1
fi

# ── Redirect ALL temp I/O to Corsair ─────────────────────────
# import — before any Python-level guard can activate.
CORSAIR_TMP="${PROJECT_DIR}/.tmp"
mkdir -p "${CORSAIR_TMP}"

# Core temp dirs
export TMPDIR="${CORSAIR_TMP}"
export TEMP="${CORSAIR_TMP}"
export TMP="${CORSAIR_TMP}"
export TEMPDIR="${CORSAIR_TMP}"

# macOS-specific: override confstr(_CS_DARWIN_USER_TEMP_DIR)
export DARWIN_USER_TEMP_DIR="${CORSAIR_TMP}"
export DARWIN_USER_CACHE_DIR="${CORSAIR_TMP}"

export HOME="${PROJECT_DIR}"

export MPLCONFIGDIR="${CORSAIR_TMP}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

export FONTCONFIG_PATH="${CORSAIR_TMP}/fontconfig"
export FC_CACHEDIR="${CORSAIR_TMP}/fontconfig"
mkdir -p "${CORSAIR_TMP}/fontconfig"

# PIL/Pillow temp dir (PillowWriter GIF frames)
export PILLOW_CACHE_DIR="${CORSAIR_TMP}"

# NumPy / SciPy MKL + FFTW
export MKL_TMPDIR="${CORSAIR_TMP}"
export FFTW_WISDOM_DIR="${CORSAIR_TMP}/fftw"
mkdir -p "${CORSAIR_TMP}/fftw"

# Reduce thread temp buffers (fewer parallel temp allocations)
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

export XDG_CACHE_HOME="${CORSAIR_TMP}"
export XDG_DATA_HOME="${CORSAIR_TMP}"
export XDG_CONFIG_HOME="${CORSAIR_TMP}"
export XDG_RUNTIME_DIR="${CORSAIR_TMP}"


# Set CWD to Corsair — prevents any relative path from landing on disk0
cd "${PROJECT_DIR}"

# Force non-interactive matplotlib backend (no GUI windows)
export MPLBACKEND=Agg

PYTHON="${PYTHON:-python3}"

# ── Swap Redirection (Disabled) ────────────────────────────────
# dynamic_pager is unstable on modern macOS and causes SIGKILL (9).
# Swap remains on disk0, but temp files (majority of I/O) are redirected.
CORSAIR_SWAP="${CORSAIR_TMP}/swap" # Placeholder for logging only
echo "Swap redirection disabled for stability."

echo "============================================================"
echo " DARK SCAFFOLD COSMOLOGY — HIGH-RESOLUTION RUN"
echo " Project : ${PROJECT_DIR}"
echo " HOME    : ${HOME}"
echo " Swap    : ${CORSAIR_SWAP}"
echo " Temp Dir: ${CORSAIR_TMP}"
echo " Python  : $(${PYTHON} --version 2>&1)"
echo "============================================================"
echo ""

run_script() {
    local script="$1"
    echo "────────────────────────────────────────────────────────────"
    echo "▶  Running ${script} --hires"
    echo "────────────────────────────────────────────────────────────"
    ${PYTHON} "${PROJECT_DIR}/${script}" --hires
    echo ""
}

run_script scaffold_generator.py
run_script seeping_simulation.py
run_script nbody_simulation.py
run_script infall_simulation.py
run_script energy_budget.py
# Added statistical likelihood assessment
run_script theory_likelihood.py

# Clean up temp directory
echo "Cleaning up temp files..."
rm -rf "${CORSAIR_TMP}"

echo "============================================================"
echo " ALL HIGH-RESOLUTION SIMULATIONS COMPLETE"
echo "============================================================"

