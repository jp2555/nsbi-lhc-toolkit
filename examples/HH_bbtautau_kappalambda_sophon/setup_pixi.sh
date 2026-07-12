#!/usr/bin/env bash
# One-time pixi setup for the Option A driver's CPU stages (Perlmutter or any linux/mac).
# Installs pixi user-locally if missing, builds the toolkit's nsbi-env, verifies the
# imports the driver actually uses, and prints the per-shell exports.
#
# Usage:
#   ./setup_pixi.sh          # CPU env only (all run_option_a.sh CPU stages)
#   ./setup_pixi.sh --gpu    # + nsbi-env-gpu (only needed for the sophon sbatch scripts;
#                            #   EveNet train/predict use the shifter image, NOT pixi)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# Keep the package cache off the small $HOME quota; it is re-downloadable, so the
# pscratch purge policy is harmless here. The envs themselves live in $REPO_ROOT/.pixi/.
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-${PSCRATCH:-$HOME}/.pixi-cache}"

# 1. pixi itself (installs to ~/.pixi/bin; NERSC $HOME is persistent)
if ! command -v pixi >/dev/null 2>&1; then
    echo "== installing pixi to \$HOME/.pixi/bin"
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
fi
echo "== pixi $(pixi --version 2>/dev/null || true)  cache: $PIXI_CACHE_DIR"

# 2. build the env(s) from the repo's pixi.toml/pixi.lock
cd "$REPO_ROOT"
pixi install -e nsbi-env
[ "${1:-}" = "--gpu" ] && pixi install -e nsbi-env-gpu

# 3. verify exactly what the driver's CPU stages import
pixi run -e nsbi-env python - <<'EOF'
import numpy, uproot, sklearn, yaml, matplotlib, torch
print(f"numpy {numpy.__version__} | uproot {uproot.__version__} | "
      f"sklearn {sklearn.__version__} | torch {torch.__version__}")
print("nsbi-env OK")
EOF

cat <<'EOF'

Done. Per shell (or add to ~/.bashrc):
    export PATH="$HOME/.pixi/bin:$PATH"
    export CONVERT_PY="pixi run -e nsbi-env python"
Then, from examples/HH_bbtautau_kappalambda_sophon/:
    ./run_option_a.sh ceiling
EOF
