#!/usr/bin/env bash
set -euo pipefail

BASE=/export/home/ext.luohaowen1/continuum
SETUP="$BASE/reproduction/conda-setup"
PREFIX="$BASE/envs/continuum"
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null
export UV_CACHE_DIR="$SETUP/uv-cache"
export UV_CONCURRENT_DOWNLOADS=4
export UV_CONCURRENT_INSTALLS=4
export UV_LINK_MODE=hardlink
export TMPDIR
TMPDIR=$(mktemp -d /tmp/continuum-install-1039.XXXXXX)
printf '%s\n' "$TMPDIR" > "$SETUP/install-tmpdir.txt"
cd "$SETUP"
AVAILABLE=$(df -Pk "$BASE" | awk 'NR==2 {print $4}')
test "$AVAILABLE" -gt 20971520 || { echo "Less than 20 GiB available"; exit 1; }
"$PREFIX/bin/uv" pip install --python "$PREFIX/bin/python" --default-index https://pypi.org/simple \
  --dry-run --only-binary :all: -r requirements-runtime.txt -c constraints.txt > dependency-plan.log 2>&1
tail -12 dependency-plan.log
"$PREFIX/bin/uv" pip install --python "$PREFIX/bin/python" --default-index https://pypi.org/simple \
  --only-binary :all: -r requirements-runtime.txt -c constraints.txt
"$PREFIX/bin/python" -m pip check
"$PREFIX/bin/python" -m pip freeze > "$SETUP/pip-before-fork.txt"
