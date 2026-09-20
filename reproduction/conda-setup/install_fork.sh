#!/usr/bin/env bash
set -euo pipefail

BASE=/export/home/ext.luohaowen1/continuum
SETUP="$BASE/reproduction/conda-setup"
PREFIX="$BASE/envs/continuum"
SOURCE="$BASE/workspaces/continuum-316a587"
COMMIT=316a58794a6ff86b216e579b74fd56ed0c5a911f
WHEEL="$SETUP/downloads/vllm-0.10.2-cp38-abi3-manylinux1_x86_64.whl"
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset PYTHONPATH
export TMPDIR
TMPDIR=$(mktemp -d /tmp/continuum-fork-1039.XXXXXX)
printf '%s\n' "$TMPDIR" > "$SETUP/fork-tmpdir.txt"
test "$(git -C "$BASE/vllm-continuum" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$BASE/vllm-continuum" status --porcelain)"
"$PREFIX/bin/python" "$SETUP/compare_kernel_sources.py"
mkdir -p "$BASE/workspaces"
if [ ! -d "$SOURCE" ]; then
  mkdir "$SOURCE"
  git -C "$BASE/vllm-continuum" archive "$COMMIT" | tar -xf - -C "$SOURCE"
  printf '%s\n' "$COMMIT" > "$SOURCE/.source-commit"
fi
test "$(cat "$SOURCE/.source-commit")" = "$COMMIT"
if [ ! -f "$WHEEL" ]; then
  curl --fail --location --retry 2 --connect-timeout 20 --max-time 600 \
    https://files.pythonhosted.org/packages/a2/1a/365479f413e7408b314c0237d6c929569874d5c002bc7c8b5a7fbf40c7d9/vllm-0.10.2-cp38-abi3-manylinux1_x86_64.whl \
    --output "$WHEEL"
fi
echo "e0cba6110483d9bf25c4402d8655cf78d366dd13e4155210980cc3480ed98b7b  $WHEEL" | sha256sum --check
if [ "${1:-}" = "--prepare-only" ]; then
  exit 0
fi
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL" \
  VLLM_TARGET_DEVICE=cuda SETUPTOOLS_SCM_PRETEND_VERSION=0.10.2+continuum.316a587 \
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.10.2+continuum.316a587 \
  "$PREFIX/bin/python" -m pip install --no-cache-dir --no-deps --no-build-isolation -e "$SOURCE"
"$PREFIX/bin/python" -m pip install --no-cache-dir --no-deps --no-build-isolation -e "$SOURCE/mini-swe-agent"
"$PREFIX/bin/python" -m pip check
"$PREFIX/bin/python" -m pip freeze > "$SETUP/pip-freeze.txt"
test -z "$(git -C "$BASE/vllm-continuum" status --porcelain)"
