#!/usr/bin/env bash
set -euo pipefail

BASE=/export/home/ext.luohaowen1/continuum
SETUP="$BASE/reproduction/conda-setup"
PREFIX="$BASE/envs/continuum"
export HOME="$BASE/tooling/conda-home"
export CONDARC="$SETUP/condarc"
export CONDA_PKGS_DIRS="$BASE/tooling/conda-packages"
export CONDA_ENVS_PATH="$BASE/envs"
export CONDA_OVERRIDE_CUDA=12.8
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LITELLM_LOCAL_MODEL_COST_MAP=True
unset PYTHONPATH
"$BASE/tooling/miniforge3/bin/conda" env config vars set --prefix "$PREFIX" \
  PYTHONNOUSERSITE=1 PIP_CONSTRAINT="$SETUP/constraints.txt"
"$PREFIX/bin/python" "$SETUP/verify_environment.py"
"$PREFIX/bin/vllm" --version > "$SETUP/vllm-version.txt" 2>&1
"$PREFIX/bin/mini-extra" swebench --help > "$SETUP/mini-swebench-help.txt" 2>&1
"$PREFIX/bin/python" -m pip freeze > "$SETUP/pip-freeze.txt"
"$BASE/tooling/miniforge3/bin/conda" list --prefix "$PREFIX" --explicit > "$SETUP/conda-explicit.txt"
"$BASE/tooling/miniforge3/bin/conda" env export --prefix "$PREFIX" > "$SETUP/environment.yml"
du -sh "$BASE/tooling" "$PREFIX" "$BASE/workspaces/continuum-316a587" "$SETUP"
df -h "$BASE"
test -z "$(git -C "$BASE/vllm-continuum" status --porcelain)"
