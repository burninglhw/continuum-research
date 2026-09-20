#!/usr/bin/env bash
set -euo pipefail

BASE=/export/home/ext.luohaowen1/continuum
SETUP="$BASE/reproduction/conda-setup"
FORGE="$BASE/tooling/miniforge3"
PREFIX="$BASE/envs/continuum"
INSTALLER="$SETUP/downloads/Miniforge3-26.7.2-0-Linux-x86_64.sh"
AVAILABLE=$(df -Pk "$BASE" | awk 'NR==2 {print $4}')
test "$AVAILABLE" -gt 20971520 || { echo "Less than 20 GiB available; refusing installation"; exit 1; }
mkdir -p "$SETUP/downloads" "$SETUP/logs" "$BASE/tooling/conda-home" "$BASE/tooling/conda-packages" "$BASE/envs"
export HOME="$BASE/tooling/conda-home"
export CONDA_PKGS_DIRS="$BASE/tooling/conda-packages"
export CONDA_ENVS_PATH="$BASE/envs"
export CONDARC="$SETUP/condarc"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
export CONDA_OVERRIDE_CUDA=12.8
if [ ! -x "$FORGE/bin/conda" ]; then
  test ! -e "$FORGE" || { echo "Target exists but is not a valid Conda installation"; exit 1; }
  if [ ! -f "$INSTALLER" ]; then
    curl --fail --location --retry 2 --connect-timeout 20 --max-time 300 \
      https://github.com/conda-forge/miniforge/releases/download/26.7.2-0/Miniforge3-Linux-x86_64.sh \
      --output "$INSTALLER"
  fi
  echo "281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05  $INSTALLER" | sha256sum --check
  bash "$INSTALLER" -b -p "$FORGE"
fi
"$FORGE/bin/conda" --version
if [ ! -x "$PREFIX/bin/python" ]; then
  test ! -e "$PREFIX" || { echo "Environment target already exists unexpectedly"; exit 1; }
  "$FORGE/bin/conda" create --yes --prefix "$PREFIX" --override-channels -c conda-forge python=3.11 pip
fi
"$PREFIX/bin/python" --version
"$FORGE/bin/conda" list --prefix "$PREFIX" --explicit > "$SETUP/conda-explicit.txt"
