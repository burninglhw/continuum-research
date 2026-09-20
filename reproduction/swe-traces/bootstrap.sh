set -euo pipefail
umask 077
base=/export/home/ext.luohaowen1/continuum
setup="$base/reproduction/swe-traces"
prefix="$base/envs/continuum-traces"
source_dir="$base/workspaces/continuum-316a587/mini-swe-agent"
export CONDA_OVERRIDE_CUDA=12.8
export CONDA_PKGS_DIRS="$base/tooling/conda-packages"
export CONDARC="$base/reproduction/conda-setup/condarc"
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null
export CUDA_VISIBLE_DEVICES=
export MSWEA_SILENT_STARTUP=1
export MSWEA_GLOBAL_CONFIG_DIR="$setup/agent-config"
unset PIP_CONSTRAINT PYTHONPATH
mkdir -p "$setup/agent-config" "$setup/source"
if [ ! -x "$prefix/bin/python" ]; then
    "$base/tooling/miniforge3/bin/conda" create --offline --prefix "$prefix" python=3.11.16 pip -y
fi
if [ ! -f "$setup/source/mini-swe-agent/pyproject.toml" ]; then
    cp -a "$source_dir" "$setup/source/mini-swe-agent"
fi
"$prefix/bin/python" -m pip install --no-cache-dir --index-url https://pypi.org/simple -r "$setup/requirements.txt" "$setup/source/mini-swe-agent"
"$prefix/bin/python" -m pip check
"$base/tooling/miniforge3/bin/conda" env config vars set --prefix "$prefix" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= MSWEA_SILENT_STARTUP=1 MSWEA_GLOBAL_CONFIG_DIR="$setup/agent-config"
"$prefix/bin/python" -m pip freeze > "$setup/pip-freeze.txt"
"$base/tooling/miniforge3/bin/conda" list --prefix "$prefix" --explicit > "$setup/conda-explicit.txt"
"$prefix/bin/python" -c 'import minisweagent, requests, pyarrow, numpy, transformers; print("Agent:",minisweagent.__version__); print("NumPy:",numpy.__version__); print("Transformers:",transformers.__version__)'
df -h "$base"
