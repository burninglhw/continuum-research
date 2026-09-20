#!/usr/bin/env bash
set -euo pipefail
BASE=/export/home/ext.luohaowen1/continuum
SETUP="$BASE/reproduction/conda-setup"
PREFIX="$BASE/envs/continuum"
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export UV_CACHE_DIR="$SETUP/uv-cache"
export UV_CONCURRENT_DOWNLOADS=8
export UV_CONCURRENT_INSTALLS=4
export UV_LINK_MODE=hardlink
cd "$SETUP"
"$PREFIX/bin/python" - <<'PY'
import re
from pathlib import Path
from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
setup = Path.cwd()
deferred = {canonicalize_name(parse_wheel_filename(path.name)[0]) for path in (setup / 'downloads').glob('*.whl')}
deferred.add('torch')
planned = re.findall(r'^\s+\+ ([a-zA-Z0-9_.-]+)==([^\s]+)', (setup / 'dependency-plan.log').read_text(), re.MULTILINE)
selected = [f'{name}=={version}' for name, version in planned if canonicalize_name(name) not in deferred]
(setup / 'small-packages.txt').write_text('\n'.join(selected) + '\n')
print('Small package count:', len(selected), flush=True)
PY
"$PREFIX/bin/uv" pip install --python "$PREFIX/bin/python" --default-index https://pypi.org/simple \
  --only-binary :all: --no-deps -r small-packages.txt -c constraints.txt
