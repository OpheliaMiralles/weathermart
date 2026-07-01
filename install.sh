#!/usr/bin/env bash
WEATHERMART_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VENV_DIR="$WEATHERMART_ROOT/.venv"
ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# create environment
uv venv -p3.12
uv sync --all-extras
echo "Activating venv..."
source "$ACTIVATE_SCRIPT"
echo "Python version:"
python --version
echo "Removing unrelated PyPI coda package if present..."
python -m pip uninstall -y coda || true
mkdir -p ~/software
cd ~/software
mkdir -p coda
cd coda
wget https://github.com/stcorp/coda/releases/download/2.25.6/coda-2.25.6.tar.gz
tar -xzf coda-2.25.6.tar.gz
cd coda-2.25.6
mkdir -p $HOME/coda_install_py312
CODA_SRC=~/software/coda/coda-2.25.6
CODA_PREFIX=$HOME/coda_install_py312

echo "CODA source: $CODA_SRC"
echo "CODA install prefix: $CODA_PREFIX"
echo "Configuring CODA with Python from venv..."
./configure \
  --prefix="$CODA_PREFIX" \
  --enable-python \
  PYTHON=$(which python)
echo "Building CODA..."
make -j4
echo "Installing CODA..."
make install
echo "Installing CODA Python into venv..."
echo "$CODA_PREFIX/lib/python3.12/site-packages" \
    > $VENV_DIR/lib/python3.12/site-packages/coda_path.pth
echo "Verifying CODA Python bindings..."
python -c "import coda; print(coda.__file__); assert hasattr(coda, 'open')"

cd ~/software
mkdir -p harp
cd harp
wget https://github.com/stcorp/harp/releases/download/1.29/harp-1.29.tar.gz
tar -xzf harp-1.29.tar.gz # CODA + HARP definitions are needed for IASI data
echo "Patching venv activate script..."
cat << 'EOF' >> "$ACTIVATE_SCRIPT"

# --- CODA IO environment ---
export CODA_HOME="$HOME/coda_install_py312"
export PATH="$CODA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CODA_HOME/lib:$LD_LIBRARY_PATH"
export CODA_DEFINITION="$CODA_HOME/share/coda/definitions"
export CODA_DEFINITION=$HOME/software/harp/harp-1.29/definitions:$CODA_DEFINITION
EOF
cd "$WEATHERMART_ROOT"
echo "Done! Activate the environment in your shell with:"
echo "source $ACTIVATE_SCRIPT"
