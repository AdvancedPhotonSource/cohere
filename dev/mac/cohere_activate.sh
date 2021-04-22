ENVS_DIR="$(dirname "$CONDA_PREFIX")"
CONDA_DIR="$(dirname "$ENVS_DIR")"
PKG_DIR="$CONDA_DIR/pkgs"
COH_DIR=($PKG_DIR/cohere-1.2-*_0)
RECIPE=$COH_DIR/info/recipe
export DYLD_LIBRARY_PATH="$RECIPE/lib"
