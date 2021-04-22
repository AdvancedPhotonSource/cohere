ENVS_DIR="$(dirname "$CONDA_PREFIX")"
CONDA_DIR="$(dirname "$ENVS_DIR")"
PKG_DIR="$CONDA_DIR/pkgs"
COH_DIR=($PKG_DIR/cohere-1.2-*_0)
RECIPE="$COH_DIR/info/recipe"
export LD_LIBRARY_PATH="$RECIPE/lib"

export LD_PRELOAD="$RECIPE/lib/libmkl_core.so":"$RECIPE/lib/libmkl_sequential.so"
