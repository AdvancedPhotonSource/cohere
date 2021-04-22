python setup.py install  --prefix=$PREFIX

mkdir -p "$PREFIX/etc/conda/activate.d"
cp cohere_activate.sh $PREFIX/etc/conda/activate.d

mkdir -p "$PREFIX/etc/conda/deactivate.d"
cp cohere_deactivate.sh $PREFIX/etc/conda/deactivate.d
