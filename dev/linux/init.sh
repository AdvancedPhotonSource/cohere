#!/bin/sh

cp cohere/src_py/cyth/bridge_cpu.templ cohere/src_py/cyth/bridge_cpu.pyx
cp cohere/src_py/cyth/bridge_cuda.templ cohere/src_py/cyth/bridge_cuda.pyx
cp cohere/src_py/cyth/bridge_opencl.templ cohere/src_py/cyth/bridge_opencl.pyx

echo -n "enter ArrayFire installation directory > "
read af_dir
AF='AF_DIR'
AFLIB='AF_LIB'
sed -i 's?'$AF'?'$af_dir'?g' cohere/src_py/cyth/*.pyx
sed -i 's?'$AFLIB'?'$af_dir/lib64'?g' cohere/src_py/cyth/*.pyx


export LD_LIBRARY_PATH=$af_dir/lib64

python setup.py build_ext --inplace
python setup.py install
