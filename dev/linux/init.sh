#!/bin/sh

rm setup.py
mv setup.templ setup.py

echo -n "enter ArrayFire installation directory > "
read af_dir
AF='AF_DIR'
sed -i 's?'$AF'?'$af_dir'?g' setup.py
sed -i 's?'$AF'?'$af_dir'?g' setenv.sh

export LD_LIBRARY_PATH=$af_dir/lib64

python setup.py install
