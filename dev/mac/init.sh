#!/bin/sh

rm setup.py
mv setup.templ setup.py

echo -n "enter ArrayFire installation directory > "
read af_dir
sed -i '' "s#AF_DIR#$af_dir#g" setup.py
sed -i '' "s#AF_DIR#$af_dir#g" setenv.sh

export DYLD_LIBRARY_PATH=$af_dir/lib

python setup.py install
