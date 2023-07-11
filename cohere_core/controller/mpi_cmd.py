import time
import os
import sys
import subprocess
import argparse
import cohere_core.utilities.utils as ut

def run_with_mpi(lib, conf_file, datafile, dir, devices):
    start_time = time.time()
    conf_map = ut.read_config(conf_file)
    if conf_map is None:
        return
    if 'ga_generations' in conf_map and conf_map['ga_generations'] > 1:
        script = '/reconstruction_GA.py'
#        devices = devices[:len(devices)//2]
        devices.sort()
        print('devices', devices)
    else:
        script = '/reconstruction_multi.py'
    script = os.path.realpath(os.path.dirname(__file__)).replace(os.sep, '/') + script
    command = ['mpiexec', '-n', str(len(devices)), 'python', script, lib, conf_file, datafile, dir, str(devices)]
    result = subprocess.run(command, stdout=subprocess.PIPE)
    run_time = time.time() - start_time
    print('GA reconstruction took', run_time, 'seconds')


def main(arg):
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("lib", help="lib")
    parser.add_argument("conf_file", help="conf_file")
    parser.add_argument("datafile", help="datafile")
    parser.add_argument('dir', help='dir')
    parser.add_argument('dev', help='dev')

    args = parser.parse_args()
    dev = ast.literal_eval(args.dev)
    run_with_mpi(args.lib, args.conf_file, args.datafile, args.dir, dev)


if __name__ == "__main__":
    print('args', sys.argv)
    exit(main(sys.argv[1:]))
