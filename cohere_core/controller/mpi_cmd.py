import time
import os
import subprocess
import argparse


def run_with_mpi(ga_method, lib, conf_file, datafile, dir, devices, hostfile=None):
    start_time = time.time()
    if ga_method is None:
        script = '/reconstruction_multi.py'
    else:
        script = '/reconstruction_GA.py'

    script = os.path.realpath(os.path.dirname(__file__)).replace(os.sep, '/') + script
    if hostfile is None:
        command = ['mpiexec', '-n', str(len(devices)), 'python', script, lib, conf_file, datafile, dir, str(devices)]
    else:
        command = ['mpiexec', '-n', str(len(devices)), '--hostfile', hostfile, 'python', script, lib, conf_file, datafile, dir, str(devices)]

    subprocess.run(command, capture_output=False)

    run_time = time.time() - start_time
    if ga_method is None:
        print(f'multiple reconstructions took {run_time} seconds')
    else:   # fast GA
        print(f'GA reconstructions for directory {dir} took {run_time} seconds')


def main():
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
    exit(main())
