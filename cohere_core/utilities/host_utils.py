import os
import sys
import argparse
import socket
import cohere_core.utilities as ut


def main(arg):
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("devices", help="devices")
    parser.add_argument("run_size", help="memory requirement for an operation")
    args = parser.parse_args()

    devs = ast.literal_eval(args.devices)
    run_size = ast.literal_eval(args.run_size)
    available = ut.get_avail_gpu_runs(devs[socket.gethostname()], run_size)
    print(socket.gethostname(), str(available))


if __name__ == "__main__":
    exit(main(sys.argv[1:]))