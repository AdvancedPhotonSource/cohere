import argparse
import socket
import cohere_core.utilities as ut


def main():
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("devices", help="configured devices")
    parser.add_argument("run_size", help="memory requirement for one job")
    args = parser.parse_args()

    # devices parameter is dict with host names as keys
    devs = ast.literal_eval(args.devices)
    run_size = ast.literal_eval(args.run_size)

    # host name in configuration can be full host name or a first part of full name
    # it needs to be matched with the hostname returned by socket utilities.
    configured_hosts = devs.keys()
    host_name = socket.gethostname()
    if host_name in configured_hosts:
        use_host_name = host_name
    else:
        use_host_name = host_name.split('.')[0]
        if not use_host_name in configured_hosts:
            return

    # The result available is a dictionary with key, value pars of GPU ID, number pf available jobs
    available = ut.get_avail_gpu_runs(devs[use_host_name], run_size)
    # the printed hostname and available devices will be received through pipe object by calling code
    print([use_host_name, available])


if __name__ == "__main__":
    exit(main())