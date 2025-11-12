# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

from pathlib import Path
import re


def main():
    lib_dir = Path(__file__).parent.as_posix()
    cplib = Path(f"{lib_dir}/cplib.py").read_text()
    cohlib = Path(f"{lib_dir}/cohlib.py").read_text()
    nplib = Path(f"{lib_dir}/nplib.py").read_text()
    torchlib = Path(f"{lib_dir}/torchlib.py").read_text()

    p0 = re.compile(r"    def (\S+)\(")

    needs_subclasshook = []
    needs_abstractmethod = []
    needs_npfunction = []
    needs_torchfunction = []
    for funk in re.findall(p0, cplib):
        p1 = re.compile(rf"callable\(subclass.{funk}\)")
        p2 = re.compile(rf"hasattr\(subclass, '{funk}'\)")
        p3 = re.compile(rf"def {funk}\(")
        if not (re.search(p1, cohlib) and re.search(p2, cohlib)):
            needs_subclasshook.append(funk)
        if not re.search(p3, cohlib):
            needs_abstractmethod.append(funk)
        if not re.search(p3, nplib):
            needs_npfunction.append(funk)
        if not re.search(p3, torchlib):
            needs_torchfunction.append(funk)

    names = ["cohlib", "cohlib.__subclasshook__", "nplib", "torchlib"]
    libs = [needs_subclasshook, needs_abstractmethod, needs_npfunction, needs_torchfunction]
    if not any([len(lib) for lib in libs]):
        print("All devlibs are up to date!")
        return
    for name, lib in zip(names, libs):
        if len(lib):
            print(f"\nMissing from {name}:")
            for s in lib:
                print(f"\t{s}")


if __name__ == "__main__":
    main()