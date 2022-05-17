# -*- coding: utf-8 -*-


def _fix_module_not_found_error():
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).parents[1]))


if __name__ != "__main__":
    _fix_module_not_found_error()
