"""
Tweaks for yaml loaded and dumper.

Taken from Cobaya, by Torrado & Lewis (https://cobaya.readthedocs.io)
"""

import yaml
import numpy as np
from collections import OrderedDict
from typing import Mapping


def yaml_dump(obj, stream=None, **kwds):
    """
    drop-in replacement for the yaml dumper with some tweaks:

    - Order is preserved in dictionaries and other mappings
    - Tuples are dumped as lists
    - Numpy arrays (``numpy.ndarray``) are dumped as lists
    - Numpy scalars are dumped as numbers, preserving type
    """

    class CustomDumper(yaml.Dumper):
        pass

    # Make sure dicts preserve order when dumped
    # (This is still needed even for CPython 3!)
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

    CustomDumper.add_representer(dict, _dict_representer)
    CustomDumper.add_representer(Mapping, _dict_representer)
    CustomDumper.add_representer(OrderedDict, _dict_representer)

    # Dump tuples as yaml "sequences"
    def _tuple_representer(dumper, data):
        return dumper.represent_sequence(
            yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG, list(data))

    CustomDumper.add_representer(tuple, _tuple_representer)

    def _numpy_array_representer(dumper, data):
        return dumper.represent_sequence(
            yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG, data.tolist())

    CustomDumper.add_representer(np.ndarray, _numpy_array_representer)

    def _numpy_int_representer(dumper, data):
        return dumper.represent_int(data)

    CustomDumper.add_representer(np.int64, _numpy_int_representer)

    def _numpy_float_representer(dumper, data):
        return dumper.represent_float(data)

    CustomDumper.add_representer(np.float64, _numpy_float_representer)

    # Dump!
    return yaml.dump(obj, stream, CustomDumper, allow_unicode=True, **kwds)
