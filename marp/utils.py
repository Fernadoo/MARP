import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Marker:
    CELL = 0
    BLOCK = 1
    IMPORT = 2
    EXPORT = 3
    BATTERY = 8
    ACCESSIBLE = [CELL, BATTERY]
    INACCESSIBLE = [BLOCK, IMPORT, EXPORT]


def parse_map_from_file(map_config):
    PREFIX = 'marp/layouts/'
    POSTFIX = '.map'
    if not os.path.exists(PREFIX + map_config + POSTFIX):
        raise ValueError('Map config does not exist!')
    layout = []
    with open(PREFIX + map_config + POSTFIX, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith('#'):
                pass
            else:
                row = []
                for char in line:
                    if char == '.':
                        row.append(Marker.CELL)
                    elif char == '@':
                        row.append(Marker.BLOCK)
                    elif char == 'I':
                        row.append(Marker.IMPORT)
                    elif char == 'E':
                        row.append(Marker.EXPORT)
                    elif char == 'B':
                        row.append(Marker.BATTERY)
                    else:
                        continue
                layout.append(row)
            line = f.readline()
    return np.array(layout)
