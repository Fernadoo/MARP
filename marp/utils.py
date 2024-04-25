import os

import numpy as np


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
                        row.append(0)
                    elif char == '@':
                        row.append(1)
                    else:
                        continue
                layout.append(row)
            line = f.readline()
    return np.array(layout)
