from lincov.yaml_loader import YamlLoader
import math

import pandas as pd

def find_block(time, block_dt):
    raw_id = time / block_dt
    return int(math.floor(raw_id)) + 1

def load_window(loader, label, start, end, name = 'state'):
    config = YamlLoader(label)
    start_block_id = find_block(start, config.block_dt)
    end_block_id   = find_block(end,   config.block_dt)

    # Read one block
    if start_block_id == end_block_id:
        filename = 'output/{}/state.{}.feather'.format(label, start_block_id)
        return pd.read_feather(filename)

    # Read multiple blocks
    frames = []
    for block_id in range(start_block_id, end_block_id+1):
        filename = 'output/{}/state.{}.feather'.format(label, block_id)
        frames.append( pd.read_feather(filename) )

    return pd.concat(frames)
