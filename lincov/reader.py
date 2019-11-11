from lincov.yaml_loader import YamlLoader
import math

from pyarrow.lib import ArrowIOError

import pandas as pd

def find_block(time, block_dt):
    raw_id = time / block_dt
    return int(math.floor(raw_id)) + 1

def load_window(loader, label, start, end, name = 'state_sigma'):
    config = YamlLoader(label)
    start_block_id = find_block(start, config.block_dt)
    end_block_id   = find_block(end,   config.block_dt)

    # Read one block
    if start_block_id == end_block_id:
        filename = 'output/{}/{}.{:04d}.feather'.format(label, name, start_block_id)
        return pd.read_feather(filename)

    # Read multiple blocks
    frames = []
    for block_id in range(start_block_id, end_block_id+1):
        filename = 'output/{}/{}.{:04d}.feather'.format(label, name, block_id)
        frames.append( pd.read_feather(filename) )

    return pd.concat(frames)

def load_sample(label, start, end, name = 'state_sigma'):
    """Only load one entry from each block (except from the last, where we
    load first and last)."""

    frames = []
    for ii in range(start, end+1):
        filename = 'output/{}/{}.{:04d}.feather'.format(label, name, ii)

        try:
            if ii == end:
                frame = pd.read_feather(filename)
                frames.append(frame.iloc[0:1])
                frames.append(frame.iloc[-1:])
            else:
                frames.append( pd.read_feather(filename).iloc[0:1] )
        except ArrowIOError:
            continue

    return pd.concat(frames)    
