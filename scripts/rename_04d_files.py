#!/usr/bin/env python3

"""Script is for renaming files to have more zeros in the index number
padding. This was mainly used to transition the analysis files when
the numbering scheme changed.

"""

from pathlib import Path
import os
import sys

if __name__ == '__main__':

    dir = Path(sys.argv[1])

    file_list = [f for f in dir.glob('*') if f.is_file()]
    
    for filename in file_list:
        fields = filename.name.split('.')
        label  = fields[0]
        count  = int(fields[1])
        ext    = fields[2]

        os.rename(filename, os.path.join(filename.parent, "{}.{:04d}.{}".format(label, count, ext)))
