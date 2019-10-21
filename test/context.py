import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pyquat as pq
from scipy.linalg import norm

import lincov.frames as frames
