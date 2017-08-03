"""
Settings and set constants used throughout the assignment.
"""

import os
import sys

# constants
GAMMA = 0.99
MAX_EP_LEN = 300
ENV_A = 'CartPole-v0'

curfilePath = os.path.abspath(__file__)
curDir = os.path.abspath(os.path.join(curfilePath, os.pardir)) # this will return current directory in which python file resides.
PROJECT_ROOT = os.path.abspath(os.path.join(curDir, os.pardir)) # Parent directory

PATHS = {
    "project_root": PROJECT_ROOT,
    "save_dir": os.path.join(PROJECT_ROOT, 'models', 'save'),
    "best_dir": os.path.join(PROJECT_ROOT, 'models', 'best_models'),
}

