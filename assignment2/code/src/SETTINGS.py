"""
Set variables, parameters and constants that should be immutable.
"""

import os
import sys

curfilePath = os.path.abspath(__file__)
curDir = os.path.abspath(os.path.join(curfilePath, os.pardir)) # this will return current directory in which python file resides.
PROJECT_ROOT = os.path.abspath(os.path.join(curDir, os.pardir)) # Parent directory

PATHS = {
    "project_root": PROJECT_ROOT,
    "data_dir": os.path.join(PROJECT_ROOT, 'data', 'raw'),
    "save_dir": os.path.join(PROJECT_ROOT, 'data', 'save'),
    "best_dir": os.path.join(PROJECT_ROOT, '..', 'models'),
    "inpainting_data_dir": os.path.join(PROJECT_ROOT, 'data', 'inpainting_data'),
    "img_dir": os.path.join(PROJECT_ROOT, '..', 'report', 'img'),
    "npy_dir": os.path.join(PROJECT_ROOT, 'data', 'inpainting_data')
}

# task 2 models

U32MODEL = 'task2GRU32st1-lr0.001_bs256_dc0.99_1488480798'
U64MODEL = 'task2GRU64st1-lr0.001_bs256_dc0.99_1488488776'
U128MODEL = 'task2GRU128st1-lr0.001_bs256_dc0.99_1488496514'
U32STACKEDMODEL = 'task2GRU32st3-lr0.001_bs256_dc0.99_1488504944'

U32MODELPATH = os.path.join(PATHS['best_dir'], U32MODEL)
U64MODELPATH = os.path.join(PATHS['best_dir'], U64MODEL)
U128MODELPATH = os.path.join(PATHS['best_dir'], U128MODEL)
U32STACKEDMODELPATH = os.path.join(PATHS['best_dir'], U32STACKEDMODEL)

# Hyperparameters decided from the command line

# Constants
N_INPUT = 1
N_CLASSES = 10
N_STEPS = 28*28
INPUT_DROPOUT_PROB = 1.0
OUTPUT_DROPOUT_PROB = 1.0
