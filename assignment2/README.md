# Name: John Isak Texas Falk
# Student Number: 16095283

# Assignment 2: inpainting datasets

# Setup

## Python version

Version 2.7

## Library dependencies

Tensorflow 0.12.1
matplotlib 2.0.0
seaborn 0.7.1
tqdm 4.11.1 (progress bar, small library)
numpy: 1.11.3
scipy: 0.18.1

## Installing from conda

Assuming that you have conda installed on the computer, installing the environment is straightforward. Without being in a virtual environment, from the command line type the command

```
conda env create -f environment.yml
```

from the parent directory. This will create the same environment as mine which you can remove after you are done.

# Directories

## Models

The model directories use a naming convention of '{name}{cell type}{number of units}-lr{learning rate}\_bs{batch size}\_dc{decay rate}\_{timestamp}'.

## Code

The code directory consist of a data folder, a src folder and a Makefile.

### Makefile

The Makefile have two commands, both for evaluating the test/train accuracy/loss for task1 and task2. These are the following:

```
make task1
```

and

```
make task2
```

### data

The data folder consist of all the data needed to train and evaluate the models. the raw folder in the data folder holds the raw mnist data sets downloaded from Yann LeCun's website. If you want to do anything with the data, make sure to download the data (this should be done automatically if the zipped data is not found) or copy the data to this folder.

inpainting_data consist of the .npy files for task 3. The saved .npy files for task 3b are called 'optimal\_inpaintings\_1pixel\_patch.npy' and 'optimal\_inpaintings\_2x2pixel\_patch.npy'.

### src

Source directory consisting of all of the relevant python (and other) files for the assignment.

#### preprocess.py

File keeping all of the preprocessing functions, mainly the binarize function.

#### SETTINGS.py

Definitions of variables and constants. Important as it defines the different paths with respect to the system at use.

#### task1.py

Train any of the models for task 1. Relies on the auxiliary functions and classes defined in tf_utils1.py

#### tf_utils1.py

Helper functions for task1.py

#### task2a.py

Train any of the models for task 2. Relies on the auxiliary functions and classes defined in tf_utils2.py

#### task2b.py

File consisting of all of the needed function to generate all of the possible images and cross entropies for task 2b.

#### task3.py

File consisting of all of the needed functions to generate the inpaintings and cross entropies for task3 b.

#### visualize.py

Not used.

## Report

### 16095283_report.pdf

The report in pdf format.

### report_files

All of the files used to generate the report including image files and output text files.
