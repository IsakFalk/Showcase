# Name: John Isak Texas Falk
# Student Number: 16095283

# assignment-3-reinforcement-learning-IsakFalk

# Setup

## Python version

Version 3.6

## Library dependencies

Tensorflow 1.0.0
matplotlib 2.0.0
seaborn 0.7.1
numpy: 1.12.0
scipy: 0.19.0

## Installing from conda

Assuming that you have conda installed on the computer, installing the environment is straightforward. Without being in a virtual environment, from the command line type the command

```
conda env create -f environment.yml
```

from the parent directory. This will create the same environment as mine which you can remove after you are done.

If not you could just install it with pip using whatever virtual environment you like the most.

# Directories

## Report

This directory only contain the pdf of the report for the assignment.

## models

In this directory we have one folder 'best_models' which contain the optimal saved models for each of the exercises which need saved models. For A, this is 3, 4, 5, 6, 7 and for B, this is one for each game, Pong, MsPacman, Boxing.

## code

The python scripts for all of the models and other needed classes and functions. Most of the python scripts have command line arguments. Finding out which ones can be set is simple, run the file with

```
python A1.py -h
```

where you may substitute A1.py for whatever file you want to run.

We have the following.

### SETTINGS.py

Paths, constants and variable definitions used in the scripts.

### tf_utils.py

Utility functions with regards to tensorflow.

### A_utils.py

Python file containing all of the needed classes for the different agents, estimators and environments. All of the estimators has been given names which reflect what kind of Q-value function they are, going with the earlier estimators first. Same thing for the agents. These classes are used to train and evaluate the performance of the agents/estimator functions in the scripts A'x'.py where 'x' is from 1 to 7.

### B.py

Same as A_utils.py but for part B of the assignment.

### A1.py

Script for running A1.

### A2.py

Script for running A2.

### A3.py

Script for running A3.

### A45.py

Script for running A4 and A5.

### A6.py

Script for running A6.

### A7.py

Script for running A7.

### B1.py

Script for running B1.

### B2.py

Script for running B2.

### B3.py

Script for running B3.

### B4.py

Script for running B4. This script can be used to check the models since it loads the saved models and evaluates them. Number of episodes to average over can be selected from the command line.

### evaluate_A.py

Evaluate all of the models that was saved for A.





