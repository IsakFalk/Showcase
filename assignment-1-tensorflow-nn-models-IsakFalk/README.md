# assignment-1-tensorflow-nn-models-IsakFalk

## Evaluating the models

I will assume that the latest conda package manager is installed and that you are using a UNIX based system (Sorry Windows). Conda can be downloaded from here https://www.continuum.io/downloads.
I make use of a makefile in order to streamline the process. Makefiles should work on most UNIX based systems. If not, I will include what commands to run from the terminal.

## Make

### 1.

Run the command

```{r, engine='bash', count_lines}
make environment
```

in the environment of the local git repository, the same directory as this README and Makefile can be found.
This will set up the correct environment with the needed Python dependencies.

### 2.

Run the command

```{r, engine='bash', count_lines}
source activate tf1
```

which will change your environment to that of the first assignment.

### 3.

Run the command

```{r, engine='bash', count_lines}
make evaluate
```

which will run the models from the saved file, tf checkpoint files for part 1 and python pickle files for part 2,
on the test set.

## Terminal

These are essentially the same commands as in make. Run these in order.

### 1.

```{r, engine='bash', count_lines}
conda env create -f environment.yml
```

### 2.

```{r, engine='bash', count_lines}
source activate tf1
```

### 3.

```{r, engine='bash', count_lines}
cd ./code/Part1/ && python evaluate_part1.py && cd ./../Part2 && python evaluate_part2.py && cd ./../../
```
