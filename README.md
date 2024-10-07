# Python Bootcamp 2024

## Setup / installation

### Heavy installation

Install the full anaconda environment following the instructions in: https://docs.anaconda.com/free/anaconda/install/index.html

Install additional packages not included in the anaconda installation:

```
mamba install ffmpeg ipympl jupytext nodejs
```

### Lightweight installation

You should first install the `conda` package manager or it's alternative `mamba` (recommended): https://mamba.readthedocs.io/en/latest/micromamba-installation.html

Create the environment with the minimal number of packages

```
mamba env create -f environment.yml
```

To be able to use the environment you have to activate it:

```
conda activate PythonBootcamp2024
```

## Launching a notebook server

Start a jupyterlab notebook server

`jupyter lab`

Or a "traditional" jupyter notebook server

`jupyter notebook`



