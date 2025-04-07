# efit2desc
Python tool for converting an EFIT g-file to a (fixed-boundary) DESC Equilibrium

## Installation

In order to use the scripts in this repo, one must have DESC installed and `omfit-classes` (the full OMFIT is not needed, just the subset contained in the package `omfit-classes`). The `requirements.txt` file contains [all of the needed packages](https://github.com/gafusion/OMFIT-source/issues/7110) for installing `omfit-classes` as well as `DESC`, so you should be able to do:

```bash
# first create your environment, using conda here as an example
conda create --name efit2desc 'python>=3.9, <=3.12'
# activate the environemnt
conda activate efit2desc
# install the requirements
# the --prefer-binary flag was found to be necessary when installing on a mac
pip install -r requirements.txt --prefer-binary
```

## Usage

