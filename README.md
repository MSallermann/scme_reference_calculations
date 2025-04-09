# bulk_properties_SCME
This repository contains workflows to compute the bulk properties of the SCME potential. It uses `micromamba` to manage a virtual python environment and `snakemake` to define workflows.

## Installation of dependencies

### 1. Setting up the conda environment
Get micromamba
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Install the environment
```bash
micromamba create -f environment.yml
```

__Activate the environment__!

This is important for all subsequent steps
```bash
micromamba activate scme_bulk_properties_env
```

### 2. Install scme into the environment

The SCME code is currently not publicly available. If you have access to the private gitlab repository you can clone the rework branch and install it, like so

```bash
git clone git@gitlab.com:MSallermann/scmecpp.git --branch rework
cd scmecpp
micromamba activate scme_bulk_properties_env
pip install .
```