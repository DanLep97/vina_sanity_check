# Vina sanity check
This repository contains all necessary scripts to replicate the PoseBuster's Vina results as presented in their original [paper](https://doi.org/10.1039/d3sc04185a) using the ML docking review [pre-print](https://doi.org/10.26434/chemrxiv.15007152/v1) pipeline.

This repository contains the necessary scripts to re-dock the 428 protein-ligand complexes extracted from PoseBuster's benchmark and its analysis.

The RMSD was computed using PoseBuster's RMSD module based on RDKit's [CalcRMS](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html).

## Prerequisite
### Conda environement and utilities
1. [Install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) if not already. Then, create a new environment with the right python version and activate it:
    - `conda create --name ml_review python=3.11`
    - `conda activate ml_review`
2. Install last version of torch from [their website](https://pytorch.org/get-started/locally/). GPU acceleration is recommanded but not necessary for quick replication, required for complete replication.
3. Install required packages using pip:
    - `pip install -r requirements.txt`
4. Install usalign:
    - `conda install -c bioconda usalign`
5. Install obabel:
    - `conda install conda-forge::openbabel`
6. Install [ADFR](https://ccsb.scripps.edu/adfr/) and update its path in the `src/vina/sanitycheck.py` file.

## Quick replication
The repository contains the processed outputs and analysis plots but can be used to replicate the results from scratch.

### Database
The database is ready to use with the 428 protein-ligand complexes already available in `databases/Vina/inputs`. They were downloaded from PoseBuster's zenodo [repository](https://zenodo.org/records/8278563).

### Docking
Run `src/vina/sanitycheck.py` for re-docking.

### Compute metrics
Run `src/results/combine.py`

### Explore metrics
Follow jupyter notebook in `src/results/viz.ipynb`
