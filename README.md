# ML_for_kirigami_design (PyTorch version)
Python package to model and to perform topology optimization for graphene kirigami using deep learning. We use convolutional neural networks (similar to VGGNet architecure) for regression. 

## General usage 

* Python script to create and test fine grid dataset from coarse grid dataset is avalaible in `bin\create-and-test-fine-grid.py`

1. Regression and optimization 
* A python code to perform regression with TensorFlow is avalaible in `notebooks\CNN\01-regression.ipynb`
* A python code to perform search optimal design with TensorFlow is avalaible in `notebooks\CNN\02-search-designs.ipynb`

2. Dataset 
* Raw dataset of coarse-grained configs is avaialble in `raw`. This dataset generated using AIREBO potential with 1.7 mincutoff which is the default of CH.airebo.


This package is still under developement. More features will be added soon.

## Building the conda environment

All dependencies are included in `environment.yml` file. We can either have environment setup in our own project directory 
or in the envs directory of the miniconda folder. The advantage of installing in miniconda folder is that we can use same env
for different packages and it saves bandwidth and storage. The first method on the other hand is more robust. Prcedure is as follows

First of all if mamba do not exists then install it via
```bash
conda activate base
conda install mamba -n base -c conda-forge
```

To setup env in project dir 
```bash
mamba env create --prefix ./env --file environment.yml --force
```
or to setup in miniconda envs dir
```bash
mamba env create --name pytorch-1.12.0 --file environment.yml --force
```

## Update the conda env

If you add (remove) dependencies to (from) the environment.yml file or the requirements.txt file
after the environment has already been created, then you can re-create the environment with the
following command.

To update env in project `./env` dir
```bash
conda activate base
mamba env update --prefix ./env --file environment.yml  --prune
```

To update in Miniconda envs dir
```bash
conda actiavte base
mamba env update --name pytorch-1.12.0 --file environment.yml  --prune
```

## Remove the conda env

If you are low on storage and want to completely remove the conda env. You can do it by 

```bash
conda deactivate
conda env remove --prefix ./env
# or to remove from conda env dirs
conda env remove --name pytorch-1.12.0
```

## References
* <a href="https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.042006" style="color:#268cd7">[1] **Paul Z. Hanakata**, E. D. Cubuk, D. K. Campbell, H.S. Park, *Phys. Rev. Research*, 2, 042006(R) (2020).</a>
* <a href="https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.255304" style="color:#268cd7">[2] **Paul Z. Hanakata**, E. D. Cubuk, D. K. Campbell, H.S. Park, *Phys. Rev. Lett*, 121, 255304  (2018).</a>
* [[3] Paul Z. Hanakata](https://github.com/phanakata/ML_for_kirigami_design.git)

