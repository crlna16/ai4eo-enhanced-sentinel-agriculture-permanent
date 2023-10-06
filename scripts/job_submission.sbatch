#!/bin/bash

#SBATCH --job-name=superresolution
#SBATCH --time=0-24:00:00
#SBATCH --mem=0
#SBATCH --gres gpu
#SBATCH -c 24 # n CPUS
#SBATCH -G 1
#SBATCH --mail-type=ALL
#SBATCH --tasks=1
#SBATCH --account=hai_earth_03
#SBATCH --partition=booster

# output files
#SBATCH -o /p/project/hai_earth_03/caroline/gitlab/ai4eo-enhanced-sentinel-agriculture-permanent/scripts/jobs/ai4eo_superresolution_%j.out
#SBATCH -e /p/project/hai_earth_03/caroline/gitlab/ai4eo-enhanced-sentinel-agriculture-permanent/scripts/jobs/ai4eo_superresolution_%j.err

source $HOME/.bashrc
echo 'Starting script'
cd $PROJECT_hai_earth_02/caroline/gitlab/ai4eo-enhanced-sentinel-agriculture-permanent

source modules.sh
source .ai4eo_superresolution_env/bin/activate
#export PYTHONPATH="/p/software/juwelsbooster/stages/2023/software/PyTorch-Lightning/1.8.2-foss-2022a-CUDA-11.7/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/tensorboard/2.11.2-foss-2022a/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/tqdm/4.64.0-GCCcore-11.3.0/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/expecttest/0.1.3-GCCcore-11.3.0/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/protobuf-python/3.19.4-GCCcore-11.3.0/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/scikit-image/0.19.3-foss-2022a/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/imageio/2.22.2-foss-2022a/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/networkx/2.8.4-gcccoremkl-11.3.0-2022.1.0/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/netcdf4-python/1.6.1-gompi-2022a/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/mpi4py/3.1.4-gompi-2022a/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/xarray/2022.9.0-gcccoremkl-11.3.0-2022.1.0/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/Seaborn/0.12.1-gcccoremkl-11.3.0-2022.1.0/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/matplotlib/3.5.2-gcccoremkl-11.3.0-2022.1.0/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/Tkinter/3.10.4-GCCcore-11.3.0/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/dask/2022.10.2-foss-2022a/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/bokeh/2.4.2-gcccoremkl-11.3.0-2022.1.0/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/typing-extensions/4.3.0-GCCcore-11.3.0/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/Pillow-SIMD/9.2.0-GCCcore-11.3.0/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/PyYAML/6.0-GCCcore-11.3.0/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/h5py/3.7.0-GCCcore-11.3.0-serial/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/scikit-learn/1.1.2-foss-2022a/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/SciPy-bundle/2022.05-gcccoremkl-11.3.0-2022.1.0/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/pybind11/2.9.2-GCCcore-11.3.0/lib/python3.10/site-packages:/p/software/juwelsbooster/stages/2023/software/Python/3.10.4-GCCcore-11.3.0/easybuild/python"
export HYDRA_FULL_ERROR=1

srun --cpus-per-task=24 python scripts/train.py
