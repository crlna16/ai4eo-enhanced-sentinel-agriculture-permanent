

<div align="center">

# AI4EO-Slovenia
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/HelmholtzAI-Consultants-Munich/ML-Pipeline-Template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
<a href="https://github.com/pyscaffold/pyscaffoldext-dsproject"><img alt="Template" src="https://img.shields.io/badge/-Pyscaffold--Datascience-017F2F?style=flat&logo=github&labelColor=gray"></a>

</div>

# Description
This project is working on superresolution for Earth observation data using the permanently open AI4EO challenge for Enhanced Sentinel 2 Agriculture in Slovenia.


# Install on HAICORE
Before using the template, one needs to install the project as a package.
* First activate the necessary modules:
    ```
    source modules.sh
    ```
* Then create a virtual environment:
    ```
    python3 -m venv --system-site-packages .ai4eo_superresolution_env
    ```
* Then, activate the environment
    ```
    source .ai4eo_superresolution_env/bin/activate
    ```
* Finally, install the necessary packages and then the project as a package. Run:
    ```
    pip install --upgrade pip
    pip install --upgrade setuptools
    pip install -e .
    ```
  

## Run training
To run the training (resp. testing) pipeline, simply run (from within the project directory):
```
source modules.sh
source .ai4eo_superresolution_env/bin/activate
python scripts/train.py
# or python scripts/test.py
```
Or, if you want to submit the training job to a submit (resp. interactive) cluster node via slurm, run:
```
sbatch job_submission.sbatch

```
> * The experiments, evaluations, etc., are stored under the `logs` directory.
> * When evaluating (running `test.py`), make sure you give the correct checkpoint path in `configs/test.yaml`



# Project Organization
```
├── configs                              <- Hydra configuration files
│   ├── callbacks                               <- Callbacks configs
│   ├── datamodule                              <- Datamodule configs
│   ├── debug                                   <- Debugging configs
│   ├── experiment                              <- Experiment configs
│   ├── hparams_search                          <- Hyperparameter search configs
│   ├── local                                   <- Local configs
│   ├── log_dir                                 <- Logging directory configs
│   ├── logger                                  <- Logger configs
│   ├── model                                   <- Model configs
│   ├── trainer                                 <- Trainer configs
│   │
│   ├── test.yaml                               <- Main config for testing
│   └── train.yaml                              <- Main config for training
│
├── data                                 <- Project data
│   └── MNIST
│       ├── processed                               <- Processed data
│       └── raw                                     <- Raw data
│
├── docs                                 <- Directory for Sphinx documentation in rst or md.
├── models                               <- Trained and serialized models, model predictions
├── notebooks                            <- Jupyter notebooks.
├── reports                              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                                 <- Generated plots and figures for reports.
├── scripts                              <- Scripts used in project
│   ├── job_submission.sbatch               <- Submit training job to slurm
│   ├── job_submission_interactive.sbatch   <- Submit training job to slurm (interactive node)
│   ├── test.py                             <- Run testing
│   └── train.py                            <- Run training
│
├── src/ai4eo_slovenia             <- Source code
│   ├── datamodules                             <- Lightning datamodules
│   ├── models                                  <- Lightning models
│   ├── utils                                   <- Utility scripts
│   │
│   ├── testing_pipeline.py                     <- Model evaluation workflow
│   └── training_pipeline.py                    <- Model training workflow
│
├── tests                                <- Tests of any kind
│   ├── helpers                                 <- A couple of testing utilities
│   ├── shell                                   <- Shell/command based tests
│   └── unit                                    <- Unit tests
│
├── .gitignore                           <- List of files/folders ignored by git
├── .pre-commit-config.yaml              <- Configuration of pre-commit hooks for code formatting
├── requirements.txt                     <- File for installing python dependencies
├── setup.cfg                            <- Configuration of linters and pytest
├── LICENSE.txt                          <- License as chosen on the command-line.
├── pyproject.toml                       <- Build configuration. Don't change! Use `pip install -e .`
│                                           to install for development or to build `tox -e build`.
├── setup.cfg                            <- Declarative configuration of your project.
├── setup.py                             <- [DEPRECATED] Use `python setup.py develop` to install for
│                                           development or `python setup.py bdist_wheel` to build.
└── README.md
```
