# Instructions for Reproducibility


The following document outlines the essential requirements and procedures necessary to reproduce the experimental part of our work.


## Hardware requirements

The training and assessment of our models have been performed on an *NVIDIA DGXA100* system, with 8 *NVIDIA A100 SXM4* GPUs (each endowed with 40 GB of dedicated memory). A similar setup will be required to match the training times we report. A greater or equal amount of dedicated memory will be required to run the scripts provided without modification to (at least) the batch size. At least 16 free CPU threads and 64 GB of free main memory, per GPU, are also assumed to be available.


## Software requirements

Our experiments have been run on *Ubuntu 22.04.3 LTS*, with *Python 3.11.9*. A `bash` shell, the `curl` command and a working Python installation (with the pip command available) are assumed to be present on the system. Access to resources is assumed to be controlled by the `slurm` scheduler (in our case: *slurm 22.05.9*).

A minimal Python environment required to reproduce the experiments can be set-up by issuing the following commands:

```bash
pip install --upgrade \
                        "safe-assert>=0.5" \
                        "torch>=2.2.2" \
                        "torchvision>=0.17" \
                        "tqdm>=4.66.2" \
                        "wandb>=0.16.5" \
                        "safetensors>=0.4.3" \
                        "git+https://github.com/fra31/auto-attack.git" \
                        "git+https://github.com/Harry24k/adversarial-attacks-pytorch.git"

pip install --upgrade   "git+https://github.com/BorealisAI/advertorch.git"
pip install --upgrade   "ebtorch>=0.24.2"
pip install --upgrade   "requests>=2.28"
```

An eventual error about `pip` not being able to solve a dependency conflict involving the `requests` package is expected, and does not compromise the correct reproducibility of the results.


## Datasets, *pre-trained* models

The datasets required for both training and evaluation (*CIFAR-10*, *CIFAR-100*, *TinyImageNet-200*) are automatically downloaded from the Internet by the respective training and evaluation Python scripts provided.

The training phase further requires access to adversarially pre-trained model weights from *[Cui et al., 2023]* and *[Wang et al., 2023]*, which can be downloaded from a dedicated online storage bucket (set up *ad hoc* for the sake of reproducibility). Download of model weights can be initiated by issuing the following command:

```bash
cd models; bash get_models_train.sh
```

that it is assumed to have been manually run before starting the training itself. The weights are provided in *Huggingface `safetensors`* format, converted from those originally offered by their original Authors (in *Pickle-based* PyTorch weights format).


To reproduce the evaluation phase directly, without the need for model training, we provide the pre-trained weights for all required (sub)models. Those can be equivalently obtained by reproducing the training phase beforehand. Such pre-trained weights can also be obtained from the same online storage bucket as:

```bash
cd models; bash get_models_test.sh
```

The script is assumed to have been manually run before starting the testing itself. The weights are provided in *Huggingface `safetensors`* format.


## Model training

To reproduce the entire training phase, assuming the availability of a *slurm* queue called `DGX`, the following commands should be sufficient:

```bash
# Scenario (a)
sbatch carsotrain_a.sh

# Scenario (b)
sbatch carsotrain_b.sh

# Scenario (c)
sbatch carsotrain_c.sh
```

Such scripts implicitly assume that the contents of current folder are placed in the `$HOME/Downloads/CARSO` directory, and that the relevant Python interpreter binary is located in `$HOME/micromamba/envs/nightorch/bin/python`. Adaptation to different setups is possible by editing the relevant environment variables set un in the respective scripts.


## Model evaluation

To reproduce the entire evaluation phase, assuming the availability of a *slurm* queue called `DGX`, the following commands should be sufficient:

```bash
# Scenario (a): randAA evaluation
sbatch carsoeval_a.sh

# Scenario (a): randAA evaluation for gradient obfuscation
sbatch carsoeval_a_hieps.sh

# Scenario (a): PGD+EoT evaluation
sbatch carsoeval_a_pgdeot.sh


# Scenario (b): randAA evaluation
sbatch carsoeval_b.sh

# Scenario (b): randAA evaluation for gradient obfuscation
sbatch carsoeval_b_hieps.sh


# Scenario (c): randAA evaluation
sbatch carsoeval_c.sh

# Scenario (c): randAA evaluation for gradient obfuscation
sbatch carsoeval_c_hieps.sh

```

Such scripts implicitly assume that the contents of current folder are placed in the `$HOME/Downloads/CARSO` directory, and that the relevant Python interpreter binary is located in `$HOME/micromamba/envs/nightorch/bin/python`. Adaptation to different setups is possible by editing the relevant environment variables set un in the respective scripts.


## Time requirements

A time of $\approx$ 14h is required for a full reproduction of the training phase assuming the availability of our same hardware and software infrastructure.
A time of $\approx$ 71h is required for a full reproduction of the evaluation phase assuming the availability of our same hardware and software infrastructure.

Individual timings for each script are provided as part of the `#SBATCH` preamble, with a 1.5$\times$ safety margin.
