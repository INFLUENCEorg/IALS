# IALS

Source code for the paper [Influence-Augmented Local Simulators: a Scalable Solution for Fast Deep RL in Large Networked Systems](https://proceedings.mlr.press/v162/suau22a.html) by Miguel Suau, Jinke He, Mustafa Mert Çelikok, Matthijs Spaan, and Frans Oliehoek.

## Requirements
[Singularity](https://sylabs.io/docs/)

## Installation
```console 
sudo singularity build IALS.sif IALS.def
```
This will create a singularity container and install all the required packages. Alternatively, you can create a virtual environment and install the packages listed in IALS.def

## Running an experiment
Launch the singularity shell:
```console
singularity shell --writable-tmpfs IALS.sif
```
To run a new experiment do:
```console
cd runners
python experiment.py with ./configs/warehouse/IALS.yaml
```
This will train a new policy for the warehouse environment on IALS. To train on the global simulator change the config file path to `./configs/warehouse/global.yaml`.
