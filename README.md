# IALS

Source code for the paper [Influence-Augmented Local Simulators: a Scalable Solution for Fast Deep RL in Large Networked Systems](https://proceedings.mlr.press/v162/suau22a.html):

## Requirements
[Singularity](https://sylabs.io/docs/)

## Installation
```console 
sudo singularity build IALS.sif IALS.def
```

## Running an experiment
Launch the singularity shell:
```console
singularity shell --writable-tmpfs IALS.sif
```
To run a new experiment do:
```console
cd runners
python experiment.py with ./configs/warehouse/local_fnn_framestack.yaml
```
This will train a new policy in the warehouse env on IALS. To train on the global simulator change the config file path to `./configs/warehouse/global_fnn_framestack.yaml`.
