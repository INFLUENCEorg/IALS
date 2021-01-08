# scalable-simulations

Source code for influence augmented local simulation for fast policy search in structured domains. The code includes:

* A policy search algorithm to train RL agents (PPO). 
* Global and local simulators of the Warehouse environment. 
* An influence predictor for the local simulator.

## Installation

### SUMO

Install the dependencies:
```
sudo apt-get install cmake python g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig
```
Clone the sumo repository:
```
git clone --recursive https://github.com/eclipse/sumo
```
Set the environment variables:
```
echo 'export SUMO_HOME="/path/to/sumo"' >> ~/.bashrc
echo 'export PYTHONPATH="/path/to/sumo/tools/"' >> ~/.bashrc 
```
Build the sumo binaries:
```
mkdir sumo/build/cmake-build && cd sumo/build/cmake-build
cmake ../..
make -j$(nproc)
```

## Running an experiment
To run a new experiment do:

```console
cd runners
python experimentor.py with ./configs/Warehouse/partial.yaml
```
This will train a new policy on the partial simulator. To train on the global simulator change the config file path to `./configs/Warehouse/global.yaml`.
