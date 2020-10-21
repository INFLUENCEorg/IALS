# scalable-simulations

Source code for influence augmented local simulation for fast policy search in structured domains. The code includes:

* A policy search algorithm to train RL agents (PPO). 
* Global and local simulators of the Warehouse environment. 
* An influence predictor for the local simulator.

To run a new experiment do:

```console
cd runners
python experimentor.py with ./configs/Warehouse/partial.yaml
```
This will train a new policy on the partial simulator. To train on the global simulator change the config file path to `./configs/Warehouse/global.yaml`.
