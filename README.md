# scalable-simulations

Source code for influence augmented local simulation for fast policy search in structured domains. The code includes:

* A policy search algorithm to train RL agents (PPO). 
* Global and local simulators of the Warehouse environment. 
* An influence predictor for the local simulators.

To run a new experiment do:

```console
cd runners
python experimentor.py --config=./configs/Warehouse/agent.yaml
```
You can choose to use the global or partial simulator, by setting the variable `simulator` in the [config file](runners/configs/Warehouse/agent.yaml) to `global` or `partial`

