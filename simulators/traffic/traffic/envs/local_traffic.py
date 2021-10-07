from flow.core.params import NetParams
from flow.networks.traffic_light_grid import TrafficLightGridNetwork
from flow.envs import TrafficLightGridBitmapEnv
from flow.core.params import TrafficLightParams
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.controllers import SimCarFollowingController, GridRouter
import numpy as np
from gym import spaces

V_ENTER = 10
INNER_LENGTH = 100
LONG_LENGTH = 100
SHORT_LENGTH = 100
N_ROWS = 1
N_COLUMNS = 1
NUM_CARS_LEFT = 0
NUM_CARS_RIGHT = 0
NUM_CARS_TOP = 0
NUM_CARS_BOT = 0
tot_cars = (NUM_CARS_LEFT + NUM_CARS_RIGHT) * N_COLUMNS \
           + (NUM_CARS_BOT + NUM_CARS_TOP) * N_ROWS
grid_array = {
    "short_length": SHORT_LENGTH,
    "inner_length": INNER_LENGTH,
    "long_length": LONG_LENGTH,
    "row_num": N_ROWS,
    "col_num": N_COLUMNS,
    "cars_left": NUM_CARS_LEFT,
    "cars_right": NUM_CARS_RIGHT,
    "cars_top": NUM_CARS_TOP,
    "cars_bot": NUM_CARS_BOT
}
speed_limit = 10
horizontal_lanes = 1
vertical_lanes = 1
traffic_lights = True
additional_env_params = {'target_velocity': 50,
                         'switch_time': 3.0,
                         'num_observed': 2,
                         'discrete': True,
                         'tl_type': 'actuated',
                         'tl_controlled': ['center0'],
                         'scale': 10}
horizon = 300

class LocalTraffic(TrafficLightGridBitmapEnv):
    ACTION_SIZE = 2
    OBS_SIZE = 40
    """
    """
    def __init__(self, influence, seed):
        additional_net_params = {'grid_array': grid_array,
                                 'speed_limit': speed_limit,
                                 'horizontal_lanes': horizontal_lanes, 
                                 'vertical_lanes': vertical_lanes,
                                 'traffic_lights': True}
        net_params = NetParams(additional_params=additional_net_params)
        vehicles = VehicleParams()
        vehicles.add(veh_id='idm',
                     acceleration_controller=(SimCarFollowingController, {}),
                     car_following_params=SumoCarFollowingParams(
                        min_gap=2.5,
                        decel=7.5,  # avoid collisions at emergency stops
                        max_speed=V_ENTER,
                        speed_mode="all_checks",),
                    routing_controller=(GridRouter, {}),
                    num_vehicles=tot_cars)
        # initial_config, net_params = get_inflow_params(col_num=N_COLUMNS,
        #                                                row_num=N_ROWS,
        #                                                additional_net_params=additional_net_params)
        initial_config = InitialConfig(spacing='custom', lanes_distribution=float('inf'), shuffle=True)                                                       
        network = TrafficLightGridNetwork(name='grid', vehicles=vehicles, net_params=net_params, initial_config=initial_config)
        
        env_params = EnvParams(horizon=horizon, additional_params=additional_env_params)
        sim_params = SumoParams(render=False, restart_instance=False, sim_step=1, print_warnings=False, seed=seed)
        super().__init__(env_params, sim_params, network, simulator='traci')
        self.influence = influence

    # override
    def reset(self):
        self.influence.reset()
        # probs = self.influence.predict(np.zeros(40))
        node = self.tl_controlled[0]
        node_edges = dict(self.network.node_mapping)[node]
        self.veh_id = 0
        # remove pending vehicles that couldn't be added in the previous episode
        self.k.vehicle.kernel_api.simulation.clearPending()
        # add new vehicles
        # for i, edge in enumerate(node_edges):
        #     sample = np.random.uniform(0,1)
        #     if sample < probs[i]:
        #         # try:
        #         speed = 9.5
        #         self.k.vehicle.add(veh_id='idm_' + str(self.veh_id), type_id='idm', 
        #                            edge=edge, lane='free', pos=6, speed=9.5)
        #         # except:          
        #             # self.k.vehicle.remove('idm_' + str(self.veh_id))
        #             # self.k.vehicle.add(veh_id='idm_' + str(self.veh_id), type_id='idm', 
        #                             #    edge=edge, lane='free', pos=6, speed=10)
        #         self.veh_id += 1
        state = super().reset()
        observation = []
        infs = []
        for edge in range(len(node_edges)):
            observation.append(state[edge][:-1])
        observation.append(state[-1]) #  append traffic light info
        observation = np.concatenate(observation)
        self.dset = observation
        if self.influence.aug_obs:
            observation = np.append(observation, self.influence.get_hidden_state())
        reward = 0
        done = False
        return observation

    # override
    def step(self, rl_actions):
        probs = self.influence.predict(self.dset)
        node = self.tl_controlled[0]
        node_edges = dict(self.network.node_mapping)[node]
        self.k.vehicle.kernel_api.simulation.clearPending()
        for i, edge in enumerate(node_edges):
            sample = np.random.uniform(0,1)
            if sample < probs[i]:
                speed = 9.5
                if len(self.k.vehicle.get_ids_by_edge(edge)) > 8:
                    speed = 3
                # while len(self.k.vehicle.kernel_api.vehicle.getIDList()) == total_vehicles or speed == 0:
                    # self.k.vehicle.kernel_api.simulation.clearPending()
                    # self.k.vehicle.add(veh_id='idm_' + str(self.veh_id), type_id='idm', 
                                #    edge=edge, lane='allowed', pos=6, speed=speed)
                    # print(len)
                    # speed -= 1
                self.k.vehicle.add(veh_id='idm_' + str(self.veh_id), type_id='idm',
                                   edge=edge, lane='allowed', pos=6, speed=speed)
                self.veh_id += 1
        state, reward, done, _ = super().step(rl_actions)
        # self.k.vehicle.kernel_api.simulation.clearPending()
        # remove pending vehicles that couldn't be added
        node_edges = dict(self.network.node_mapping)[node]
        observation = []
        infs = []
        for edge in range(len(node_edges)):
            observation.append(state[edge][:-1])
            infs.append(state[edge][-1]) # last bit is influence source
        observation.append(state[-1]) #  append traffic light info again
        observation = np.concatenate(observation)
        infs = np.array(infs)
        self.dset = observation
        if self.influence.aug_obs:
            observation = np.append(observation, self.influence.get_hidden_state())
        if done:
            self.k.vehicle.kernel_api.simulation.clearPending()
        return observation, reward, done, {'dset': self.dset, 'infs': infs}
    
    # override
    @property
    def observation_space(self):
        pass

    def load_influence_model(self):
        print('loaded')
        self.influence._load_model()

    def close(self):
        print('terminated')
        self.terminate()
    
    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(self.OBS_SIZE,))
    
    @property
    def action_space(self):
        """
        Returns A gym dict containing the number of action choices for all the
        agents in the environment
        """
        return spaces.Discrete(self.ACTION_SIZE)