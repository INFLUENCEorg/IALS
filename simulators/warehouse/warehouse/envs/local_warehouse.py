from warehouse.envs.global_warehouse import GlobalWarehouse
from warehouse.envs.utils import *
import numpy as np
sys.path.append("..")

class LocalWarehouse(GlobalWarehouse):
    """
    warehouse environment
    """

    ACTIONS = {0: 'UP',
               1: 'DOWN',
               2: 'LEFT',
               3: 'RIGHT'}

    def __init__(self, influence, seed):
        self.parameters = read_parameters('local_warehouse.yaml')
        # parameters = parse_arguments()
        self.n_columns = self.parameters['n_columns']
        self.n_rows = self.parameters['n_rows']
        self.n_robots_row = self.parameters['n_robots_row']
        self.n_robots_column = self.parameters['n_robots_column']
        self.distance_between_shelves = self.parameters['distance_between_shelves']
        self.robot_domain_size = self.parameters['robot_domain_size']
        self.prob_item_appears = self.parameters['prob_item_appears']
        # The learning robot
        self.learning_robot_id = self.parameters['learning_robot_id']
        self.max_episode_length = self.parameters['n_steps_episode']
        self.obs_type = self.parameters['obs_type']
        self.items = []
        self.img = None
        self.influence = influence
        self.seed(seed)

    def reset(self):
        """
        Resets the environment's state
        """
        self.robot_id = 0
        self._place_robots()
        self.item_id = 0
        self.items = []
        self._add_items()
        obs = self._get_observation()
        self.episode_length = 0
        self.influence.reset()
        return obs

    def step(self, action):
        """
        Performs a single step in the environment.
        """
        self.probs = self.influence.predict(self.get_dset)
        self._increase_item_waiting_time()
        self._robots_act([action])
        # ext_robot_locs = self._sample_ext_robot_locs(self.probs)
        reward = self._compute_reward()
        self._remove_items(self.probs)
        self._add_items()
        obs = self._get_observation()
        self.episode_length += 1
        done = (self.max_episode_length <= self.episode_length)
        # if self.parameters['render']:
        #     self.render(self.parameters['render_delay'])
        # Influence-augmented observations
        if self.influence.aug_obs:
            obs = np.append(obs, self.influence.get_hidden_state())
        return obs, reward, done, {}
        
    ######################### Private Functions ###########################

    def _remove_items(self, probs):
        """
        Removes items collected by robots. Robots collect items by steping on
        them
        """
        for robot in self.robots:
            robot_pos = robot.get_position
            for item in np.copy(self.items):
                item_pos = item.get_position
                if robot_pos[0] == item_pos[0] and robot_pos[1] == item_pos[1]:
                    self.items.remove(item)
        ext_robot_locs = self._sample_ext_robot_locs(probs)
        for ext_robot_loc in ext_robot_locs:
            if ext_robot_loc is not None:
                for item in np.copy(self.items):
                    item_pos = item.get_position
                    if ext_robot_loc[0] == item_pos[0] and ext_robot_loc[1] == item_pos[1]:
                        self.items.remove(item)

    def item_pos2coor(self, pos):
        bitmap = np.zeros((self.parameters['n_rows'], self.parameters['n_columns']))
        bitmap[pos[0], pos[1]] = 1
        vec = np.concatenate((bitmap[[0,-1], 1:-1].flatten(),
                              bitmap[1:-1, [0,-1]].flatten()))
        coor = np.where(vec == 1)[0][0]
        return coor

    def _sample_ext_robot_locs(self, probs):
        locations = []
        for neighbor_id, prob in enumerate(probs):
            loc = np.random.choice(np.arange(len(prob)), p=prob)
            # bitmap = np.zeros(len(prob))
            # bitmap[sample] = 1
            # bitmap = np.reshape(bitmap, (self.robot_domain_size[0], self.robot_domain_size[1]))
            # intersection = np.array(self._get_intersection(neighbor_id, bitmap))
            # if all(intersection == np.zeros(len(intersection))):
            #     location = None
            # else:
            if loc < len(prob)-1:
                location = self._find_loc(neighbor_id, loc+1)
            else:
                location = None
            locations.append(location)
        return locations
    
    def _find_loc(self, neighbor_id, loc):
        locations = {0: [loc, self.robot_domain_size[1]-1], 1: [self.robot_domain_size[0]-1, loc], 2: [loc, 0], 3: [0, loc]}
        return locations[neighbor_id]

    def _get_intersection(self, neighbor_id, bitmap):
        intersections = {0: bitmap[:, 0], 1: bitmap[0, :], 2: bitmap[:, self.robot_domain_size[1]-1], 3: bitmap[self.robot_domain_size[0]-1, :]}
        return intersections[neighbor_id]

    def load_influence_model(self):
        self.influence._load_model()

    # def _robots_act(self, action):
    #     """
    #     Learning robot takes an action in the environment.
    #     """
    #     self.robots[self.learning_robot_id].act(action)

