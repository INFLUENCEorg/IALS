from warehouse.envs.mini_warehouse import MiniWarehouse
from warehouse.envs.utils import *
import numpy as np
sys.path.append("..")

class LocalMiniWarehouse(MiniWarehouse):
    """
    warehouse environment
    """

    ACTIONS = {0: 'UP',
               1: 'DOWN',
               2: 'LEFT',
               3: 'RIGHT'}

    def __init__(self, influence, seed):
        self.influence = influence
        self.total_steps = 0
        super(LocalMiniWarehouse, self).__init__(seed)

    def reset(self):
        """
        Resets the environment's state
        """
        self.robot_id = 0
        self._place_robots()
        self.item_id = 0
        self.items = []
        self.just_removed_list = []
        self._add_items()
        obs = self._get_observation()
        self.episode_length = 0
        self.influence.reset()
        if self.influence.aug_obs:
            obs = np.append(obs, self.influence.get_hidden_state())
        return obs

    def step(self, action):
        """
        Performs a single step in the environment.
        """
        self.probs = self.influence.predict(self.get_dset)
        self._increase_item_waiting_time()
        self._robots_act([action])
        reward = self._compute_reward(self.robots[self.learning_robot_id])
        self._remove_items(self.probs)
        self._add_items()
        obs = self._get_observation()
        self.episode_length += 1
        self.total_steps += 1
        done = (self.max_episode_length <= self.episode_length)
        return obs, reward, done, {}
        
    ######################### Private Functions ###########################

    def _remove_items(self, probs):
        """
        Removes items collected by robots. Robots collect items by steping on
        them
        """
        self.just_removed_list = []
        for robot in self.robots:
            robot_pos = robot.get_position
            for item in np.copy(self.items):
                item_pos = item.get_position
                if robot_pos[0] == item_pos[0] and robot_pos[1] == item_pos[1]:
                    self.items.remove(item)
                    self.just_removed_list.append(item.get_position)
        
        for prob, item_loc in zip(self.probs, self.get_item_locs()):
            remove_item = np.random.choice([False, True], p=prob)
            if remove_item:
                for item in np.copy(self.items):
                    if item.get_position[0] == item_loc[0] and item.get_position[1] == item_loc[1]:
                        self.items.remove(item)
                        self.just_removed_list.append(item.get_position)
                        # SAVE ITEM WAITING TIME
                        # from pathlib import Path
                        # p = Path('../plots/data_NM.csv')
                        # with p.open('ab') as f:
                        #     # np.save(f, item.get_waiting_time)
                        #     np.savetxt(f, [item.get_waiting_time], delimiter=",")

                        


    def get_item_locs(self):
        bitmap_locs = np.zeros((self.n_columns, self.n_rows))
        bitmap_locs[[0,-1], 1:-1] = 1
        bitmap_locs[1:-1, [0,-1]] = 1
        item_locs = np.argwhere(bitmap_locs == 1)
        return item_locs
    

