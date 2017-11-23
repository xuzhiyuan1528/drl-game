import numpy as np


class RandomAgent():
    """
        This is our random agent. It picks actions at random!
    """

    def __init__(self, actions):
        print('actions: ', actions)
        self.actions = actions

    def pickAction(self, reward, obs):
        return self.actions[np.random.randint(0, len(self.actions))]
