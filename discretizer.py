"""
Define discrete action spaces for Gym Retro environments with a limited set of button combos
Example taken from: https://github.com/openai/retro/blob/master/retro/examples/discretizer.py
"""

import gym
import numpy as np

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class SFDiscretizer(Discretizer):
    """
    Use Street Fighter-specific discrete actions based on Ryu's moveset
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        dpad = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN']]
        dpad_combos = [['LEFT', 'UP'], ['LEFT', 'DOWN'], ['RIGHT', 'UP'], ['RIGHT', 'DOWN']]
        buttons = [['A'],['B'],['X'],['Y'],['L'],['R']]
        button_combos = [['LEFT', 'X'], ['LEFT', 'Y'], ['LEFT', 'L'],  # Forward + any Punch
                         ['RIGHT', 'X'], ['RIGHT', 'Y'], ['RIGHT', 'L'],  # Forward + any Punch
                         ['LEFT', 'DOWN', 'X'], ['LEFT', 'DOWN', 'Y'], ['LEFT', 'DOWN', 'L'],  # Down Forward + any Punch
                         ['RIGHT', 'DOWN', 'X'], ['RIGHT', 'DOWN', 'Y'], ['RIGHT', 'DOWN', 'L'],  # Down Forward + any Punch
                         ['LEFT', 'A'], ['LEFT', 'B'], ['LEFT', 'R'],  # Forward + any Kick
                         ['RIGHT', 'A'], ['RIGHT', 'B'], ['RIGHT', 'R']  # Forward + any Kick
                         ]

        combos = dpad + dpad_combos + buttons + button_combos
        print(f"Number of possible actions: {len(combos)}")
        super().__init__(env=env, combos=combos)
