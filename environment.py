import retro
from discretizer import SFDiscretizer

def make_env(game, state, rank, logdir, seed=0):
    """
    Utility function for multiprocessed env.
    see: https://stable-baselines.readthedocs.io/en/master/guide/examples.html

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = retro.make(game=game, state=state, record=logdir)
        env = SFDiscretizer(env)
        env.seed(seed + rank)
        return env
    return _init
