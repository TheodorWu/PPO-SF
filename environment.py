import retro
from discretizer import SFDiscretizer

def make_env(game, state, rank, logdir, seed=0):
    """
    Utility function for multiprocessed env.
    see: https://stable-baselines.readthedocs.io/en/master/guide/examples.html
    """
    def _init():
        if logdir:
            # passing the logdir allows to record the actions of the agent for later visualization
            env = retro.make(game=game, state=state, record=logdir)
        else:
            env = retro.make(game=game, state=state)
        env = SFDiscretizer(env)
        env.seed(seed + rank)
        return env
    return _init
