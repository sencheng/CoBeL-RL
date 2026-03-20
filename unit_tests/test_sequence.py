"""
Basic unit test for the Sequence interface class.
"""
# basic imports
import numpy as np
from gymnasium.spaces import Discrete, Box, Tuple, Dict
# CoBeL-RL
from cobel.interface import Sequence
# typing
from typing import Literal
from cobel.typing import Trial, Observation


def prepare_trials() -> list[Trial]:
    """
    Prepares the trial sequence.

    Returns
    -------
    trials : list of cobel.interface.sequence.Trial
        The prepared trial sequence.
    """
    trials: list[Trial] = []
    for _ in range(16):
        trials.append(
                [{"observation": "A", "reward": np.array([1., 0.]), "action": 0}]
        )
        trials.append(
                [{"observation": "B", "reward": np.array([0., 1.]), "action": 0}]
        )

    return trials


def prepare_observations(
    obs_type: Literal["Simple", "List", "Dict"]
) -> tuple[dict[str, Observation], Box | Tuple | Dict]:
    """
    Prepares observations.

    Parameters
    ----------
    obs_type : "Simple", "List" or "Dict"
        The type of observation that will be prepared.

    Returns
    -------
    obs_dict : dict of Observation
        A dictionary containing the prepared observations.
    obs_space : gymnasium.spaces.Box, gymnasium.spaces.Tuple or gymnasium.spaces.Dict
        The observation space.
    """
    obs_dict: dict[str, Observation]
    obs_space: Box | Tuple | Dict
    if obs_type == "Simple":
        obs_dict = {
            "A": np.array([1., 0.]),
            "B": np.array([0., 1.])
        }
        obs_space = Box(0., 1., (2, ))
    elif obs_type == "List":
        obs_dict = {
            "A": [np.array([1., 0.]), np.array([1., 0.])],
            "B": [np.array([0., 1.]), np.array([0., 1.])]
        }
        obs_space = Tuple(
            [Box(0., 1., (2, )), Box(0., 1., (2, ))]
        )
    else:
        obs_dict = {
            "A": {"1": np.array([1., 0.]), "2": np.array([1., 0.])},
            "B": {"1": np.array([0., 1.]), "2": np.array([0., 1.])}
        }
        obs_space = Dict(
            {"1": Box(0., 1., (2, )), "2": Box(0., 1., (2, ))}
        )

    return obs_dict, obs_space


def main() -> None:
    """
    The main function.
    """
    trials = prepare_trials()
    obs, obs_space = prepare_observations("Simple")
    env = Sequence(trials, obs, obs_space, 2)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)
    assert env.observation_space.shape == (2, )
    assert env.action_space.n == 2
    states_true = ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
    rewards_true = [1., 0., 1., 0., 1., 0., 1., 0., 1., 0.]
    for i in range(10):
        env.reset()
        s = env.trials[env.current_trial][0]["observation"]
        _, r, t, _, _ = env.step(0)
        assert s == states_true[i]
        assert r == rewards_true[i]
        assert t
    # list observations
    obs, obs_space = prepare_observations("List")
    env = Sequence(trials, obs, obs_space, 2)
    assert isinstance(env.observation_space, Tuple)
    assert isinstance(env.observation_space.spaces[0], Box)
    assert isinstance(env.observation_space.spaces[1], Box)
    assert isinstance(env.action_space, Discrete)
    assert env.observation_space.spaces[0].shape == (2, )
    assert env.observation_space.spaces[1].shape == (2, )
    assert env.action_space.n == 2
    # dict observations
    obs, obs_space = prepare_observations("Dict")
    env = Sequence(trials, obs, obs_space, 2)
    assert isinstance(env.observation_space, Dict)
    assert isinstance(env.observation_space.spaces["1"], Box)
    assert isinstance(env.observation_space.spaces["2"], Box)
    assert isinstance(env.action_space, Discrete)
    assert env.observation_space.spaces["1"].shape == (2, )
    assert env.observation_space.spaces["2"].shape == (2, )
    assert env.action_space.n == 2
    # test action overwrite
    obs, obs_space = prepare_observations("Simple")
    env = Sequence(trials, obs, obs_space, 2, True)
    for _ in range(10):
        env.reset()
        log = env.step(1)[-1]
        assert log["step_action"] != 1
        assert log["step_action"] != log["action"]


if __name__ == "__main__":
    main()
