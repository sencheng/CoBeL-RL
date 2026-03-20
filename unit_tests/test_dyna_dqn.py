"""
Performs test simulations of the Dyna-DQN agent with different settings.
"""
# basic imports
from itertools import product
# PyTorch
from torch import set_num_threads, Tensor
from torch.nn import Module, Linear
from torch.nn.functional import relu
# CoBeL-RL
from cobel.agent import DynaDQN
from cobel.policy import EpsilonGreedy
from cobel.network import TorchNetwork
from cobel.interface import Gridworld
from cobel.misc.gridworld_tools import make_open_field


class Model(Module):

    def __init__(self) -> None:
        super().__init__()
        self.hidden = Linear(25, 32)
        self.output = Linear(32, 4)
        self.double()

    def forward(self, batch: Tensor) -> Tensor:
        x = relu(self.hidden(batch))
        x = self.output(x)

        return x


def simulation(
    use_test_policy: bool = True,
    mask_actions: bool = True,
    ddqn: bool = False,
    target_update: float = 0.001
) -> None:
    """
    Represents one test simulation.

    Parameters
    ----------
    use_test_policy : bool, default=True
        Indicates whether a separate test policy should be used.
    mask_actions : bool, default=True
        Indicates whether an action mask should be used.
    ddqn : bool, default=True
        Indicates whether the DDQN mode should be used.
    target_update : float, default=0.001
        The update rate for the DQN's target network.
        If less than one, the target network is constantly
        being blended with the online network.
    """
    set_num_threads(1)
    env = Gridworld(make_open_field(5, 5, 0, 1))
    policy = EpsilonGreedy()
    policy_test: None | EpsilonGreedy = None
    if use_test_policy:
        policy_test = EpsilonGreedy(0.)
    model = TorchNetwork(Model())
    agent = DynaDQN(
        env.observation_space,
        env.action_space,
        policy,
        model,
        policy_test=policy_test,
    )
    agent.target_update = target_update
    agent.mask_actions = mask_actions
    agent.DDQN = ddqn
    agent.train(env, 5, 20)
    agent.test(env, 5, 20)


def main() -> None:
    """
    The main function.
    """
    policy = [True, False]
    mask_actions = [True, False]
    ddqn = [True, False]
    target_update = [0.001, 10]
    for combination in product(policy, mask_actions, ddqn, target_update):
        simulation(*combination)


if __name__ == "__main__":
    main()
