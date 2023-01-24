import sys
sys.path.insert(0, "Desktop/RL-book")
#from ast import Constant
from dataclasses import dataclass
from typing import Mapping, Dict, Tuple
from rl.distribution import Constant, Categorical, FiniteDistribution
from rl.markov_process import FiniteMarkovProcess, NonTerminal, FiniteMarkovRewardProcess
from scipy.stats import poisson
from itertools import islice
from rl.gen_utils.plot_funcs import plot_list_of_curves
import matplotlib.pyplot as plt
import numpy as np

@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


# state
@dataclass(frozen=True)
class SnakesAndLaddersState:
    position_number: int


class SimpleInventoryMPFinite(FiniteMarkovProcess[SnakesAndLaddersState]):

    def __init__(
            self,
            capacity: int,
            poisson_lambda: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_transition_map())

    # Modify this. defines movements of the rules
    def get_transition_map(self) -> \
            Mapping[InventoryState, FiniteDistribution[InventoryState]]:
        # Our own code
        d: Dict[SnakesAndLaddersState, Categorical[
            SnakesAndLaddersState]] = {}  # The type of this when defining? What is categorical meaning?

        ladders_snakes_dict = {1: 38, 4: 14, 8: 30, 21: 42, 28: 76,
                               50: 67, 71: 92, 80: 99,
                               97: 78, 95: 56, 88: 24, 62: 18, 48: 26,
                               36: 6, 32: 10}

        for position in range(100):
            state = SnakesAndLaddersState(position)

            # Inner map of next state and probability from this state
            state_probs_map: Mapping[state, float] = {
                SnakesAndLaddersState(ladders_snakes_dict[
                                          position + step_addition] if position + step_addition in ladders_snakes_dict else min(
                    position + step_addition, 100)):
                    (1 / 6) for step_addition in range(1, 7)
            }

            d[SnakesAndLaddersState(position)] = Categorical(state_probs_map)

        return d


class RewardFMP(FiniteMarkovRewardProcess[SnakesAndLaddersState]):

    def __init__(
            self,
            capacity: int,
            poisson_lambda: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_transition_reward_map())

    # Modify this. defines movements of the rules
    def get_transition_reward_map(self) -> \
            Mapping[InventoryState, FiniteDistribution[Tuple[InventoryState, float]]]:
        # Our own code
        d: Dict[SnakesAndLaddersState, Categorical[
            Tuple[SnakesAndLaddersState, float]]] = {}  # The type of this when defining? What is categorical meaning?

        ladders_snakes_dict = {1: 38, 4: 14, 8: 30, 21: 42, 28: 76,
                               50: 67, 71: 92, 80: 99,
                               97: 78, 95: 56, 88: 24, 62: 18, 48: 26,
                               36: 6, 32: 10}

        for position in range(100):
            state = SnakesAndLaddersState(position)

            # Inner map of next state and probability from this state
            state_probs_map: Mapping[state, float] = {
                (SnakesAndLaddersState(ladders_snakes_dict[
                                           position + step_addition] if position + step_addition in ladders_snakes_dict else min(
                    position + step_addition, 100)), -1):
                    (1 / 6) for step_addition in range(1, 7)
            }

            d[SnakesAndLaddersState(position)] = Categorical(state_probs_map)

        return d


if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0

    # Defined instance of class that holds transition info (see above)
    si_mp = SimpleInventoryMPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda
    )

    start_state_distribution = Constant(NonTerminal(SnakesAndLaddersState(0)))
    traces = si_mp.traces(start_state_distribution)
    for t in range(5):
        trace = next(traces)
        positions = [s.state.position_number for s in trace]
        print(positions)
        print(len(positions))
        # make data
        ypoints = np.array(positions)
        xpoints = np.array(range(len(positions)))

        plt.plot(xpoints, ypoints)

        plt.savefig("traces.png")
    lengths = []
    for t in range(1000):
        trace = next(traces)
        positions = [s.state.position_number for s in trace]
        lengths.append(len(positions))
    fig2 = plt.figure("Histogram")
    plt.hist(lengths, bins = 20)
    plt.savefig("hist.png")


    si_rmp = RewardFMP(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda
    )

    start_state_rewards_distribution = Constant(NonTerminal(SnakesAndLaddersState(0)))
    expected_val = si_rmp.get_value_function_vec(1)

    # prints out expected value of each step
    print(expected_val)
    print(expected_val[0])

    print("Transition Map")
    print("--------------")


    # print(si_mp)

    # print("Stationary Distribution")
    # print("-----------------------")
    # si_mp.display_stationary_distribution()


