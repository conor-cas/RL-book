from dataclasses import dataclass
from typing import Mapping, Dict
from rl.distribution import Categorical, FiniteDistribution
from rl.markov_process import FiniteMarkovProcess
from scipy.stats import poisson


@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order

class Snakes_and_Ladders:
    position: int

class SimpleInventoryMPFinite(FiniteMarkovProcess[InventoryState]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> \
            Mapping[InventoryState, FiniteDistribution[InventoryState]]:
        d: Dict[InventoryState, Categorical[InventoryState]] = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state = InventoryState(alpha, beta)
                ip = state.inventory_position()
                beta1 = self.capacity - ip
                state_probs_map: Mapping[InventoryState, float] = {
                    InventoryState(ip - i, beta1):
                    (self.poisson_distr.pmf(i) if i < ip else
                     1 - self.poisson_distr.cdf(ip - 1))
                    for i in range(ip + 1)
                }
                d[InventoryState(alpha, beta)] = Categorical(state_probs_map)
        d: Dict[Snakes_and_Ladders, FiniteDistribution[Snakes_and_Ladders]] = {}

        ladders_snakes_dict = {1: 38, 4: 14, 8: 30, 21: 42,

                               28: 76, 50: 67, 71: 92, 80: 99,

                               97: 78, 95: 56, 88: 24, 62: 18, 48: 26,

                               36: 6, 32: 10}

        for index in range(100):
            state = Snakes_and_Ladders(index)
            inner_map: Mapping[state, float] = {
                Snakes_and_Ladders(ladders_snakes_dict[index + roll] if index + roll in ladders_snakes_dict else
                                   index + roll): (1/6) for roll in range(1, 7)

            }
            d[Snakes_and_Ladders(index)] = Categorical(inner_map)

        return d


if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0

    si_mp = SimpleInventoryMPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda
    )

    print("Transition Map")
    print("--------------")
    print(si_mp)

    print("Stationary Distribution")
    print("-----------------------")
    si_mp.display_stationary_distribution()
