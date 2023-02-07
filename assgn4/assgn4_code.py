from typing import Iterator, Tuple, TypeVar, Sequence, List, Mapping
from operator import itemgetter
import numpy as np

from rl.approximate_dynamic_programming import ValueFunctionApprox, evaluate_mrp
from rl.distribution import Distribution, Choose
from rl.dynamic_programming import evaluate_mrp_result
from rl.function_approx import FunctionApprox
from rl.iterate import iterate
from rl.markov_process import (FiniteMarkovRewardProcess, MarkovRewardProcess,
                               RewardTransition, NonTerminal, State)
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        MarkovDecisionProcess,
                                        StateActionMapping)
from rl.policy import (DeterministicPolicy, FinitePolicy, FiniteDeterministicPolicy, Policy, RandomPolicy)
from rl.monte_carlo import greedy_policy_from_qvf

S = TypeVar('S')
A = TypeVar('A')
ValueFunctionApprox = FunctionApprox[NonTerminal[S]]
QValueFunctionApprox = FunctionApprox[Tuple[NonTerminal[S], A]]
NTStateDistribution = Distribution[NonTerminal[S]]

def approx_policy_iteration(
    mdp: MarkovDecisionProcess[S, A],
    gamma: float

) -> Iterator[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]]:
    '''Calculate the value function (V*) of the given MDP by improving
    the policy repeatedly after evaluating the value function for a policy
    '''

    def update(vfapprox_policy: Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]) -> Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]:
        vfa, pi = vfapprox_policy
        mrp = MarkovRewardProcess[S] = mdp.apply_policy(pi)
        policy_vf: ValueFunctionApprox[S] = evaluate_mrp(mrp, gamma, vfa, NonTerminal[S])
        #qfva is qvalue function approx, don't know how to create this

        improved_pi: DeterministicPolicy[S, A] = greedy_policy_from_qvf(qfva, NonTerminal[S])


        return policy_vf, improved_pi

    v_0: ValueFunctionApprox[S] = FunctionApprox[NonTerminal[S]]
#create policy
    pi_0: DeterministicPolicy[S, A] = DeterministicPolicy(
        {s.state: Choose(mdp.actions(s)) for s in mdp.non_terminal_states}
    )
    return iterate(update, (v_0, pi_0))