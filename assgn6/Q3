import rl.markov_process as mp
from rl.function_approx import FunctionApprox, Tabular
from rl.markov_decision_process import TransitionStep
from typing import Callable, Iterable, Iterator, TypeVar, Set, Sequence, Tuple

S = TypeVar('S')

def mc_tabular_prediction(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    approx_0: Tabular[FunctionApprox[S]],
    γ: float,
    episode_length_tolerance: float = 1e-6
) -> Iterator[Tabular[FunctionApprox[S]]]:
    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = \
        (returns(trace, γ, episode_length_tolerance) for trace in traces)
    V = approx_0
    yield V

    for episode in episodes:
        xy_seq = [(V.values_map[step.state], step.return_) for step in episode]
        x_seq = [V.values_map[step.state] for step in episode]
        y_seq = [step.return_ for step in episode]
        def deriv_func(x_seq, y_seq):
            return V.count_to_weight_func * (y_seq - V.evaluate(x_seq))
        obj_grad = V.objective_gradient(xy_seq, deriv_func)
        V = V.update_with_gradient(obj_grad)
        yield V

def td_tabular_prediction(
        transitions: Iterable[mp.TransitionStep[S]],
        approx_0: Tabular[FunctionApprox[S]],
        γ: float
) -> Iterator[Tabular[FunctionApprox[S]]]:
    def step(
            V: Tabular[FunctionApprox[S]],
            transition: mp.TransitionStep[S]
    ) -> Tabular[FunctionApprox[S]]:
        f = V.count_to_weight_func
        ns = transition.next_state
        xy_seq = [(transition.state, transition.reward)]
        def deriv_func(x_seq, y_seq):
            a = V.evaluate(x_seq)
            b = V.evaluate([ns])
            return V.count_to_weight_func * (y_seq + γ * b - a)
        obj_grad = V.objective_gradient(xy_seq, deriv_func)
        V = V.update_with_gradient(obj_grad)
        return V
    return iterate.accumulate(transitions, step, initial=approx_0)
