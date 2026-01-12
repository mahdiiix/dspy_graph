from typing import Optional, Protocol, Tuple

import dspy
import optuna

from dspy_graph.algo_param import AlgoParam
from dspy_graph.graph import CompiledDspy


class Optimizer(Protocol):
    def compile(self) -> Tuple[dspy.Module, optuna.Study]: ...


if __name__ == "__main__":
    # Example usage
    class SomeOptimizer:
        def __init__(self, graph: CompiledDspy):
            self.study = optuna.create_study()
            self.metric = lambda example, prediction: None
            self.optim = dspy.MIPROv2(metric=self.metric)
            self.trainset = [dspy.Example(), dspy.Example()]
            self.graph = graph
            self.evaluate = dspy.Evaluate(devset=self.trainset)
            self.best_optimized_graph: CompiledDspy = graph

        def objective(self, trial: optuna.trial.Trial):
            AlgoParam.parameter("a").value = trial.suggest_int("a", 1, 10)
            optimized_graph = self.optim.compile(self.graph, trainset=self.trainset)
            result = self.evaluate(optimized_graph)

            # Store best graph if this is the best trial so far
            if trial.number == 0 or result.score > self.study.best_value:
                self.best_optimized_graph = optimized_graph

            return result.score

        def compile(self) -> Tuple[CompiledDspy, optuna.Study]:
            """Returns (best_optimized_graph, study)"""
            self.study.optimize(self.objective, n_trials=10)
            return self.best_optimized_graph, self.study
