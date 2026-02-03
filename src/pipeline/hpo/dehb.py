"""
Differential Evolution Hyperband (DEHB) wrapper.
"""

import time
from typing import Callable, Dict, Tuple, Union

import ConfigSpace
from dehb import DEHB


class DifferentialEvolutionHyperband(DEHB):
    """
    Wrapper for DEHB to be compatible with the existing interface expected by
    the WSmart-Route HPO pipeline and tests.
    """

    def __init__(
        self,
        cs: Dict[str, Union[Tuple[float, float], list]],
        f: Callable,
        min_fidelity: int = 1,
        max_fidelity: int = 10,
        eta: int = 3,
        n_workers: int = 1,
        output_path: str = "./dehb_output",
        **kwargs,
    ):
        """
        Initialize DifferentialEvolutionHyperband wrapper.

        Args:
            cs: Configuration space definition (dict of ranges).
            f: Objective function to minimize.
            min_fidelity: Minimum fidelity (e.g. epochs).
            max_fidelity: Maximum fidelity.
            eta: Halving rate.
            n_workers: Number of workers.
            output_path: Path for logs and results.
            **kwargs: Additional arguments for DEHB.
        """
        self.parameter_names = list(cs.keys()) if isinstance(cs, dict) else []

        # Convert simple dict config space to ConfigSpace object if needed
        config_space: Union[ConfigSpace.ConfigurationSpace, Dict] = cs
        if isinstance(cs, dict):
            config_space = ConfigSpace.ConfigurationSpace()
            for name, (low, high) in cs.items():
                # specific handling for validation that expects floats
                hp = ConfigSpace.UniformFloatHyperparameter(name, lower=low, upper=high)
                config_space.add_hyperparameter(hp)

        super().__init__(
            cs=config_space,
            f=f,
            min_fidelity=min_fidelity,
            max_fidelity=max_fidelity,
            eta=eta,
            n_workers=n_workers,
            output_path=output_path,
            **kwargs,
        )

    def run(self, fevals: int = 100, **kwargs):
        """
        Run DEHB optimization.

        Args:
            fevals: Number of function evaluations budget.

        Returns:
            Tuple of (best_config, runtime, history)
        """
        start_time = time.time()

        # DEHB.run returns (traj, runtime, history) arrays
        # We pass fevals to DEHB.run
        super().run(fevals=fevals, **kwargs)

        total_runtime = time.time() - start_time

        # Get best configuration found
        # incumbents returns (config, score)
        best_config, best_score = self.get_incumbents()

        # Convert ConfigSpace configuration to dict for compatibility
        if hasattr(best_config, "get_dictionary"):
            best_config = best_config.get_dictionary()

        return best_config, total_runtime, self.history
