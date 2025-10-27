# (C) 2019 Baris Ozmen <hbaristr@gmail.com>
# (C) 2024 Peter Norvig

import skopt
import numpy as np

AUG_TYPES = [
    "crop", "gaussian-blur", "rotate", "shear", "translate-x", "translate-y",
    "sharpen", "emboss", "additive-gaussian-noise", "dropout", "coarse-dropout",
    "gamma-contrast", "brighten", "invert", "fog", "clouds",
    "add-to-hue-and-saturation", "coarse-salt-pepper", "horizontal-flip",
    "vertical-flip",
]

class Controller:
    def __init__(self, config):
        self.method = config["method"]
        if self.method.startswith("bayes"):
            self.method = "bayesian_optimization"
            self._init_skopt(config["opt_initial_points"])
        elif self.method.startswith("random"):
            self.method = "random_search"
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _init_skopt(self, n_initial_points):
        space = []
        for policy in "ABCDE":
            space.append(skopt.space.Categorical(AUG_TYPES, name=f"{policy}_aug1_type"))
            space.append(skopt.space.Real(0.0, 1.0, name=f"{policy}_aug1_magnitude"))
            space.append(skopt.space.Categorical(AUG_TYPES, name=f"{policy}_aug2_type"))
            space.append(skopt.space.Real(0.0, 1.0, name=f"{policy}_aug2_magnitude"))

        self.opt = skopt.Optimizer(
            space,
            n_initial_points=n_initial_points,
            base_estimator="RF",
            acq_func="EI",
            random_state=0,
        )

    def ask(self):
        if self.method == "bayesian_optimization":
            return self.opt.ask()
        else: # random_search
            return [
                np.random.choice(AUG_TYPES), np.random.rand(),
                np.random.choice(AUG_TYPES), np.random.rand(),
            ] * 5

    def tell(self, trial_hyperparams, f_val):
        if self.method == "bayesian_optimization":
            self.opt.tell(trial_hyperparams, f_val)
