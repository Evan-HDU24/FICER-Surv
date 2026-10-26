import numpy as np
from sksurv.util import Surv
from threadpoolctl import threadpool_limits

from .cagr import refine_selection
from .gss import coefficient_path, group_budgets, screen
from .tcsi import TimeConditionedSurvivalInference


class FICERSurv(TimeConditionedSurvivalInference):
    def __init__(self, options, core_fraction=None):
        self.options = options
        self.core_fraction = options["core_fraction"] if core_fraction is None else core_fraction
        if not 0 < self.core_fraction <= 1:
            raise ValueError("core_fraction must be in (0, 1].")

    def fit(self, frame, groups):
        time = frame[self.options["time_column"]].to_numpy(dtype=float)
        event = frame[self.options["event_column"]].to_numpy(dtype=bool)
        target = Surv.from_arrays(event=event, time=time)
        quotas = group_budgets(groups, self.options["num_features"])
        en = self.options["elastic_net"]
        self.groups_ = {}
        self.features_ = []
        with threadpool_limits(limits=self.options["selector_threads"]):
            for name, columns in groups.items():
                quota = quotas[name]
                if not quota:
                    continue
                names, coefficients = coefficient_path(frame, columns, target, en)
                original, _, ranking = screen(names, coefficients, quota, en["coefficient_threshold"])
                expanded_count = min(len(columns), int(np.ceil(self.options["pool_multiplier"] * quota)))
                expanded, scores, _ = screen(names, coefficients, expanded_count, en["coefficient_threshold"])
                if len(original) != quota:
                    raise ValueError(f"Not enough nonzero training features to fill the quota for {name}.")
                pool = list(dict.fromkeys(original + expanded))
                selected = refine_selection(frame, pool, original, scores, ranking, quota, self.core_fraction, self.options["graph"])
                self.groups_[name] = selected
                self.features_.extend(selected)
        if len(self.features_) != self.options["num_features"]:
            raise ValueError("The selected feature count differs from num_features.")
        return self._fit_context(frame, time, event)
