import gc
import os

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


class TimeConditionedSurvivalInference:
    def _fit_context(self, frame, time, event):
        bins = self.options["num_bins"]
        if bins < 2 or not event.any():
            raise ValueError("At least two bins and one observed training event are required.")
        self.time_grid_ = np.unique(np.quantile(time[event], np.linspace(1 / bins, (bins - 1) / bins, bins - 1)))
        repeats = len(self.time_grid_)
        context = frame[self.features_].astype(np.float64).iloc[np.repeat(np.arange(len(frame)), repeats)].reset_index(drop=True)
        cut_times = np.tile(self.time_grid_, len(frame))
        labels = np.where(cut_times < np.repeat(time, repeats), 0.0, np.where(np.repeat(event, repeats), 1.0, np.nan))
        context["_cut_time"] = cut_times
        context["_target"] = labels
        context = context.dropna(subset=["_target"]).reset_index(drop=True)
        context["_target"] = context["_target"].astype(int)
        limit = self.options["max_context_rows"]
        if limit is not None and len(context) > limit:
            context = context.sample(n=limit, random_state=self.options["seed"]).reset_index(drop=True)
        if context["_target"].nunique() != 2:
            raise ValueError("The observable training context must contain both event classes.")
        self.context_ = context
        requested = self.options["device"]
        self.device_ = "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested
        self._fit_predictor()
        return self

    def _fit_predictor(self):
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")
        torch.set_num_threads(self.options["threads"])
        settings = dict(
            n_estimators=self.options["n_estimators"],
            auto_scale_n_estimators=self.options["auto_scale_n_estimators"],
            device=self.device_,
            random_state=self.options["seed"],
            n_preprocessing_jobs=self.options["threads"],
            fit_mode="fit_preprocessors",
            memory_saving_mode="auto",
            inference_precision="auto",
        )
        if self.options["checkpoint"]:
            self.classifier_ = TabPFNClassifier(model_path=self.options["checkpoint"], **settings)
        else:
            self.classifier_ = TabPFNClassifier.create_default_for_version(ModelVersion.V3, **settings)
        try:
            self.classifier_.fit(self.context_.drop(columns=["_target"]), self.context_["_target"])
        except torch.cuda.OutOfMemoryError:
            if self.device_ == "cpu":
                raise
            self.classifier_ = None
            gc.collect()
            torch.cuda.empty_cache()
            self.device_ = "cpu"
            self._fit_predictor()

    def predict_survival(self, frame):
        if not len(frame):
            raise ValueError("The query set is empty.")
        query = frame[self.features_].astype(np.float64).iloc[np.repeat(np.arange(len(frame)), len(self.time_grid_))].reset_index(drop=True)
        query["_cut_time"] = np.tile(self.time_grid_, len(frame))
        try:
            probabilities = self.classifier_.predict_proba(query)
        except torch.cuda.OutOfMemoryError:
            if self.device_ == "cpu":
                raise
            self.classifier_ = None
            gc.collect()
            torch.cuda.empty_cache()
            self.device_ = "cpu"
            self._fit_predictor()
            probabilities = self.classifier_.predict_proba(query)
        positive = list(self.classifier_.classes_).index(1)
        survival = 1 - np.asarray(probabilities)[:, positive].reshape(len(frame), len(self.time_grid_))
        survival = np.minimum.accumulate(np.clip(survival, 0.0, 1.0), axis=1)
        return pd.DataFrame(survival, index=frame.index, columns=self.time_grid_)

    def predict_risk(self, frame):
        return np.mean(1 - self.predict_survival(frame).to_numpy(), axis=1)

    def close(self):
        self.classifier_ = None
        self.context_ = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
