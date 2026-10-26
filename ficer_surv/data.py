import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .gss import functional_groups


def load_data(source, signature_path, options):
    frame = pd.read_csv(source, low_memory=False, float_precision="round_trip")
    id_col, time_col, event_col = (options[name] for name in ["id_column", "time_column", "event_column"])
    required = {id_col, time_col, event_col}
    if not required.issubset(frame):
        raise ValueError(f"CSV is missing columns: {sorted(required - set(frame.columns))}")
    if frame[id_col].isna().any() or frame[id_col].duplicated().any():
        raise ValueError("Patient IDs must be present and unique.")
    time = pd.to_numeric(frame[time_col], errors="raise").to_numpy(dtype=float)
    event = pd.to_numeric(frame[event_col], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(time).all() or np.any(time < 0) or not np.isin(event, [0, 1]).all():
        raise ValueError("Time must be finite and nonnegative; event must be 0 or 1.")
    fold_col = options["fold_column"]
    if fold_col not in frame:
        folds = np.empty(len(frame), dtype=int)
        splitter = StratifiedKFold(n_splits=options["num_folds"], shuffle=True, random_state=options["seed"])
        for fold, (_, test) in enumerate(splitter.split(frame, event)):
            folds[test] = fold
        frame[fold_col] = folds
    raw_folds = pd.to_numeric(frame[fold_col], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(raw_folds).all() or not np.equal(raw_folds, np.floor(raw_folds)).all() or np.any(raw_folds < 0):
        raise ValueError("Fold assignments must be nonnegative integers.")
    frame[fold_col] = raw_folds.astype(int)
    signatures = pd.read_csv(signature_path)
    metadata = {id_col, time_col, event_col, fold_col, "censorship", "slide_id", "site", "age", "is_female", "oncotree_code", "train"}
    groups = functional_groups([column for column in frame if column not in metadata], signatures)
    return frame, groups
