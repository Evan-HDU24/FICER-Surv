import numpy as np
from scipy.linalg import solve


def correlations(frame):
    values = frame.to_numpy(dtype=np.float64)
    medians = np.nan_to_num(np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0))
    values = np.where(np.isfinite(values), values, medians)
    values = values - values.mean(axis=0)
    lengths = np.linalg.norm(values, axis=0)
    if np.any(lengths <= 1e-10):
        raise ValueError("A selected candidate is constant in the training set.")
    values = values / np.maximum(lengths, 1e-10)
    rho = np.clip(values.T @ values, -1.0, 1.0)
    np.fill_diagonal(rho, 1.0)
    return rho


def graph_refinement(quality, rho, options):
    adjacency = rho ** 2
    np.fill_diagonal(adjacency, 0)
    adjacency[np.abs(rho) < options["correlation_threshold"]] = 0
    neighbors = min(options["neighbors"], max(0, len(quality) - 1))
    keep = np.zeros_like(adjacency, dtype=bool)
    if neighbors:
        indices = np.argsort(-adjacency, axis=1, kind="stable")[:, :neighbors]
        keep[np.arange(len(quality))[:, None], indices] = True
    adjacency = np.where(keep | keep.T, adjacency, 0)
    degrees = adjacency.sum(axis=1)
    inverse = np.divide(1.0, np.sqrt(degrees), out=np.zeros(len(quality)), where=degrees > 0)
    laplacian = np.diag((degrees > 0).astype(float)) - inverse[:, None] * adjacency * inverse[None, :]
    refined = solve(np.eye(len(quality)) + options["regularization"] * laplacian, quality, assume_a="pos")
    refined = np.maximum(refined, 0)
    return refined / max(refined.max(), 1e-12)


def refine_selection(frame, pool, original, scores, ranking, quota, core_fraction, options):
    rho = correlations(frame[pool])
    quality = np.nan_to_num(np.array([scores.get(column, 0.0) for column in pool]))
    quality = quality / max(quality.max(), 1e-12)
    refined = graph_refinement(quality, rho, options)
    count = max(1, int(np.ceil(core_fraction * quota)))
    picked = [pool.index(column) for column in original[:count]]
    while len(picked) < quota:
        available = refined.copy()
        available[picked] = -np.inf
        picked.append(int(np.argmax(available)))
    selected = sorted((pool[index] for index in picked), key=lambda column: (ranking.get(column, 10**9), column))
    return selected
