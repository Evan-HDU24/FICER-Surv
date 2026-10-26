import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis


def functional_groups(columns, signatures):
    memberships = {
        name: set(signatures[name].dropna().astype(str).str.strip())
        for name in signatures
    }
    groups = {name: [] for name in memberships}
    for column in columns:
        symbol = column
        for suffix in ("_rnaseq", "_cnv", "_mut"):
            if symbol.endswith(suffix):
                symbol = symbol[:-len(suffix)]
                break
        for name, genes in memberships.items():
            if symbol in genes:
                groups[name].append(column)
                break
    return groups


def group_budgets(groups, budget):
    sizes = {name: len(columns) for name, columns in groups.items()}
    nonempty = [(name, size) for name, size in sizes.items() if size]
    if not nonempty or budget < len(nonempty) or budget > sum(sizes.values()):
        raise ValueError("The feature budget must cover each nonempty group and fit the candidate set.")
    quotas = {name: 0 for name in groups}
    total = sum(sizes.values())
    remainders = []
    for name, size in nonempty:
        raw = budget * (size / total)
        base = int(np.floor(raw))
        quotas[name] = max(1, base)
        remainders.append((name, raw - base))
    used = sum(quotas.values())
    if used > budget:
        for name, _ in sorted(remainders, key=lambda item: item[1]):
            if used == budget:
                break
            if quotas[name] > 1:
                quotas[name] -= 1
                used -= 1
    elif used < budget:
        for name, _ in sorted(remainders, key=lambda item: item[1], reverse=True):
            if used == budget:
                break
            quotas[name] += 1
            used += 1
    return quotas


def coefficient_path(frame, columns, target, options):
    x = frame[columns].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(axis=0).fillna(0.0))
    valid = x.var(axis=0, ddof=0).fillna(0.0) > 0
    names = x.columns[valid].tolist()
    if not names:
        raise ValueError("A functional group has no variable training features.")
    values = x[names].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Molecular features must be finite or missing.")
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    values = (values - values.mean(axis=0)) / scale
    estimator = CoxnetSurvivalAnalysis(
        l1_ratio=options["l1_ratio"],
        n_alphas=options["n_alphas"],
        alpha_min_ratio=options["alpha_min_ratio"],
        max_iter=options["max_iter"],
        tol=options["tolerance"],
        normalize=False,
        fit_baseline_model=False,
    )
    estimator.fit(values, target)
    return names, np.abs(np.asarray(estimator.coef_, dtype=np.float64))


def screen(names, coefficients, count, threshold):
    counts = (coefficients > threshold).sum(axis=0)
    usable = np.flatnonzero(counts > 0)
    if not len(usable):
        raise ValueError("Elastic-Net Cox returned no nonzero coefficients.")
    sufficient = usable[counts[usable] >= min(count, len(names))]
    index = (
        sufficient[np.argmin(counts[sufficient])]
        if len(sufficient)
        else usable[np.argmax(counts[usable])]
    )
    scores = pd.Series(coefficients[:, index], index=names)
    ranking = sorted(names, key=lambda name: (-scores[name], name))
    selected = [name for name in ranking if scores[name] > threshold][:count]
    return selected, scores, {name: rank for rank, name in enumerate(ranking)}
