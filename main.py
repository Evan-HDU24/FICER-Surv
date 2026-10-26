import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

from ficer_surv import FICERSurv
from ficer_surv.data import load_data


def main():
    parser = argparse.ArgumentParser(description="TabPFN-FICER-Surv molecular survival prediction | Built with PriorLabs-TabPFN")
    parser.add_argument("--options", type=Path, default=Path(__file__).with_name("options.json"))
    parser.add_argument("--dataset", default="ucec")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--num-features", type=int)
    parser.add_argument("--core-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print("TabPFN-FICER-Surv | Built with PriorLabs-TabPFN", flush=True)
    options = json.loads(args.options.read_text(encoding="utf-8"))
    base = args.options.resolve().parent
    if args.data is None and args.dataset not in options["datasets"]:
        parser.error("Unknown dataset; select a dataset in options.json or provide --data.")
    dataset = options["datasets"].get(args.dataset, {})
    for key in ["num_features", "core_fraction", "seed", "device"]:
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    checkpoint = args.checkpoint if args.checkpoint else options["checkpoint"]
    if checkpoint:
        path = Path(checkpoint)
        path = path.resolve() if args.checkpoint else (base / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        options["checkpoint"] = str(path)
    source = args.data.resolve() if args.data else base / dataset["csv"]
    frame, groups = load_data(source, base / options["signatures"], options)
    id_col, time_col, event_col = (options[name] for name in ["id_column", "time_column", "event_column"])
    fold_col = options["fold_column"]
    available = sorted(frame[fold_col].unique().tolist())
    requested = available if args.fold == "all" else [int(args.fold)]
    if len(available) < 2 or not set(requested).issubset(available):
        raise ValueError(f"Select a held-out fold from {available}, or use --fold all.")
    root = args.output.resolve() if args.output else base / options["output_dir"] / args.dataset
    if any((root / f"fold_{fold}").exists() for fold in requested):
        raise FileExistsError("The requested output already exists. Choose a new --output directory.")
    scores = []
    for fold in requested:
        train = frame.loc[frame[fold_col] != fold]
        test = frame.loc[frame[fold_col] == fold]
        fraction = options["core_fraction"]
        if args.data is None and args.core_fraction is None:
            fractions = dataset.get("core_fractions", [])
            if fold < len(fractions):
                fraction = fractions[fold]
        print(f"{args.dataset} fold {fold}: {len(train)} train, {len(test)} test, {options['num_features']} features", flush=True)
        model = FICERSurv(copy.deepcopy(options), core_fraction=fraction)
        try:
            model.fit(train, groups)
            curves = model.predict_survival(test)
            risk = np.mean(1 - curves.to_numpy(), axis=1)
            score = float(concordance_index_censored(test[event_col].to_numpy(dtype=bool), test[time_col].to_numpy(dtype=float), risk)[0])
            folder = root / f"fold_{fold}"
            folder.mkdir(parents=True)
            prediction = test[[id_col, time_col, event_col]].copy()
            prediction["risk"] = risk
            for horizon in curves.columns:
                prediction[f"survival_{horizon:.8g}"] = curves[horizon].to_numpy()
            prediction.to_csv(folder / "predictions.csv", index=False)
            pd.DataFrame([(name, column) for name, columns in model.groups_.items() for column in columns], columns=["group", "feature"]).to_csv(folder / "features.csv", index=False)
            result = dict(fold=int(fold), c_index=score, num_features=len(model.features_), core_fraction=fraction, seed=options["seed"], device=model.device_)
            (folder / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            scores.append(result)
            print(f"C-index: {score:.6f} ({model.device_})", flush=True)
        finally:
            model.close()
    summary = dict(dataset=args.dataset, mean_c_index=float(np.mean([row["c_index"] for row in scores])), std_c_index=float(np.std([row["c_index"] for row in scores], ddof=1)) if len(scores) > 1 else None, folds=scores)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Mean C-index: {summary['mean_c_index']:.6f}", flush=True)


if __name__ == "__main__":
    main()
