from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pandas as pd

from src.config import load_config
from src.data.manifest import build_manifest, write_manifest, check_split_overlap
from src.data.validation import validate_split
from src.pipeline.cqcc import extract_cqcc_split
from src.pipeline.tkeo import extract_tkeo_split
from src.pipeline.acoustic import extract_acoustic_split
from src.pipeline.fusion import build_fusion_splits, build_group_feature_matrix, combine_group_feature_matrices
from src.models.xgb_fusion import train_xgb_fusion
from src.models.explain import explain_xgb_model


def get_splits(config, split_names):
    resolved = {}
    for split in split_names:
        paths = config.split_paths(split)
        resolved[split] = paths
    return resolved


def command_inspect(config, split_names):
    manifests = {}
    for split, paths in get_splits(config, split_names).items():
        print(f"\n[{split.upper()}]")
        print(f"Protocol : {paths.protocol_path} (exists={paths.protocol_path.is_file()})")
        print(f"Audio dir: {paths.audio_dir} (exists={paths.audio_dir.is_dir()})")
        if not paths.protocol_path.is_file():
            continue
        df = build_manifest(paths.protocol_path, paths.audio_dir, split)
        manifests[split] = df
        print(f"Trials   : {len(df):,}")
        print("Classes  :")
        print(df["key"].value_counts().to_string())
        print("Attack IDs:")
        print(df["attack_id"].value_counts(dropna=False).to_string())
    if len(manifests) > 1:
        overlap = check_split_overlap(manifests)
        print("\nSplit overlap:")
        print(json.dumps(overlap, indent=2))


def command_manifest(config, split_names):
    manifests = {}
    for split, paths in get_splits(config, split_names).items():
        df = build_manifest(paths.protocol_path, paths.audio_dir, split)
        output = config.manifest_dir / f"PA_{split}.csv"
        write_manifest(df, output)
        manifests[split] = df
        print(f"{split}: {len(df):,} rows -> {output}")
    if len(manifests) > 1:
        overlap = check_split_overlap(manifests)
        print("\nSplit overlap:")
        print(json.dumps(overlap, indent=2))
        if overlap["has_overlap"]:
            raise SystemExit("ERROR: utterance IDs overlap between supplied splits.")


def command_validate(config, split_names):
    audio_cfg = config.raw["audio"]
    summaries = []
    for split, paths in get_splits(config, split_names).items():
        summary = validate_split(
            protocol_path=paths.protocol_path,
            audio_dir=paths.audio_dir,
            split=split,
            expected_sample_rate=int(audio_cfg.get("expected_sample_rate", 16000)),
            allowed_channels=tuple(audio_cfg.get("allowed_channels", [1, 2])),
            min_duration_seconds=float(audio_cfg.get("min_duration_seconds", 0.1)),
            max_duration_seconds=audio_cfg.get("max_duration_seconds"),
            decode_check=bool(audio_cfg.get("decode_check", True)),
            output_dir=config.validation_dir,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    if len(summaries) > 1:
        manifests = {}
        for split, paths in get_splits(config, split_names).items():
            manifests[split] = build_manifest(paths.protocol_path, paths.audio_dir, split)
        overlap = check_split_overlap(manifests)
        print("\nCross-split overlap:")
        print(json.dumps(overlap, indent=2))
        if overlap["has_overlap"]:
            raise SystemExit("ERROR: utterance IDs overlap between supplied splits.")


def main():
    parser = argparse.ArgumentParser(description="ASVspoof PA data tools")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("inspect", "manifest", "validate", "extract-cqcc", "extract-tkeo", "extract-acoustic", "build-fusion", "build-group-matrix", "combine-fusion", "train-xgb", "explain-xgb"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", default="configs/data.yaml")
        cmd.add_argument(
            "--splits",
            nargs="+",
            default=["train", "dev"],
            choices=["train", "dev", "eval"],
        )
        if name in ("extract-cqcc", "extract-tkeo", "extract-acoustic"):
            cmd.add_argument("--overwrite", action="store_true")
            cmd.add_argument("--limit", type=int, default=None)
            cmd.add_argument("--cache-root", default=None,
                             help="Override feature-cache root (use /kaggle/temp for large temporary caches)")
            default_workers = min(os.cpu_count() or 1, 8)
            cmd.add_argument(
                "--workers", type=int, default=default_workers,
                help=f"Parallel worker processes (default: {default_workers}; use 1 for serial)",
            )
        elif name == "build-fusion":
            cmd.add_argument("--feature-groups", nargs="+", default=["cqcc", "tkeo", "acoustic"],
                             choices=["cqcc", "tkeo", "acoustic"])
            cmd.add_argument("--stats", nargs="+", default=["mean", "std", "p10", "p50", "p90"],
                             choices=["mean", "std", "p10", "p50", "p90"])
            cmd.add_argument("--output-dir", default="results/fusion_features")
            cmd.add_argument("--overwrite", action="store_true")
        elif name == "build-group-matrix":
            cmd.add_argument("--feature-group", required=True, choices=["cqcc", "tkeo", "acoustic"])
            cmd.add_argument("--cache-root", default=None)
            cmd.add_argument("--output-dir", default="results/group_features")
            cmd.add_argument("--stats", nargs="+", default=["mean", "std", "p10", "p50", "p90"],
                             choices=["mean", "std", "p10", "p50", "p90"])
            cmd.add_argument("--overwrite", action="store_true")
        elif name == "combine-fusion":
            cmd.add_argument("--feature-groups", nargs="+", default=["cqcc", "tkeo", "acoustic"],
                             choices=["cqcc", "tkeo", "acoustic"])
            cmd.add_argument("--group-feature-dir", default="results/group_features")
            cmd.add_argument("--output-dir", default="results/fusion_features")
            cmd.add_argument("--overwrite", action="store_true")
        elif name == "train-xgb":
            cmd.add_argument("--feature-groups", nargs="+", default=["cqcc", "tkeo", "acoustic"],
                             choices=["cqcc", "tkeo", "acoustic"])
            cmd.add_argument("--feature-dir", default="results/fusion_features")
            cmd.add_argument("--model-dir", default="results/xgb_fusion")
            cmd.add_argument("--n-estimators", type=int, default=None)
            cmd.add_argument("--early-stopping-rounds", type=int, default=None)
            cmd.add_argument("--max-depth", type=int, default=None)
            cmd.add_argument("--learning-rate", type=float, default=None)
            cmd.add_argument("--n-jobs", type=int, default=None)
            cmd.add_argument("--device", choices=["cpu", "cuda"], default=None,
                             help="XGBoost training device (default: config or cpu)")
        elif name == "explain-xgb":
            cmd.add_argument("--feature-dir", default="results/fusion_features")
            cmd.add_argument("--model-dir", default="results/xgb_fusion")
            cmd.add_argument("--split", choices=["train", "dev", "eval"], default="dev")
            cmd.add_argument("--max-samples", type=int, default=1000)

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "inspect":
        command_inspect(config, args.splits)
    elif args.command == "manifest":
        command_manifest(config, args.splits)
    elif args.command == "validate":
        command_validate(config, args.splits)
    elif args.command == "extract-cqcc":
        if args.limit is not None and args.limit < 1:
            raise SystemExit("--limit must be a positive integer")
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        print(f"Using {args.workers} worker process(es); BLAS/OpenMP limited to 1 thread per worker.")
        for split in args.splits:
            manifest_path = config.manifest_dir / f"PA_{split}.csv"
            summary = extract_cqcc_split(
                manifest_path=manifest_path,
                split=split,
                cache_root=args.cache_root or config.cache_dir,
                expected_sample_rate=config.expected_sample_rate,
                overwrite=args.overwrite,
                limit=args.limit,
                workers=args.workers,
            )
            summary_path = config.processed_dir / "feature_reports" / f"cqcc_{split}_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
            print(f"Summary saved to: {summary_path}")
            if summary["failed_count"]:
                print(f"WARNING: {summary['failed_count']} utterances failed.")

    elif args.command == "extract-tkeo":
        if args.limit is not None and args.limit < 1:
            raise SystemExit("--limit must be a positive integer")
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        feature_config = config.raw.get("features", {}).get("tkeo", {})
        print(f"Using {args.workers} worker process(es); BLAS/OpenMP limited to 1 thread per worker.")
        for split in args.splits:
            summary = extract_tkeo_split(
                manifest_path=config.manifest_dir / f"PA_{split}.csv", split=split,
                cache_root=args.cache_root or config.cache_dir, expected_sample_rate=config.expected_sample_rate,
                feature_config=feature_config, overwrite=args.overwrite,
                limit=args.limit, workers=args.workers,
            )
            summary_path = config.processed_dir / "feature_reports" / f"tkeo_{split}_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
            print(f"Summary saved to: {summary_path}")

    elif args.command == "extract-acoustic":
        if args.limit is not None and args.limit < 1:
            raise SystemExit("--limit must be a positive integer")
        if args.workers < 1:
            raise SystemExit("--workers must be at least 1")
        feature_config = config.raw.get("features", {}).get("acoustic", {})
        print(f"Using {args.workers} worker process(es); BLAS/OpenMP limited to 1 thread per worker.")
        for split in args.splits:
            summary = extract_acoustic_split(
                manifest_path=config.manifest_dir / f"PA_{split}.csv", split=split,
                cache_root=args.cache_root or config.cache_dir, expected_sample_rate=config.expected_sample_rate,
                feature_config=feature_config, overwrite=args.overwrite,
                limit=args.limit, workers=args.workers,
            )
            summary_path = config.processed_dir / "feature_reports" / f"acoustic_{split}_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
            print(f"Summary saved to: {summary_path}")

    elif args.command == "build-fusion":
        results = build_fusion_splits(
            config.manifest_dir, config.cache_dir, args.output_dir, args.splits,
            feature_groups=args.feature_groups, stats=args.stats, overwrite=args.overwrite,
        )
        print(json.dumps(results, indent=2))

    elif args.command == "build-group-matrix":
        cache_root = args.cache_root or config.cache_dir
        results = []
        for split in args.splits:
            results.append(build_group_feature_matrix(
                config.manifest_dir / f"PA_{split}.csv", cache_root,
                args.output_dir, split, args.feature_group,
                stats=args.stats, overwrite=args.overwrite,
            ))
        print(json.dumps(results, indent=2))

    elif args.command == "combine-fusion":
        results = combine_group_feature_matrices(
            args.group_feature_dir, args.output_dir, args.splits,
            args.feature_groups, overwrite=args.overwrite,
        )
        print(json.dumps(results, indent=2))

    elif args.command == "train-xgb":
        model_cfg = config.raw.get("model", {}).get("xgboost", {})
        result = train_xgb_fusion(
            args.feature_dir, args.model_dir, feature_groups=args.feature_groups,
            n_estimators=args.n_estimators or int(model_cfg.get("n_estimators", 2000)),
            max_depth=args.max_depth or int(model_cfg.get("max_depth", 5)),
            learning_rate=args.learning_rate or float(model_cfg.get("learning_rate", 0.03)),
            early_stopping_rounds=(args.early_stopping_rounds or
                                   int(model_cfg.get("early_stopping_rounds", 100))),
            n_jobs=args.n_jobs or int(model_cfg.get("n_jobs", min(os.cpu_count() or 1, 6))),
            device=args.device or str(model_cfg.get("device", "cpu")),
            random_state=int(config.raw.get("project", {}).get("seed", 42)),
            use_class_weight=str(model_cfg.get("class_weight", "balanced")).lower() == "balanced",
        )
        print(json.dumps(result, indent=2))

    elif args.command == "explain-xgb":
        result = explain_xgb_model(
            args.feature_dir, args.model_dir, split=args.split,
            max_samples=args.max_samples,
            random_state=int(config.raw.get("project", {}).get("seed", 42)),
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
