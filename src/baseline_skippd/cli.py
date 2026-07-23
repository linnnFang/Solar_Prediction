"""
CLI for the SKIPP'D pipeline.

    python -m src.baseline_skippd.cli inspect      --config configs/skippd/pv_only_mlp.yaml
    python -m src.baseline_skippd.cli build-index  --config ...
    python -m src.baseline_skippd.cli train        --config ... --model sunset_forecast
    python -m src.baseline_skippd.cli train        --config ... --model vanilla_transformer
    python -m src.baseline_skippd.cli evaluate      --config ... --model ...
    python -m src.baseline_skippd.cli run           --config ... --model ...   (train + evaluate)
"""

import argparse
import json

from src.baseline_skippd import notebook_api as api


def _dm(cfg):
    dm = api.build_skippd_datamodule(cfg)
    dm.setup()
    return dm


def main(argv=None):
    p = argparse.ArgumentParser(prog="baseline_skippd")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "build-index", "train", "evaluate", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
        if name in ("train", "evaluate", "run"):
            sp.add_argument("--model", required=True)
    args = p.parse_args(argv)
    cfg = api.load_skippd_config(args.config)

    if args.cmd == "inspect":
        store = api.SKIPPDProcessedStore(cfg["data"]["root"])
        print(json.dumps(store.report(), indent=2))
    elif args.cmd == "build-index":
        dm = _dm(cfg)
        print(f"windows: {len(dm.window_index)} | test samples: {len(dm.window_index.subset('test'))}")
    elif args.cmd in ("train", "run"):
        dm = _dm(cfg)
        if args.model in ("naive_persistence", "clear_sky_persistence"):
            print(json.dumps(api.run_rule_baseline(args.model, dm, cfg), indent=2)); return
        model = api.build_model(args.model, cfg, dm)
        api.train_model(model, dm, cfg)
        if args.cmd == "run":
            print(json.dumps(api.evaluate_model(model, dm, cfg), indent=2))
    elif args.cmd == "evaluate":
        dm = _dm(cfg)
        if args.model in ("naive_persistence", "clear_sky_persistence"):
            print(json.dumps(api.run_rule_baseline(args.model, dm, cfg), indent=2)); return
        model = api.build_model(args.model, cfg, dm)
        print(json.dumps(api.evaluate_model(model, dm, cfg), indent=2))


if __name__ == "__main__":
    main()
