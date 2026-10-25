import argparse
from dataclasses import fields
import json
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(prog="pepclip")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Filter UniRef90 FASTA and build linear peptide graphs")
    prepare.add_argument("--fasta", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--source-version", required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--val-fraction", type=float, default=0.1)
    prepare.add_argument("--test-fraction", type=float, default=0.1)
    prepare.add_argument("--groups", help="Optional CSV: sample_id,group_id")
    extract = commands.add_parser("extract")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--modality", choices=["sequence", "graph"], required=True)
    extract.add_argument("--encoder", required=True)
    extract.add_argument("--model", default="esm2_t33_650M_UR50D")
    extract.add_argument("--checkpoint")
    extract.add_argument("--layer", type=int)
    extract.add_argument("--trusted-checkpoint", action="store_true")
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--batch-size", type=int, default=16)
    train = commands.add_parser("train")
    for name in ("manifest", "sequence-cache", "graph-cache", "output"):
        train.add_argument("--" + name, required=True)
    train.add_argument("--config")
    train.add_argument("--resume", help="Resume last.pt into a new output directory")
    from .train import TrainConfig
    for field in fields(TrainConfig):
        train.add_argument("--" + field.name.replace("_", "-"), type=field.type)
    evaluate = commands.add_parser("evaluate")
    for name in ("checkpoint", "manifest", "sequence-cache", "graph-cache"):
        evaluate.add_argument("--" + name, required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--device", default="cpu")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        from .data import prepare
        print(json.dumps(prepare(args.fasta, args.output, args.source_version, args.seed,
                                 args.val_fraction, args.test_fraction, args.groups), indent=2))
    elif args.command == "extract":
        from .encoders import build_encoder
        from .cache import extract
        kwargs = dict(device=args.device)
        if args.modality == "sequence":
            kwargs.update(model_name=args.model, checkpoint=args.checkpoint, layer=args.layer)
        else:
            if not args.checkpoint:
                raise ValueError("Graph extraction requires --checkpoint")
            kwargs.update(checkpoint=args.checkpoint, trusted_checkpoint=args.trusted_checkpoint)
        encoder = build_encoder(args.modality, args.encoder, **kwargs)
        extract(args.manifest, args.output, encoder, args.modality, args.batch_size)
    elif args.command == "train":
        from .train import TrainConfig, train
        config = json.loads(Path(args.config).read_text()) if args.config else {}
        config.update({f.name: getattr(args, f.name) for f in fields(TrainConfig)
                       if getattr(args, f.name) is not None})
        train(args.manifest, args.sequence_cache, args.graph_cache, args.output,
              TrainConfig(**config), args.resume)
    else:
        from .train import TrainConfig, load_checkpoint, load_inputs, model_from_checkpoint
        from .cache import PairedFeatures
        from .evaluate import evaluate
        state = load_checkpoint(args.checkpoint)
        config = TrainConfig(**state["config"])
        rows, seq, graph, provenance = load_inputs(args.manifest, args.sequence_cache, args.graph_cache, config)
        if provenance != state["provenance"]:
            raise ValueError("Evaluation inputs differ from checkpoint provenance")
        metrics = evaluate(model_from_checkpoint(state, args.device),
            PairedFeatures(rows, seq, graph, args.split), args.device, config.batch_size, config.temperature)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
