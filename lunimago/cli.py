"""lun'imago CLI — train --game osu --data ./replays"""
from __future__ import annotations
import argparse
import sys


GAMES: dict[str, str] = {
    "osu": "lunimago.games.osu.plugin",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lunimago",
        description="lun'imago — game imitation learning framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a model on replay data")
    train.add_argument("--game",     required=True, choices=list(GAMES), help="Game plugin to use")
    train.add_argument("--data",     required=True, help="Directory containing replay + beatmap files")
    train.add_argument("--out",      default="checkpoints", help="Output directory for checkpoints")
    train.add_argument("--epochs",   type=int,   default=20)
    train.add_argument("--batch",    type=int,   default=128)
    train.add_argument("--window",   type=int,   default=32, help="Context window size (frames)")
    train.add_argument("--device",   default="auto", help="auto | cuda | directml | mps | cpu")
    train.add_argument("--export",   default=None, help="Export final model to this .onnx path")

    args = parser.parse_args()

    if args.command == "train":
        _cmd_train(args)


def _cmd_train(args: argparse.Namespace) -> None:
    import importlib
    from lunimago.core.device  import resolve_device
    from lunimago.core.trainer import Trainer
    from lunimago.core.dataset import ReplayDataset
    from lunimago.core.export.onnx_exporter import export_onnx

    plugin = importlib.import_module(GAMES[args.game])
    parser  = plugin.make_parser()
    model   = plugin.make_model()
    device  = resolve_device(args.device)

    print(f"Device: {device}")
    print(f"Game:   {args.game}  |  feature_dim={parser.feature_dim}  action_dim={parser.action_dim}")

    pairs = plugin.collect_pairs(args.data)
    if not pairs:
        print(f"No replay pairs found in {args.data}", file=sys.stderr)
        sys.exit(1)

    print(f"Replays: {len(pairs)}")
    dataset = ReplayDataset(pairs, parser, window=args.window)
    print(f"Samples: {len(dataset)}")

    trainer = Trainer(model, device)
    trainer.fit(dataset, epochs=args.epochs, batch_size=args.batch, checkpoint_dir=args.out)

    if args.export:
        export_onnx(model, args.export, window=args.window)


if __name__ == "__main__":
    main()
