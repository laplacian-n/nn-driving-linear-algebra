"""Copy a trained checkpoint (population weights + training progress) from
one checkpoint directory into another - e.g. from fast_train.py's
checkpoints_fast/ into the normal run's checkpoints/, so a windowed run can
pick up and watch what the fast-forward trainer produced.

Only safe while nothing is currently writing to the destination: stop
whatever run owns that checkpoint dir first, clone, then start/resume it.
The two runs must share --seed, --track-seed, --num-cars and network shape -
main.py silently starts fresh instead if the cloned population doesn't match
the architecture it expects.
"""
import argparse
import os
import shutil

CHECKPOINT_FILES = ("population.npz", "meta.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="checkpoint dir to copy from, e.g. checkpoints_fast")
    p.add_argument("dest", help="checkpoint dir to copy into, e.g. checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    missing = [f for f in CHECKPOINT_FILES if not os.path.exists(os.path.join(args.source, f))]
    if missing:
        raise SystemExit(f"source checkpoint incomplete, missing: {missing}")
    os.makedirs(args.dest, exist_ok=True)
    for name in CHECKPOINT_FILES:
        src = os.path.join(args.source, name)
        dst = os.path.join(args.dest, name)
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")


if __name__ == "__main__":
    main()
