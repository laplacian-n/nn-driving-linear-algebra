"""Headless training at full CPU speed, kept in its own checkpoint/log
lineage so it never collides with a normal (windowed, 60fps-capped) run
reading and writing checkpoints/ at the same time.

This is a thin wrapper, not a second simulation: --headless already runs as
fast as the CPU allows (main.py's render block, including the clock.tick(60)
cap, only runs when a window is open) - all this does is default to headless
and to separate directories, so you can point it at a copy of a real
checkpoint and let it run flat-out without touching the live one. Use
clone_checkpoint.py afterwards to bring the result over to whatever
checkpoint dir a normal run reads from.

Accepts every main.py flag (--num-cars, --generations, --seed, --track-seed,
--checkpoint-dir, --log-dir, ...) and passes them straight through - only
--headless and default dir values are injected when not already given.
"""
import sys

from main import main

DEFAULT_CHECKPOINT_DIR = "checkpoints_fast"
DEFAULT_LOG_DIR = "logs_fast"

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--headless" not in argv:
        argv = ["--headless"] + argv
    if "--checkpoint-dir" not in argv:
        argv += ["--checkpoint-dir", DEFAULT_CHECKPOINT_DIR]
    if "--log-dir" not in argv:
        argv += ["--log-dir", DEFAULT_LOG_DIR]
    sys.argv = [sys.argv[0]] + argv
    main()
