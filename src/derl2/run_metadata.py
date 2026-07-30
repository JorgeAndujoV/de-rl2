"""run_metadata.json — the single authoritative record of a job.

This replaces the three per-job metadata files de-rl2 used to write
(config_used.yaml, git_commit.txt, key_result.txt). It is written once at job
start with exit_status "running" and rewritten at job end with the final status,
so a run killed by a Slurm timeout is still labelled correctly rather than
looking half-finished.

`resolved_config` is the config AFTER --smoke scaling and --set overrides are
applied, so this file — not the experiment's on-disk config.yaml — is what
actually ran. This matters for smoke mode, which scales the config down (D=5,
budget=3000): config.yaml is not what ran, run_metadata.json is.

Ownership: run_experiment.sh writes the start record (before any training, so a
crash in training still leaves a "running" record the persist trap can flip to
"failed") and the finish record (in the persist trap). train.py writes a start
record only as a fallback when one does not already exist, so a hand-run
`python -m derl2.training.train` outside run_experiment.sh still leaves a
run_metadata.json that evaluate.py can read.

    python -m derl2.run_metadata start  --out-dir DIR --config CFG --kind KIND \
        [--smoke] [--slurm-job-id ID] [--commit SHA] [--git-dirty] [--set K=V ...]
    python -m derl2.run_metadata finish --out-dir DIR --status completed|timeout|failed
"""

import argparse
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone

from derl2.config import Config

METADATA_FILE = "run_metadata.json"

# Field order matches the schema in the build spec; json.dump keeps it because
# we build an ordinary dict and do not sort keys.
_SCHEMA_ORDER = [
    "experiment", "mode", "git_commit", "git_dirty", "slurm_job_id", "hostname",
    "started_at", "finished_at", "wall_seconds", "exit_status",
    "python_version", "tensorflow_version", "numpy_version", "resolved_config",
]

_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def path_for(out_dir):
    return os.path.join(out_dir, METADATA_FILE)


def _now_iso():
    return datetime.now(timezone.utc).strftime(_TS_FORMAT)


def _versions():
    """Interpreter and library versions, so a result traces to its stack.

    numpy/tensorflow are imported lazily and defensively: writing metadata must
    never be what makes a job fall over.
    """
    versions = {"python_version": platform.python_version(),
                "numpy_version": None, "tensorflow_version": None}
    try:
        import numpy
        versions["numpy_version"] = numpy.__version__
    except Exception:
        pass
    try:
        import tensorflow as tf
        versions["tensorflow_version"] = tf.__version__
    except Exception:
        pass
    return versions


def _git(cwd, args):
    try:
        out = subprocess.run(["git", "-C", cwd, *args],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except Exception:
        return ""


def build_start(cfg, kind, slurm_job_id=None, commit=None, git_dirty=None,
                cwd=None):
    """Assemble the start-of-job metadata dict from a resolved Config."""
    cwd = cwd or os.getcwd()
    if commit is None:
        commit = _git(cwd, ["rev-parse", "HEAD"]) or None
    if git_dirty is None:
        git_dirty = bool(_git(cwd, ["status", "--porcelain"]))
    versions = _versions()
    return {
        "experiment": cfg.exp_id,
        "mode": kind,
        "git_commit": commit,
        "git_dirty": git_dirty,
        "slurm_job_id": slurm_job_id or os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "started_at": _now_iso(),
        "finished_at": None,
        "wall_seconds": None,
        "exit_status": "running",
        "python_version": versions["python_version"],
        "tensorflow_version": versions["tensorflow_version"],
        "numpy_version": versions["numpy_version"],
        "resolved_config": cfg.as_dict(),
    }


def _dump(meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ordered = {k: meta[k] for k in _SCHEMA_ORDER if k in meta}
    ordered.update({k: v for k, v in meta.items() if k not in ordered})
    with open(path_for(out_dir), "w") as fh:
        json.dump(ordered, fh, indent=2)
        fh.write("\n")


def write_start(out_dir, cfg, kind, **kwargs):
    meta = build_start(cfg, kind, **kwargs)
    _dump(meta, out_dir)
    return meta


def ensure_start(out_dir, cfg, kind, **kwargs):
    """Write a start record only if none exists (train.py's dev fallback)."""
    if not os.path.exists(path_for(out_dir)):
        return write_start(out_dir, cfg, kind, **kwargs)
    return None


def write_finish(out_dir, status):
    """Rewrite the record with the final status and wall time.

    Robust to a missing start record (a crash before start was written): it
    synthesises a minimal one so a failed job is still labelled.
    """
    path = path_for(out_dir)
    if os.path.exists(path):
        with open(path) as fh:
            meta = json.load(fh)
    else:
        meta = {k: None for k in _SCHEMA_ORDER}
    meta["finished_at"] = _now_iso()
    meta["exit_status"] = status
    started = meta.get("started_at")
    if started:
        try:
            t0 = datetime.strptime(started, _TS_FORMAT).replace(
                tzinfo=timezone.utc)
            t1 = datetime.strptime(meta["finished_at"], _TS_FORMAT).replace(
                tzinfo=timezone.utc)
            meta["wall_seconds"] = int((t1 - t0).total_seconds())
        except Exception:
            meta["wall_seconds"] = None
    _dump(meta, out_dir)
    return meta


def load_resolved_config(train_dir):
    """Return the Config that a job actually ran, from its run_metadata.json.

    Evaluation reads this rather than the experiment's config.yaml, so a smoke
    evaluation uses the smoke-scaled config and never the unscaled on-disk one.
    """
    path = path_for(train_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No {METADATA_FILE} in {train_dir}. Evaluation must use the "
            f"config the training job actually ran, recorded under "
            f"'resolved_config' in {METADATA_FILE}."
        )
    with open(path) as fh:
        meta = json.load(fh)
    if "resolved_config" not in meta or not meta["resolved_config"]:
        raise ValueError(f"{path} has no 'resolved_config' block.")
    return Config(meta["resolved_config"], source=path)


# ------------------------------------------------------------------- CLI

def _main():
    parser = argparse.ArgumentParser(description="Read/write run_metadata.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start")
    p_start.add_argument("--out-dir", required=True)
    p_start.add_argument("--config", required=True)
    p_start.add_argument("--kind", required=True)
    p_start.add_argument("--smoke", action="store_true")
    p_start.add_argument("--slurm-job-id", default=None)
    p_start.add_argument("--commit", default=None)
    p_start.add_argument("--git-dirty", action="store_true")
    # --set mirrors the training/config CLI so overrides land in resolved_config.
    p_start.add_argument("--set", action="append", default=[], dest="overrides",
                         metavar="KEY=VALUE")

    p_finish = sub.add_parser("finish")
    p_finish.add_argument("--out-dir", required=True)
    p_finish.add_argument("--status", required=True,
                          choices=["running", "completed", "timeout", "failed"])

    # Tolerate stray args forwarded from run_experiment.sh's EXTRA_ARGS
    # (e.g. --train-dir for the eval path); only --set is meaningful here.
    args, _unknown = parser.parse_known_args()

    if args.cmd == "start":
        cfg = Config.from_file(args.config, args.overrides, smoke=args.smoke)
        # An explicit --commit means the caller (run_experiment.sh) is
        # authoritative about git state, so trust its --git-dirty flag too.
        commit = args.commit
        git_dirty = args.git_dirty if commit is not None else None
        write_start(args.out_dir, cfg, args.kind,
                    slurm_job_id=args.slurm_job_id, commit=commit,
                    git_dirty=git_dirty)
        print(f"wrote {path_for(args.out_dir)} (running)")
    elif args.cmd == "finish":
        meta = write_finish(args.out_dir, args.status)
        print(f"wrote {path_for(args.out_dir)} ({meta['exit_status']}, "
              f"wall {meta.get('wall_seconds')}s)")


if __name__ == "__main__":
    _main()
