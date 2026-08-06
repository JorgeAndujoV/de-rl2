"""Vectorized environment for on-policy rollout collection (EXP010).

A VecEnv steps N independent copies of the SAME DEEnv and auto-resets any copy
whose episode ends, exposing the standard vectorized interface:

    obs = venv.reset()                      # (N, obs_dim)
    next_obs, rewards, dones, infos = venv.step(actions)   # actions: length-N

`step` returns, for each env, the reward and done of the step just taken and the
info dict of that (possibly terminal) step; when an env is done, its slot in
`next_obs` already holds the FIRST observation of its next episode (auto-reset),
so the caller never has to reset explicitly. dones is float (1.0/0.0) so it drops
straight into GAE masking.

Two implementations share this contract:
  * SerialVecEnv    — steps the N envs in-process, one after another. Correct and
                      simple; used to validate the training loop. Slow (no
                      parallelism), but semantically identical to the parallel one.
  * (SubprocVecEnv) — the same contract with N worker processes, added next for
                      throughput; a drop-in replacement.

Per-env episode seeds are a deterministic function of (base_seed, env index,
that env's episode counter), so a rollout is reproducible and the N streams
neither overlap each other nor the evaluation seed bands. Function selection is
left to DEEnv.reset (random per episode from the configured set), matching how
single-env training samples functions.
"""

import multiprocessing as mp

import numpy as np

from derl2.environments.env import DEEnv


def _episode_seed(base_seed, env_index, episode_count):
    # Wide, collision-free separation: env streams are 1e6 apart, well clear of
    # the 8e5/9e5 evaluation seed bands.
    return int(base_seed) * 1_000_000_000 + int(env_index) * 1_000_000 \
        + int(episode_count)


class SerialVecEnv:
    def __init__(self, cfg, n_envs, base_seed=0):
        self.n_envs = int(n_envs)
        self.base_seed = base_seed
        self.envs = [DEEnv.from_config(cfg) for _ in range(self.n_envs)]
        self._ep = [0] * self.n_envs
        self.obs_dim = self.envs[0].obs_dim

    def reset(self):
        obs = [e.reset(seed=_episode_seed(self.base_seed, i, self._ep[i]))[0]
               for i, e in enumerate(self.envs)]
        return np.asarray(obs, dtype=np.float32)

    def step(self, actions):
        next_obs, rewards, dones, infos = [], [], [], []
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, info = e.step(a)
            done = bool(term or trunc)
            if done:
                # auto-reset: return the NEXT episode's first obs, but keep this
                # step's terminal info (its tau/best_error) for the caller.
                self._ep[i] += 1
                o = e.reset(seed=_episode_seed(self.base_seed, i, self._ep[i]))[0]
            next_obs.append(o)
            rewards.append(r)
            dones.append(1.0 if done else 0.0)
            infos.append(info)
        return (np.asarray(next_obs, dtype=np.float32),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=np.float32), infos)

    def close(self):
        pass


def _worker(conn, cfg, env_index, base_seed):
    """One VecEnv worker process: build a DEEnv from cfg and serve reset/step
    over the pipe, auto-resetting on episode end with the SAME per-episode seed
    as SerialVecEnv (so the two backends are bit-for-bit identical). Each worker
    pins TF to a single thread so N workers don't oversubscribe the cpus."""
    import os
    for var in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
                "TF_NUM_INTEROP_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        import tensorflow as tf
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass                      # threads already initialized; harmless
        from derl2.environments.env import DEEnv as _DEEnv
        env = _DEEnv.from_config(cfg)
        ep = 0
        conn.send(("ready", None))
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                o, _ = env.reset(seed=_episode_seed(base_seed, env_index, ep))
                conn.send(("ok", o))
            elif cmd == "step":
                o, r, term, trunc, info = env.step(data)
                done = bool(term or trunc)
                if done:
                    ep += 1
                    o, _ = env.reset(
                        seed=_episode_seed(base_seed, env_index, ep))
                conn.send(("ok", (o, float(r), done, float(info["tau"]))))
            elif cmd == "close":
                conn.close()
                return
    except Exception:
        import traceback
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass


class SubprocVecEnv:
    """The SerialVecEnv contract, run across N worker processes for throughput.

    Uses the 'spawn' start method: fork-after-TF-init deadlocks, so each worker
    is a fresh process that imports TF cleanly. Semantically identical to
    SerialVecEnv — same per-episode seeds, same auto-reset — so swapping backends
    changes only wall-clock, never the transitions or any output. infos carry the
    'tau' the on-policy loop needs (the full info dict, with its heavy trajectory
    arrays, is not shipped over the pipe)."""

    def __init__(self, cfg, n_envs, base_seed=0):
        self.n_envs = int(n_envs)
        self.base_seed = base_seed
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i in range(self.n_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, cfg, i, base_seed),
                            daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        for c in self.conns:            # wait for each worker to build its env
            self._recv(c)
        self.obs_dim = None

    @staticmethod
    def _recv(conn):
        tag, payload = conn.recv()
        if tag == "error":
            raise RuntimeError(f"VecEnv worker failed:\n{payload}")
        return payload

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        obs = [self._recv(c) for c in self.conns]
        out = np.asarray(obs, dtype=np.float32)
        self.obs_dim = out.shape[1]
        return out

    def step(self, actions):
        for c, a in zip(self.conns, actions):
            c.send(("step", a))
        results = [self._recv(c) for c in self.conns]
        next_obs = np.asarray([r[0] for r in results], dtype=np.float32)
        rewards = np.asarray([r[1] for r in results], dtype=np.float32)
        dones = np.asarray([1.0 if r[2] else 0.0 for r in results],
                           dtype=np.float32)
        infos = [{"tau": r[3]} for r in results]
        return next_obs, rewards, dones, infos

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def make_vec_env(cfg, n_envs, base_seed=0, backend=None):
    """Serial for a single env (no point paying for processes), subprocess for
    many. `backend` ('serial'|'subproc') overrides the default (e.g. to force
    serial for debugging)."""
    if backend is None:
        backend = "serial" if int(n_envs) <= 1 else "subproc"
    if backend == "serial":
        return SerialVecEnv(cfg, n_envs, base_seed)
    if backend == "subproc":
        return SubprocVecEnv(cfg, n_envs, base_seed)
    raise ValueError(f"unknown vec backend {backend!r} (serial|subproc).")
