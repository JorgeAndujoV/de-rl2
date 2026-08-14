"""Population-parallel episode evaluation for Sep-CMA-ES (EXP026) -- additive.

Mirrors vec_env.py's spawn+pipe design, but each worker owns a DEEnv AND a
policy net: it receives a flat weight vector, runs ONE full greedy episode
(reusing evaluate.run_episode -- the exact episode logic every agent is
evaluated with), and returns that episode's final best-error. The whole forward
pass + episode runs in the worker (TensorFlow only, one thread each); the main
process -- which runs JAX/evosax -- never does a TF forward in the hot loop.

Worker count is configurable and INDEPENDENT of population_size: a generation's
`population_size` members are dispatched to `n_workers` workers in synchronous
batches, so a small validation run and the full run share this code unchanged.
This does NOT edit vec_env.py; it is a sibling module reusing its pattern.
"""

import multiprocessing as mp

import numpy as np


def _worker(conn, cfg, obs_dim, n_actions, param_dim, policy_hidden, seed):
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
            pass                      # threads already initialised; harmless
        from derl2.environments.env import DEEnv
        from derl2.evaluation.evaluate import run_episode
        from derl2.agents.sep_cmaes import build_policy_net, set_flat

        env = DEEnv.from_config(cfg)
        agent = build_policy_net(obs_dim, n_actions, param_dim, policy_hidden,
                                 seed=seed)
        variables = agent.policy_net.trainable_variables
        conn.send(("ready", None))
        while True:
            cmd, data = conn.recv()
            if cmd == "eval":
                flat, ep_seed, fid = data
                set_flat(variables, flat)
                # agent.act is the deterministic decode (argmax + Beta mean);
                # run_episode drives env.reset/step exactly as in evaluation.
                row, _steps, _traj = run_episode(env, ep_seed, fid, agent.act)
                conn.send(("ok", float(row["final_error"])))
            elif cmd == "close":
                conn.close()
                return
    except Exception:
        import traceback
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass


class ESWorkerPool:
    """N spawn workers, each a DEEnv + policy net, evaluating weight vectors as
    full episodes. `evaluate` fans a population out across the workers."""

    def __init__(self, cfg, n_workers, obs_dim, n_actions, param_dim,
                 policy_hidden, seed=0):
        self.n_workers = int(n_workers)
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i in range(self.n_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=_worker,
                args=(child, cfg, obs_dim, n_actions, param_dim,
                      tuple(policy_hidden), seed + i),
                daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        for c in self.conns:          # wait for each worker to build env + net
            self._recv(c)

    @staticmethod
    def _recv(conn):
        tag, payload = conn.recv()
        if tag == "error":
            raise RuntimeError(f"ES worker failed:\n{payload}")
        return payload

    def evaluate(self, flat_pop, ep_seed, function_id):
        """flat_pop: (P, D). Returns final_error (P,).

        Each member is one full greedy episode on `function_id` seeded with the
        SAME `ep_seed` (common random numbers within a generation -> members are
        ranked on the same problem instance; the loop advances the seed across
        generations). Dispatched in synchronous batches of n_workers.
        """
        P = len(flat_pop)
        results = [None] * P
        idx = 0
        while idx < P:
            batch = list(range(idx, min(idx + self.n_workers, P)))
            for w, m in enumerate(batch):
                self.conns[w].send(("eval", (flat_pop[m], int(ep_seed),
                                             int(function_id))))
            for w, m in enumerate(batch):
                results[m] = self._recv(self.conns[w])
            idx += len(batch)
        return np.asarray(results, dtype=np.float64)

    def evaluate_seeds(self, flat, seeds, function_id):
        """Evaluate ONE weight vector `flat` across a LIST of `seeds`, in parallel
        batches. Returns final_error (len(seeds),) aligned to `seeds`.

        Used for the periodic mean-policy monitor: the current Sep-CMA-ES
        distribution mean is scored on a small fixed validation seed set (distinct
        from the per-generation training seeds and from the final evaluation
        seeds) to track a real learning curve, without disturbing the population
        loop. Same weights every episode, different seed each -> a genuine
        multi-seed estimate of the mean policy's quality."""
        seeds = [int(s) for s in seeds]
        n = len(seeds)
        results = [None] * n
        idx = 0
        while idx < n:
            batch = list(range(idx, min(idx + self.n_workers, n)))
            for w, j in enumerate(batch):
                self.conns[w].send(("eval", (flat, seeds[j], int(function_id))))
            for w, j in enumerate(batch):
                results[j] = self._recv(self.conns[w])
            idx += len(batch)
        return np.asarray(results, dtype=np.float64)

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
