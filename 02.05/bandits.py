#!/usr/bin/env python3

import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

from random import Random
from tqdm import trange


class Env:
    def __init__(self, k, sigma_0=1, sigma=1, sigma_inc=0.01, rng=None):
        """
            k (Int): number of bandits
            sigma_0 (Float): std of the normal distribution used
                to generate the initial values of the bandits
            sigma (Float): std of the normal distribution from which
                the reward of the bandits is sampled
            sigma_inc (Float): std of the normal distribution of the
                increment that is added to the avg value of each bandit
                at each step
            rng (NoneType or Random): rng to use internally
        """
        self._rng = rng or Random()
        self._sigma = sigma
        self._sigma_inc = sigma_inc
        self._mu = [self._rng.gauss(0,sigma_0) for _ in range(k)]
        self._mu_orig = self._mu[:]

    def get_k(self):
        return len(self._mu)

    def step(self, a):
        is_optimal = self._mu[a] == max(self._mu)
        r = self._rng.gauss(self._mu[a], self._sigma)
        # add increment to the average value of all the bandits to
        # introduce non-stationarity
        for idx in range(len(self._mu)):
            self._mu[idx] += self._rng.gauss(0, self._sigma_inc)
        return r, is_optimal

    def reset(self):
        self._mu = self._mu_orig.copy()


class EpsilonGreedyAgent:
    def __init__(self, env, eps, rng=None):
        self._n = [0]*env.get_k()
        self._q = [0]*env.get_k()
        self._stats = {"steps": 0, "acc_reward": 0, "avg_reward": 0, "optimal_hit_rate": 0}
        self._env = env
        self._eps = eps
        self._rng = rng or Random()

    def select(self):
        rand = self._rng.random()
        if rand < self._eps:
            selected = self._rng.randint(0, self._env.get_k()-1)
        else:
            selected = max(range(self._env.get_k()),
                    key=self._q.__getitem__)
        return selected

    def spin_once(self):
        selected = self.select()
        r, is_optimal = self._env.step(selected)
        # update q and n
        q = self._q[selected]
        n = self._n[selected]
        q_upd = q + (r - q)/(n+1)
        self._q[selected] = q_upd
        self._n[selected] = n + 1
        # update optimal_hit_rate
        steps = self._stats["steps"]
        opt_rate = self._stats["optimal_hit_rate"]
        opt_rate_upd = opt_rate + (is_optimal - opt_rate)/(steps+1)
        self._stats["optimal_hit_rate"] = opt_rate_upd
        # update the rest of stats
        self._stats["steps"] += 1
        self._stats["acc_reward"] += r
        self._stats["avg_reward"] = self._stats["acc_reward"]/self._stats["steps"]

    def spin(self, niter=10000):
        for _ in range(niter):
            self.spin_once()

    def get_stats(self):
        return self._stats


NRUNS = 2000
NSTEPS = 10000

rng = Random(42)

available_eps = [0, 0.01, 0.1]

steps = list(range(NSTEPS+1))
avg_reward = [[0]*(NSTEPS+1) for _ in available_eps]
avg_optimal_rate = [[0]*(NSTEPS+1) for _ in available_eps]

for run in trange(NRUNS):
    env = Env(10, sigma_inc=0.01, rng=rng)
    for agent_idx, eps in enumerate([0, 0.01, 0.1]):
        agent = EpsilonGreedyAgent(env, eps, rng=rng)
        for step in range(NSTEPS+1):
            stats = agent.get_stats()
            avg_reward[agent_idx][step] += stats["avg_reward"]/NRUNS
            avg_optimal_rate[agent_idx][step] += stats["optimal_hit_rate"]/NRUNS
            agent.spin_once()
        env.reset()

plt.subplot(121)
plt.plot(steps, avg_reward[0])
plt.plot(steps, avg_reward[1])
plt.plot(steps, avg_reward[2])
plt.legend(("greedy", "eps=0.01", "eps=0.1"))
plt.xlabel("#steps")
plt.ylabel("Avg. reward")

plt.subplot(122)
plt.plot(steps, avg_optimal_rate[0])
plt.plot(steps, avg_optimal_rate[1])
plt.plot(steps, avg_optimal_rate[2])
plt.legend(("greedy", "eps=0.01", "eps=0.1"))
plt.xlabel("#steps")
plt.ylabel("Optimal hit rate")

plt.tight_layout()

plt.show()

