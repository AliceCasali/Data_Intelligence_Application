import numpy as np
import matplotlib.pyplot as plt
from mab.environment import *
from mab.ts_learner import *
from mab.greedy_learner import *

n_arms = 4
p = np.array([ 0.15, 0.1, 0.1, 0.35])
opt = p[3]

T = 300

n_experiments = 1000
ts_rewards_per_experiment = []
gr_rewards_per_experiment = []

for e in range(0, n_experiments):
    env = Environment(n_arms = n_arms, probabilities = p)
    ts_learner = TS_Learner(n_arms = n_arms)
    gr_learner = GreadyLearner(n_arms=n_arms)
    for t in range(0,T):
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm,reward)

        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm,reward)

    ts_rewards_per_experiment.append(ts_rewards_per_experiment)
    gr_rewards_per_experiment.append(gr_rewards_per_experiment)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment, axis=0)), 'g')
plt.legend(["TS", "Greedy"])
plt.show()


