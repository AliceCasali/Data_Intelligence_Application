import numpy as np
import matplotlib.pyplot as plt
from environment import *
from ts_learner import *
from greedy_learner import *

p = np.array([ 0.15, 0.1, 0.1, 0.35])
prices = [500, 690, 750, 850]
opt = p[3]

T = 300

n_experiments = 200
ts_rewards_per_experiment = []
gr_rewards_per_experiment = []

for e in range(0, n_experiments):
    env = Environment(n_arms=len(prices), probabilities = p)
    ts_learner = TS_Learner(arms=prices)
    gr_learner = Gready_Learner(arms=prices)
    for t in range(0,T):
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm,reward)

        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm,reward)

    ts_rewards_per_experiment.append(ts_learner.rewards)
    gr_rewards_per_experiment.append(gr_learner.rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment, axis=0)), 'g')
plt.legend(["TS", "Greedy"])
plt.show()