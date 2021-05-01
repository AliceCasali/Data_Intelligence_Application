import numpy as np
import matplotlib.pyplot as plt
from environment import *
from ts_learner import *
from UCB_Learner import *



prices = [500, 690, 750, 850]



def generate_conversion_rate(prices):
    val = np.random.uniform(size=(len(prices)))
    conversion_rates = np.sort(val)[::-1]
    return conversion_rates

T = 300

conversion_rate_first_item = generate_conversion_rate(prices)
n_customers =  1000
 
max_conv = np.argmax(conversion_rate_first_item)

opt = conversion_rate_first_item[max_conv]
print(opt)

n_experiments = 200
ts_rewards_per_experiment = []
u_rewards_per_experiment = []

for e in range(0, n_experiments):
    env = Environment(n_arms=len(prices), probabilities = conversion_rate_first_item)
    ts_learner =TS_Learner(arms=prices)
    u_learner = UCB(arms=prices)
    for t in range(0,T):
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm,reward)

        pulled_arm = u_learner.pull_arm()
        reward = env.round(pulled_arm)
        u_learner.update(pulled_arm,reward)

    ts_rewards_per_experiment.append(ts_learner.rewards)
    u_rewards_per_experiment.append(u_learner.rewards)


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - u_rewards_per_experiment, axis=0)), 'g')
plt.legend(["TS", "UCB"])
plt.show()