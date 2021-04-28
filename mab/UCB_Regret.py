from scipy.optimize import linear_sum_assignment
from UCB_Matching import UCB_Matching
from CUSUM_UCB_Matching import CUSUM_UCB_Matching
from environment import Environment
import matplotlib.pyplot as plt
import numpy as np

p0 = np.array([[1/4, 1 , 1/4], [1/2, 1/4, 1/4], [1/4, 1/4, 1]])
p1 = np.array([[1, 1/4, 1/4], [1/2, 1/4, 1/4], [1/4, 1/4, 1]])
p2 = np.array([[1, 1/4, 1/4], [1/2, 1, 1/4], [1/4, 1/4, 1]])
p = [p0, p1, p2]
T = 3000
n_exp = 10
regret_UCB = np.zeros((n_exp, T))
regret_CUSUM = np.zeros((n_exp, T))
detections = [[] for _ in range(n_exp)]
M = 100
eps = 0.1
h = np.log(T)*2

for j in range(n_exp):
    # we should implement the non stationary env in our own 'cause is not given in the lecture
    # e_UCB = Non_Stationary_Environment(p0, size, P, T)
    # e_CD = Non_Stationary_Environment(p0, size, P, T)
    learner_UCB = UCB_Matching(p0.size, *p0.shape)
    learner_CD = CUSUM_UCB_Matching(p0.size, *p0.shape, M, eps, h)
    opt_rew = []
    rew_CD = []
    rew_UCB = []
    for t in range(T):
        p = P[int(t / e_UCB.phase_size)]
        opt = linear_sum_assignment(-p)
        opt_rew.append(p[opt].sum())

        pulled_arm = learner_CD.pull_arm()
        reward = e_CD.round(pulled_arm)
        learner_CD.update(pulled_arm)
        rew_CD.append(reward.sum())

        pulled_arm = learner_UCB.pull_arm()
        reward = e_UCB.round(pulled_arm)
        learner_UCB.update(pulled_arm)
        rew_UCB.append(reward.sum())
    regret_CUSUM[j, :] = np.cumsum(opt_rew) - np.cumsum(rew_CD)
    regret_UCB[j, :] = np.cumsum(opt_rew) - np.cumsum(rew_UCB)

plt.figure(0)
plt.ylabel('Regret')
plt.xlabel('t')
plt.plot(regret_CUSUM.mean(axis = 0))
plt.plot(regret_ucb.mean(axis = 0))
plt.legend(['CD-UCB', 'UCB'])
plt.show