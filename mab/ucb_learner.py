  
from mab.learner import Learner
import numpy as np
from scipy.optimize import linear_sum_assignment

class UCB(Learner):
    def __init__(self, n_arms,n_classes = 4):
        super().__init__(n_arms,n_classes = n_classes)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf]*n_arms)

    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])#, self.empirical_means, self.confidence, upper_conf

    def update(self, pull_arm,reward):
        self.t += 1
        self.empirical_means[pull_arm] = (self.empirical_means[pull_arm]*(self.t-1) + reward)/self.t
        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf
        self.update_observations(pull_arm, reward)
        return self.confidence

    def pull_arm_matching(self, ec, ep, arms):
        graph = np.zeros((len(ec), len(ep)))
        for i in range(len(ec)):
            for j in range(len(ep)):
                arm_index = arms.index((ec[i], ep[j]))
                graph[i,j] = self.confidence[arm_index] + self.empirical_means[arm_index]
                if graph[i,j] == np.inf:
                    graph[i,j] = 1e3
        
        matched_c, matched_p = linear_sum_assignment(-graph)
        matched_tuples = [(ec[c], ep[p]) for c,p in zip(matched_c, matched_p)]

        return matched_tuples