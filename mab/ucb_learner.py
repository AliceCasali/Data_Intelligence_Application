  
from mab.learner import Learner
import numpy as np

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