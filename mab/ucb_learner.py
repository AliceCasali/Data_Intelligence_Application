  
from mab.learner import Learner
import numpy as np

class UCB(Learner):
    def __init__(self, n_arms,n_classes = 4):
        super().__init__(n_arms,n_classes = n_classes)
        self.empirical_means = np.zeros([n_arms])
        self.confidence = np.ones([n_arms])*np.inf

    def pull_arm_per_class(self):
        uppers = []
        for i in range(self.n_classes):
            uppers.append(self.empirical_means[:] + self.confidence[:])
        return [np.random.choice(np.where(x == x.max())[0]) for x in uppers]

    def update(self, pulled_arms, reward, to_be_updated):
        self.t += 1
        for c in range(self.n_classes):
            if not to_be_updated[c] : continue
            self.empirical_means[pulled_arms[c]] = (self.empirical_means[pulled_arms[c]]*(self.t-1) + reward[c])/self.t
            for a in range(self.n_arms):
                n_samples = len(self.rewards_per_arm[c])
                self.confidence[a] = (2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf
            self.update_observations(pulled_arms, reward, c)