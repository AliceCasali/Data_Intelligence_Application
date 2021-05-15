  
from mab.learner import Learner
import numpy as np

class UCB(Learner):
    def __init__(self, n_arms,n_classes = 4):
        super().__init__(n_arms,n_classes = n_classes)
        self.empirical_means = np.zeros([n_classes, n_arms])
        self.confidence = np.ones([n_classes,n_arms])*np.inf

    def pull_arm(self):
        for i in range(self.n_classes):
            upper_conf = self.empirical_means[i,:] + self.confidence[i,:]
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

    def update(self, pull_arm,reward,cust_class):
        self.t += 1
        self.empirical_means[cust_class][pull_arm] = (self.empirical_means[cust_class][pull_arm]*(self.t-1) + reward)/self.t
        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[cust_class])
            self.confidence[cust_class][a] = (2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf
        self.update_observations(pull_arm, reward,cust_class)