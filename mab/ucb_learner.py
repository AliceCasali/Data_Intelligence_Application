  
from mab.learner import Learner
import numpy as np
import random

class UCB(Learner):
    def __init__(self, arms,n_classes = 4):
        super().__init__(n_arms=arms.shape[1],n_classes=n_classes)
        self.empirical_means = np.zeros([arms.shape[1]])
        self.confidence = np.ones([arms.shape[1]])*np.inf
        self.arms = arms

    def pull_arm(self, customers):
        upper = self.empirical_means[:] * self.arms * np.reshape(customers,(4,1)) + self.confidence[:]
        rows, cols = np.where(upper == upper.max()) 
        return random.choice(list(zip(rows,cols)))
    
    def pull_arm_unkown_cust(self):
        upper = self.empirical_means[:] * self.arms * np.reshape(self.observed_customers, (4,1)) + self.confidence[:]
        rows, cols = np.where(upper == upper.max())
        return random.choice(list(zip(rows,cols)))

    def update(self, pulled_arm, reward, c):
        self.t += 1
        self.empirical_means[pulled_arm[1]] = (self.empirical_means[pulled_arm[1]]*(self.t-1) + reward)/self.t
        n_samples = len(self.rewards_per_arm[c])
        self.confidence[pulled_arm[1]] = (2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf
        self.update_observations(pulled_arm, reward, c)