import numpy as np

class Learner():
    # Generale Learner Class
    def __init__(self, arms):
        self.n_arms = len(arms)
        self.arms = arms
        self.t = 0
        self.rewards = []
        self.rewards_per_arm = [[] for i in range(self.n_arms)]
        self.pulled_arms = []

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.pulled_arms.append(pulled_arm)
        self.rewards.append(reward)