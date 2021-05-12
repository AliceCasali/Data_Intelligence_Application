import numpy as np

class Learner():
    # Generale Learner Class
    def __init__(self, n_arms, n_classes=4):
        self.n_arms = n_arms
        self.t = 0

        self.collected_rewards = [np.array([]) for i in range(self.n_classes)]
        self.pulled_arms = [np.array([]) for i in range(self.n_classes)]
        self.rewards_per_arm = [[] for i in range(self.n_arms)]

    def update_observations(self, pulled_arm, reward, cust_class):
        self.rewards_per_arm[cust_class][pulled_arm].append(reward)
        self.pulled_arms[cust_class] = np.append(self.pulled_arms[cust_class], pulled_arm)
        self.collected_rewards[cust_class] = np.append(self.collected_rewards[cust_class], reward)