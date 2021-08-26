import numpy as np

class Learner():
    # Generale Learner Class
    def __init__(self, n_arms, n_classes=4):
        self.n_arms = n_arms
        self.n_classes = n_classes
        self.t = 0
        self.collected_rewards = np.array([])
        #self.pulled_arms = [[] for i in range(n_classes)]
        self.rewards_per_arm = [[] for i in range(n_arms)]
        #self.observed_customers = np.ones(n_classes)

    def update_observations(self, pulled_arm, reward): 
        self.rewards_per_arm[pulled_arm].append(reward)
        #self.pulled_arms[c].append(pulled_arm)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        #self.observed_customers[c] += (self.observed_customers[c])/(sum(self.observed_customers))