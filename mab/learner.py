import numpy as np

class Learner():
    # Generale Learner Class
    def __init__(self, n_arms, n_classes=4):
        self.n_arms = n_arms
        self.n_classes = n_classes
        self.t = 0
        self.collected_rewards = [[]for i in range(n_classes)]
        self.pulled_arms = [[] for i in range(n_classes)]
        self.rewards_per_arm = np.array([np.zeros(self.n_arms) for i in range(n_classes)])
        self.observed_customers = np.ones(n_classes)

    def update_observations(self, pulled_arm, reward, c): 
        self.rewards_per_arm[c, pulled_arm[1]] = reward
        self.pulled_arms[0].append(pulled_arm[1])
        self.collected_rewards[0].append(reward)
        self.observed_customers[c] += (self.observed_customers[c])/(sum(self.observed_customers))