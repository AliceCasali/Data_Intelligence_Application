import numpy as np

class LearnerSW():

    def __init__(self, n_arms, n_classes=4, frame_size=30, days=365):
        self.n_arms = n_arms
        self.n_classes = n_classes
        self.t = 0
        self.days = days
        self.frame_size = frame_size
        self.collected_rewards = [[] for i in range(n_arms) for j in range(days)]
        self.rewards_per_arm = [[] for i in range(n_arms)]

    def update_observations(self, pulled_arm, reward, day):
        self.rewards_per_arm[pulled_arm].append(reward)
        #self.collected_rewards = np.append(self.collected_rewards, reward)
        self.collected_rewards[pulled_arm][day].append(reward)