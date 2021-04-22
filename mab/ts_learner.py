from learner import *


class TS_Learner(Learner):
    # Specialized Thompson Sampling Learner
    def __init__(self, arms):
        super().__init__(arms=arms)
        self.beta_param = np.ones((self.n_arms, 2))

    def pull_arm(self):
        idx = np.argmax(np.array(self.arms) * np.random.beta(self.beta_param[:, 0], self.beta_param[:, 1]))
        return idx

    def update(self, pulled_arm, reward):
        self.update_observations(pulled_arm, reward)
        self.beta_param[pulled_arm, 0] = self.beta_param[pulled_arm, 0] + reward
        self.beta_param[pulled_arm, 1] = self.beta_param[pulled_arm, 1] + 1.0 - reward
        self.t += 1