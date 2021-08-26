from mab.learner import *

class TS_Learner(Learner):
  # Specialized Thompson Sampling Learner
  def __init__(self, arms, n_classes = 4):
    super().__init__(n_arms=len(arms),n_classes=n_classes)
    self.arms = np.array(arms)
    self.beta_parameters = np.ones((len(arms), 2))
  
  """def pull_arm(self):
    to_pull = self.arms*np.random.beta(self.beta_parameters_price[:, 0], self.beta_parameters_price[:, 1])
    idx = np.unravel_index(to_pull.argmax(), to_pull.shape)
    return idx"""

  def update(self, pulled_arm, reward):
    self.t += 1
    self.update_observations(pulled_arm, reward)
    self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
    self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1 - reward
    

  def pull_arm(self):
    idx = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1]))
    return idx