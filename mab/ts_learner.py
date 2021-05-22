from mab.learner import *
import numpy as np


class TS_Learner(Learner):
  # Specialized Thompson Sampling Learner
  def __init__(self, arms,n_classes = 4):
    super().__init__(n_arms=arms.shape[1],n_classes=n_classes)
    self.arms = np.array(arms)
    self.beta_parameters = np.array([np.ones(2) for x in range(arms.shape[1])])
  
  def pull_arm(self, customers):
    to_pull = self.arms * np.reshape(customers, (4,1)) * np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
    idx = np.unravel_index(to_pull.argmax(), to_pull.shape)
    return idx
  
  def pull_arm2(self): #TODO for step4: this function does not take into account the number of customers per class
    to_pull = self.arms * np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
    idx = np.unravel_index(to_pull.argmax(), to_pull.shape)
    return idx
  
  def select_fractions(self):
    fractions_idxs = []
    for i in range(self.n_classes):
      idx = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1]))
      fractions_idxs.append(idx)
    return fractions_idxs

  def update(self, pulled_arm, reward):
    self.update_observations(pulled_arm, reward)
    self.beta_parameters[pulled_arm[1], 0] = self.beta_parameters[pulled_arm[1], 0] + reward
    self.beta_parameters[pulled_arm[1], 1] = self.beta_parameters[pulled_arm[1], 1] + 1 - reward
    self.t += 1