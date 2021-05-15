from mab.learner import *
import numpy as np


class TS_Learner(Learner):
  # Specialized Thompson Sampling Learner
  def __init__(self, arms,n_classes = 4):
    super().__init__(n_arms=len(arms),n_classes=n_classes)
    self.arms = arms
    self.beta_parameters = [np.ones([len(arms), 2]) for x in range(n_classes)]

  def pull_arm(self):
    for i in range(4):
      idx = np.argmax(np.array(self.arms) * np.random.beta(self.beta_parameters[i][:, 0], self.beta_parameters[i][:, 1]))
    return idx
  
  def select_fractions(self):
    fractions_idxs = []
    for i in range(4):
      idx = np.argmax(np.random.beta(self.beta_parameters[i][:,0], self.beta_parameters[i][:,1]))
      fractions_idxs.append(idx)
    return fractions_idxs

  def update(self, pulled_arm, reward, cust_class):
    #print(pulled_arm)
    self.update_observations(pulled_arm, reward, cust_class)
    self.beta_parameters[cust_class][pulled_arm, 0] = self.beta_parameters[cust_class][pulled_arm, 0] + reward
    self.beta_parameters[cust_class][pulled_arm, 1] = self.beta_parameters[cust_class][pulled_arm, 1] + 650 - reward #TODO: this was originally 1.0, we need some refactor to include such scenario
    self.t += 1