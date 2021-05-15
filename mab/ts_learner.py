from mab.learner import *
import numpy as np


class TS_Learner(Learner):
  # Specialized Thompson Sampling Learner
  def __init__(self, arms,n_classes = 4):
    super().__init__(n_arms=len(arms),n_classes=n_classes)
    self.arms = np.array(arms)
    self.beta_parameters = np.array([np.ones([len(arms), 2]) for x in range(n_classes)])

  def pull_arm_per_class(self):
    idx = []
    for i in range(self.n_classes):
      idx.append(np.argmax(self.arms * np.random.beta(self.beta_parameters[i][:, 0], self.beta_parameters[i][:, 1])))
    return idx
  
  def select_fractions(self):
    fractions_idxs = []
    for i in range(4):
      idx = np.argmax(np.random.beta(self.beta_parameters[i][:,0], self.beta_parameters[i][:,1]))
      fractions_idxs.append(idx)
    return fractions_idxs

  def update(self, pulled_arms, reward, to_be_updated):
    #print(pulled_arm)
    for c in range(self.n_classes):
      if not to_be_updated[c] : continue
      self.update_observations(pulled_arms, reward, c)
      self.beta_parameters[c][pulled_arms[c], 0] = self.beta_parameters[c][pulled_arms[c], 0] + reward[c]
      self.beta_parameters[c][pulled_arms[c], 1] = self.beta_parameters[c][pulled_arms[c], 1] + 1 - reward[c] #TODO: this was originally 1.0, we need some refactor to include such scenario
    self.t += 1