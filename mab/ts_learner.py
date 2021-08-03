from mab.learner import *
import numpy as np


class TS_Learner(Learner):
  # Specialized Thompson Sampling Learner
  def __init__(self, arms, arm_conv_rate2 = np.array([]), n_classes = 4):
    super().__init__(n_arms=arms.shape[1],n_classes=n_classes)
    self.arms = np.array(arms)
    self.arm_conv_rate2 = np.array(arm_conv_rate2)
    self.beta_parameters_price = np.array([np.ones(2) for x in range(arms.shape[1])])
    self.beta_parameters_conv2 = np.array([[np.ones(2) for x in range(arm_conv_rate2.shape[0])] for x in range(n_classes)]) #aggiornare la classe guessata (potrebbe aumentare il tempo di convergenza)
  
  def pull_arm(self, customers):
    to_pull = self.arms * np.reshape(customers/np.max(customers), (4,1)) * np.random.beta(self.beta_parameters_price[:, 0], self.beta_parameters_price[:, 1])
    idx = np.unravel_index(to_pull.argmax(), to_pull.shape)
    return idx
  
  def pull_arm_unkown_cust(self): 
    to_pull = self.arms *  np.random.beta(self.beta_parameters_price[:, 0], self.beta_parameters_price[:, 1]) * np.reshape(self.observed_customers, (4,1))
    idx = np.unravel_index(to_pull.argmax(), to_pull.shape)
    return idx
  
  def pull_conv_2(self): # TODO: pulla 4 conv_2 per giorno per ogni classe e aggiornare beta_params_conv2 
    to_pull = [np.argmax(self.arm_conv_rate2 * np.random.beta(self.beta_parameters_conv2[c][:, 0], self.beta_parameters_conv2[c][:, 1])) for c in range(self.n_classes)] 
    return np.reshape(to_pull, (4,1))
  
  def select_fractions(self):
    fractions_idxs = []
    for i in range(self.n_classes):
      idx = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1]))
      fractions_idxs.append(idx)
    return fractions_idxs

  def update(self, pulled_arm, reward, c):
    self.update_observations(pulled_arm, reward, c)
    self.beta_parameters_price[pulled_arm[1], 0] = self.beta_parameters_price[pulled_arm[1], 0] + reward
    self.beta_parameters_price[pulled_arm[1], 1] = self.beta_parameters_price[pulled_arm[1], 1] + 1 - reward
    self.t += 1
    if len(self.arm_conv_rate2) != 0:
      self.beta_parameters_conv2[c, pulled_arm[1], 0] = self.beta_parameters_conv2[c, pulled_arm[1], 0] + reward
      self.beta_parameters_conv2[c, pulled_arm[1], 1] = self.beta_parameters_conv2[c, pulled_arm[1], 1] + 1 - reward