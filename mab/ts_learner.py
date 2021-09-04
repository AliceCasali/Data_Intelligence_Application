from mab.learner import *
from scipy.optimize import linear_sum_assignment

class TS_Learner(Learner):
  # Specialized Thompson Sampling Learner
  def __init__(self, n_arms, n_classes = 4):
    super().__init__(n_arms=n_arms,n_classes=n_classes)
    self.beta_parameters = np.ones((n_arms, 2))
  
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

  def pull_arm_matching(self, ec, ep, arms):
        graph = np.zeros((len(ec), len(ep)))
        for i in range(len(ec)):
            for j in range(len(ep)):
                arm_index = arms.index((ec[i], ep[j]))
                graph[i,j] = np.random.beta(self.beta_parameters[arm_index,0], self.beta_parameters[arm_index, 1])
        
        matched_c, matched_p = linear_sum_assignment(-graph)
        matched_tuples = [(ec[c], ep[p]) for c,p in zip(matched_c, matched_p)]

        return matched_tuples