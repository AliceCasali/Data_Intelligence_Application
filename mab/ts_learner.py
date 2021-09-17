from mab.learner import *
from scipy.optimize import linear_sum_assignment

class TS_Learner(Learner):
  # Specialized Thompson Sampling Learner
  def __init__(self, n_arms, n_classes = 4):
    super().__init__(n_arms=n_arms,n_classes=n_classes)
    self.beta_parameters = np.ones((n_arms, 2))

  def update(self, pulled_arm, reward):
    self.t += 1
    self.update_observations(pulled_arm, reward)
    self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
    self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1 - reward
    

  def pull_arm(self):
    idx = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1]))
    return idx

  def pull_arm_price2(self, price1idx, arms):
    max_num = -1000
    max_ind = -1
    for i in range(self.n_arms):
      if arms[i][0] == price1idx:
        r = np.random.beta(self.beta_parameters[i,0], self.beta_parameters[i,1])
        if r > max_num:
          r = max_num
          max_ind = i
    return arms[max_ind][1]

  def pull_arm_matching(self, ec, ep, arms, p1idx=None, p2idx=None):
        graph = np.zeros((len(ec), len(ep)))
        for i in range(len(ec)):
            for j in range(len(ep)):
              arm_index = None
              if p1idx is not None and p2idx is not None:
                arm_index = arms.index((p1idx, p2idx, ec[i], ep[j]))
              elif p1idx is not None and p2idx:
                arm_index = arms.index((p1idx, ec[i], ep[j]))
              elif p2idx is not None and p1idx:
                arm_index = arms.index((p2idx, ec[i], ep[j]))
              else:
                arm_index = arms.index((ec[i], ep[j]))
              graph[i,j] = np.random.beta(self.beta_parameters[arm_index,0], self.beta_parameters[arm_index, 1])
        
        matched_c, matched_p = linear_sum_assignment(-graph)
        matched_tuples = [(ec[c], ep[p]) for c,p in zip(matched_c, matched_p)]

        return matched_tuples

  def pull_arm_all(self, ec, ep, arms, n_price1, n_price2):
    matched_tuples_list = []
    match_scores = []
    
    for k in range(n_price1):
      for l in range(n_price2):
        graph = np.zeros((len(ec), len(ep)))
        for i in range(len(ec)):
            for j in range(len(ep)):
              arm_index = arms.index((k, l , ec[i], ep[j]))
              graph[i,j] = np.random.beta(self.beta_parameters[arm_index,0], self.beta_parameters[arm_index, 1])
        
        matched_c, matched_p = linear_sum_assignment(-graph)
        matched_tuples = [(ec[c], ep[p]) for c,p in zip(matched_c, matched_p)]

        matched_tuples_list.append(matched_tuples)
        
        score = graph[matched_c, matched_p].sum()
        match_scores.append(score)

    return matched_tuples_list[np.argmax(match_scores)], np.argmax(match_scores)

  def reset(self):
    self.beta_parameters = np.ones((self.n_arms, 2))



