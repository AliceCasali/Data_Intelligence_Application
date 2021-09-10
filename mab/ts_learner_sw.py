from mab.learner_sw import *
from scipy.optimize import linear_sum_assignment

class TS_Learner_SW(LearnerSW):
    def __init__(self, n_arms, n_classes = 4, frame_size=30, days=365):
        super().__init__(n_arms, n_classes, frame_size, days)
        self.beta_parameters = np.ones((n_arms, 2))

    def get_beta_parameters(self, day):
        bp = np.ones((self.n_arms, 2))
        start = np.max(0, day-self.frame_size)

        for i in range(self.n_arms):
            n_rew = 0
            sum = 0
            for j in range(start, day):
                n_rew += len(self.collected_rewards[i][j])
                sum += np.sum(self.collected_rewards[i][j])
            bp[i,0] += sum
            bp[i,1] += n_rew - sum
        
        return bp

    def pull_arm(self, day):
        bp = self.get_beta_parameters(day)
        idx = np.argmax(np.random.beta(bp[:,0], bp[:,1]))
        return idx

    def pull_arm_matching(self, ec, ep, arms, day, p1idx=None, p2idx=None):
        bp = self.get_beta_parameters(day)
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
              graph[i,j] = np.random.beta(bp[arm_index,0], bp[arm_index, 1])
        
        matched_c, matched_p = linear_sum_assignment(-graph)
        matched_tuples = [(ec[c], ep[p]) for c,p in zip(matched_c, matched_p)]

        return matched_tuples
    
    def pull_arm_all(self, ec, ep, arms, n_price1, n_price2, day):
        bp = self.get_beta_parameters(day)
        matched_tuples_list = []
        match_scores = []
    
        for k in range(n_price1):
            for l in range(n_price2):
                graph = np.zeros((len(ec), len(ep)))
                for i in range(len(ec)):
                    for j in range(len(ep)):
                        arm_index = arms.index((k, l , ec[i], ep[j]))
                        graph[i,j] = np.random.beta(bp[arm_index,0], bp[arm_index, 1])
        
                matched_c, matched_p = linear_sum_assignment(-graph)
                matched_tuples = [(ec[c], ep[p]) for c,p in zip(matched_c, matched_p)]

                matched_tuples_list.append(matched_tuples)
        
                score = graph[matched_c, matched_p].sum()
                match_scores.append(score)

        return matched_tuples_list[np.argmax(match_scores)], np.argmax(match_scores)
