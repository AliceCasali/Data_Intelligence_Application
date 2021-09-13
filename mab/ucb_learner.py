  
from mab.learner import Learner
import numpy as np
from scipy.optimize import linear_sum_assignment

class UCB(Learner):
    def __init__(self, n_arms,n_classes = 4):
        super().__init__(n_arms,n_classes = n_classes)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf]*n_arms)

    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])#, self.empirical_means, self.confidence, upper_conf

    def update(self, pull_arm,reward):
        self.t += 1
        self.empirical_means[pull_arm] = (self.empirical_means[pull_arm]*(self.t-1) + reward)/self.t
        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf
        self.update_observations(pull_arm, reward)
        return self.confidence

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
                graph[i,j] = self.confidence[arm_index] + self.empirical_means[arm_index]
                if graph[i,j] == np.inf:
                    graph[i,j] = 1e3
        
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
                        graph[i,j] = self.empirical_means[arm_index] + self.confidence[arm_index]
        
                matched_c, matched_p = linear_sum_assignment(-graph)
                matched_tuples = [(ec[c], ep[p]) for c,p in zip(matched_c, matched_p)]

                matched_tuples_list.append(matched_tuples)
        
                score = graph[matched_c, matched_p].sum()
                match_scores.append(score)

        return matched_tuples_list[np.argmax(match_scores)], np.argmax(match_scores)

