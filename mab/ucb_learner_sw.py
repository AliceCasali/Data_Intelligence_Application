from mab.learner_sw import LearnerSW
import numpy as np
from scipy.optimize import linear_sum_assignment

class UCB_SW(LearnerSW):
    def __init__(self, n_arms, n_classes=4, frame_size=30, days=365):
        super().__init__(n_arms, n_classes, frame_size, days)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf]*n_arms)

    def get_empirical_means(self, day):
        em = np.zeros(self.n_arms)
        start = np.max(0, day-self.frame_size)
        for i in range(self.n_arms):
            n_rew = 0
            sum = 0
            for j in range(start, day):
                n_rew += len(self.collected_rewards[i][j])
                sum += np.sum(self.collected_rewards[i][j])
            em[i] = sum / n_rew
        return em

    def get_confidences(self, day):
        c = np.zeros(self.n_arms)
        t_rew = 0
        n_rew = np.zeros(self.n_arms)
        start = np.max(0, day-self.frame_size)
        for i in range(self.n_arms):
            for j in range(start, day):
                t_rew += len(self.collected_rewards[i][j])
                n_rew[i] += len(self.collected_rewards[i][j])
        
        for i in range(self.n_arms):
            c[i] = (2*np.log(t_rew)/n_rew[i])**0.5 if n_rew[i] > 0 else np.inf
        return c
    
    def pull_arm(self, day):
        upper_conf = self.get_empirical_means(day) + self.get_confidences(day)
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0]) 

    def update(self, pull_arm, reward, day):
        self.t += 1
        self.update_observations(pull_arm, reward, day)

    def pull_arm_matching(self, ec, ep, arms, day, p1idx=None, p2idx=None):
        c = self.get_confidences(day)
        em = self.get_empirical_means(day)
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
                graph[i,j] = em[arm_index] + c[arm_index]
                if graph[i,j] == np.inf:
                    graph[i,j] = 1e3
        
        matched_c, matched_p = linear_sum_assignment(-graph)
        matched_tuples = [(ec[c], ep[p]) for c,p in zip(matched_c, matched_p)]

        return matched_tuples

    def pull_arm_all(self, ec, ep, arms, n_price1, n_price2, day):
        c = self.get_confidences(day)
        em = self.get_empirical_means(day)

        matched_tuples_list = []
        match_scores = []
    
        for k in range(n_price1):
            for l in range(n_price2):
                graph = np.zeros((len(ec), len(ep)))
                for i in range(len(ec)):
                    for j in range(len(ep)):
                        arm_index = arms.index((k, l , ec[i], ep[j]))
                        graph[i,j] = em[arm_index] + c[arm_index]
        
                matched_c, matched_p = linear_sum_assignment(-graph)
                matched_tuples = [(ec[c], ep[p]) for c,p in zip(matched_c, matched_p)]

                matched_tuples_list.append(matched_tuples)
        
                score = graph[matched_c, matched_p].sum()
                match_scores.append(score)

        return matched_tuples_list[np.argmax(match_scores)], np.argmax(match_scores)