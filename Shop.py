import numpy as np 
import itertools
from mab.ts_learner import *
from mab.ucb_learner import *
from mab.ts_learner_sw import *
from mab.ucb_learner_sw import *
from utilities import * 
from scipy.signal import savgol_filter


class Shop():
    def __init__(self):
        self.n_classes = 4
        self.prices1 = np.linspace(80,240, num=5)
        self.prices2 = np.linspace(100,300, num=5)
        self.discounts = np.array([0.0, 0.05, 0.10, 0.25])
        self.conv1 = np.array([generate_conversion_rate(self.prices1) for x in range(self.n_classes)]) # [class x price]
        self.conv2 = np.array([[generate_conversion_rate(self.prices2) for x in range(self.n_classes)] for y in range(len(self.discounts))]) # [promo x class x price]  

        self.window_size = 10
        self.threshold = 40
        self.detection_data = []
        self.detection_window = []    

    def get_clairvoyant_prices_and_matching(self, ec, ep, n_price1, n_price2):
        matched_tuples_list = []
        match_scores = []

        for k in range(n_price1):
            for l in range(n_price2):
                graph = np.zeros((len(ec), len(ep)))
                p1 = self.prices1[k]
                p2 = self.prices2[l]
                for i in range(len(ec)):
                    for j in range(len(ep)):
                        graph[i,j] = p1*self.conv1[ec[i], k] + self.conv1[ec[i], k]*p2*(1-self.discounts[ep[j]])*self.conv2[ep[j], ec[i], l]

                matched_c, matched_p = linear_sum_assignment(-graph)
                matched_tuples = [(ec[c], ep[p]) for c,p in zip(matched_c, matched_p)]

                matched_tuples_list.append(matched_tuples)

                score = graph[matched_c, matched_p].sum()
                match_scores.append(score)
        
        return matched_tuples_list[np.argmax(match_scores)], np.argmax(match_scores), np.max(match_scores), match_scores
 
    
    def get_clairvoyant_matching(self, ec, ep, pidx1, pidx2):
        p1 = self.prices1[pidx1]
        p2 = self.prices2[pidx2]
        graph = np.zeros((len(ec), len(ep)))
        for i in range(len(ec)):
            for j in range(len(ep)):
                graph[i,j] = p1*self.conv1[ec[i], pidx1] + self.conv1[ec[i], pidx1]*p2*(1-self.discounts[ep[j]])*self.conv2[ep[j], ec[i], pidx2]
        
        matched_c, matched_p = linear_sum_assignment(-graph) # individual matching
        matched_tuples = [(ec[c], ep[p]) for c,p in zip(matched_c, matched_p)] # (customer_class, promo_type)

        return matched_tuples

    def get_customer_list(self):
	    customers_enum = list(enumerate(self.customers))
	    customer_list = np.concatenate([np.ones(c).astype(int)*p for p,c in customers_enum])
	    return customer_list

    def get_promo_list(self, promo_portions):
        en_promos = self.customers.sum()*promo_portions
        en_promos = en_promos.astype(int)
        en_promos[0] += self.customers.sum() - en_promos.sum()

        expected_promos = list(enumerate(en_promos))
        expected_promos = np.concatenate([np.ones(c).astype(int)*p for p,c in expected_promos])
        return expected_promos

    def get_promo_fractions_from_tuples(self, matched_tuples):
        promo_fractions = np.zeros((self.n_classes, len(self.discounts)))

        for t in matched_tuples:
            promo_fractions[t[-2], t[-1]] += 1

        promo_fractions /= self.customers.reshape(4,1)
        return promo_fractions

    def set_expected_customers(self, customers):
        self.customers = customers
    
    def set_conv_rate(self, conv1, conv2):
        self.conv1 = conv1
        self.conv2 = conv2
    
    def set_price_learner (self, learner, n_arms):
        if learner == 'TS':
            self.price_learner = TS_Learner(n_arms=n_arms)
        elif learner == 'UCB':
            self.price_learner = UCB(n_arms=n_arms)
    
    def set_price2_learner (self, learner, n_arms):
        if learner == 'TS':
            self.price2_learner = TS_Learner(n_arms=n_arms)
        elif learner == 'UCB':
            self.price2_learner = UCB(n_arms=n_arms)

    def set_assignment_learner(self, learner, n_arms):
        if learner == 'TS':
            self.assignment_learner = TS_Learner(n_arms=n_arms)
        elif learner == 'UCB':
            self.assignment_learner = UCB(n_arms=n_arms)

    def set_price_learner_sw (self, learner, n_arms, frame_size=60, days=365):
        if learner == 'TS':
            self.price_learner_sw = TS_Learner_SW(n_arms=n_arms, frame_size=frame_size, days=days)
        elif learner == 'UCB':
            self.price_learner_sw = UCB_SW(n_arms=n_arms, frame_size=frame_size, days=days)
    
    def set_price2_learner_sw (self, learner, n_arms, frame_size=60, days=365):
        if learner == 'TS':
            self.price2_learner_sw = TS_Learner_SW(n_arms=n_arms, frame_size=frame_size, days=days)
        elif learner == 'UCB':
            self.price2_learner_sw = UCB_SW(n_arms=n_arms, frame_size=frame_size, days=days)

    def set_assignment_learner_sw(self, learner, n_arms, frame_size=60, days=365):
        if learner == 'TS':
            self.assignment_learner_sw = TS_Learner_SW(n_arms=n_arms, frame_size=frame_size, days=days)
        elif learner == 'UCB':
            self.assignment_learner_sw = UCB_SW(n_arms=n_arms, frame_size=frame_size, days=days)
    
    
    def best_promo_per_class(self, chosen_price1 = None, chosen_price2 = None):
        N = np.array([1,1,1,1])
        weights = np.zeros((4,4))

        if (chosen_price1 is None) and (chosen_price2 is None):
            perm_prices = list(itertools.product(self.prices1, self.prices2))
        elif (chosen_price1 is not None) and (chosen_price2 is None):
            perm_prices = list(itertools.product([self.prices1[chosen_price1]], self.prices2))
        elif (chosen_price1 is None) and (chosen_price2 is not None):
            perm_prices = list(itertools.product(self.prices1, [self.prices2[chosen_price2]]))
        else:
            perm_prices = [(chosen_price1, chosen_price2)]

        
        reward = 0

        # for each price
        for p in perm_prices:
            # for each class 
            for c in range(self.n_classes):
                # for each promos
                for j in range(len(self.discounts)):
                    #TODO we are assuming that we have 5 candidate prices
                    weights[c,j] = p[0]*self.conv1[c, index(self.prices1, p[0])] + (1-self.discounts[j])*p[1]*self.conv2[j, c, index(self.prices2, p[1])]

            col_ind = [np.argmax(row_class) for row_class in weights]
            row_ind = list(range(self.n_classes))
            promo_reward = weights[row_ind,col_ind]
            total_reward = np.sum(promo_reward)

            print(p)
            print(total_reward)
            print("*******")

            if (total_reward > reward):
                self.best_price, self.matched_promos, self.promo_rewards, reward = p, col_ind, promo_reward,  total_reward

    def print_coupons(self, shop_ts = None, shop_ucb = None, promo_fractions = None):
        if(shop_ts is None and shop_ucb is None and promo_fractions is None):
            enum_customers = list(enumerate(self.customers)) # [10, 23, 30, 54] --> [(0, 10), (1, 23), (2, 30), (3, 54)]
            self.customers_random = np.concatenate([np.ones(c)*p for p,c in enum_customers])
            enum_customers = [(self.matched_promos[i], cust) for i, cust in enum_customers] # [(0, 10), (1, 23), (2, 30), (3, 54)] --> [(3, 10), (1, 23), (2, 30), (0, 54)]
            self.coupons = np.concatenate([np.ones(c)*self.discounts[p] for p,c in enum_customers]) # [(3, 10), (1, 23), (2, 30), (0, 54)] --> [0,0,0,0, ... 0.05,0.05,0.05,0.05 ... ]
        else:
            clairvoyant_n_promos = (self.customers*promo_fractions + 0.99).astype(int)
            ts_n_promos = (shop_ts.customers*promo_fractions + 0.99).astype(int)
            ucb_n_promos = (shop_ucb.customers*promo_fractions + 0.99).astype(int)

            clairvoyant_promos = [list(enumerate(l)) for l in clairvoyant_n_promos]
            ts_promos = [list(enumerate(l)) for l in ts_n_promos]
            ucb_promos = [list(enumerate(l)) for l in ucb_n_promos]

            clairvoyant_promos = [np.concatenate([np.ones(c).astype(int)*p for p,c in promo]) for promo in clairvoyant_promos]
            ts_promos = [np.concatenate([np.ones(c).astype(int)*p for p,c in promo]) for promo in ts_promos]
            ucb_promos = [np.concatenate([np.ones(c).astype(int)*p for p,c in promo]) for promo in ucb_promos]

            for i in range(4):
                np.random.shuffle(clairvoyant_promos[i])
                np.random.shuffle(ts_promos[i])
                np.random.shuffle(ucb_promos[i])

            clairvoyant_promos = [list(promo) for promo in clairvoyant_promos]
            ts_promos = [list(promo) for promo in ts_promos]
            ucb_promos = [list(promo) for promo in ucb_promos]

            promos = clairvoyant_promos*2
            return promos
    
    def detect_phase_change(self, data):
        self.detection_data.append(data)

        if len(self.detection_data) < self.window_size:
            self.detection_window.append(data)
        else:
            self.detection_window.pop(0)
            self.detection_window.append(data)

            if len(self.detection_data) > self.threshold:
                data_mean = np.mean(savgol_filter(self.detection_data, 41, 3))
            else:
                data_mean = np.mean(self.detection_data)
            
            window_mean = np.mean(self.detection_window)

            if abs(window_mean - data_mean) > self.threshold:
                print("CHANGE DETECTED!")
                self.detection_window = []
                self.detection_data = []
                self.price_learner.reset()
                #self.price2_learner.reset()
 
        


