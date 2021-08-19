import numpy as np 
import itertools
from mab.ts_learner import *
from mab.ucb_learner import *
from utilities import * 


class Shop():
    def __init__(self):
        self.n_classes = 4
        self.prices1 = np.cumsum(np.linspace(80,240, num=5))
        self.prices2 = np.cumsum(np.linspace(100,300, num=5))
        self.discounts = np.array([0.0, 0.05, 0.10, 0.25])
        self.conv1 = np.array([generate_conversion_rate(self.prices1) for x in range(self.n_classes)]) # [class x price]
        self.conv2 = np.array([[generate_conversion_rate(self.prices2) for x in range(self.n_classes)] for y in range(len(self.discounts))]) # [class x promo x price]      
        
    def set_expected_customers(self, customers):
        self.customers = customers
    
    def set_conv_rate(self, conv1, conv2):
        self.conv1 = conv1
        self.conv2 = conv2
    
    def set_price_learner (self, learner):
        if learner == 'TS':
            self.price_learner = TS_Learner(arms=self.prices1)
        elif learner == 'UCB':
            self.price_learner = UCB(n_arms=len(self.prices1))
    
    def best_promo_per_class(self, chosen_price1 = None, chosen_price2 = None):
        N = np.array([1,1,1,1])
        weights = np.zeros((4,4))

        if (chosen_price1 is None) and (chosen_price2 is None):
            perm_prices = list(itertools.product(self.prices1, self.prices2))
        elif (chosen_price1 is not None) and (chosen_price2 is None):
            perm_prices = list(itertools.product([self.prices1[chosen_price1]], self.prices2))
        else:
            perm_prices = [(chosen_price1, chosen_price2)]

        reward = 0

        # for each price
        for p in perm_prices:
            # for each class 
            for c in range(self.n_classes):
                # for each promos
                for j in range(len(self.discounts)):
                    weights[c,j] = p[0]*self.conv1[c, index(self.prices1, p[0])] + (1-self.discounts[j])*p[1]*self.conv2[c,j,index(self.prices2, p[1])]

            col_ind = [np.argmax(row_class) for row_class in weights]
            row_ind = list(range(self.n_classes))
            promo_reward = weights[row_ind,col_ind]
            total_reward = np.sum(promo_reward)

            if (total_reward > reward):
                self.best_price, self.matched_promos, self.promo_rewards, reward = p, col_ind, promo_reward,  total_reward
    
    def print_coupons(self):
        enum_customers = list(enumerate(self.customers)) # [10, 23, 30, 54] --> [(0, 10), (1, 23), (2, 30), (3, 54)]
        self.customers_random = np.concatenate([np.ones(c)*p for p,c in enum_customers])
        enum_customers = [(self.matched_promos[i], cust) for i, cust in enum_customers] # [(0, 10), (1, 23), (2, 30), (3, 54)] --> [(3, 10), (1, 23), (2, 30), (0, 54)]
        self.coupons = np.concatenate([np.ones(c)*self.discounts[p] for p,c in enum_customers]) # [(3, 10), (1, 23), (2, 30), (0, 54)] --> [0,0,0,0, ... 0.05,0.05,0.05,0.05 ... ]
        

