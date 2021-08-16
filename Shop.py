import utilities
import numpy as np 
import itertools


class Shop():
    def __init__(self):
        self.n_classes = 4
        self.prices1 = np.cumsum(np.linspace(80,240, num=5))
        self.prices2 = np.cumsum(np.linspace(100,300, num=5))
        self.discounts = np.array([0.0, 0.05, 0.10, 0.25])
        self.conv1 = np.array([utilities.generate_conversion_rate(self.prices1) for x in range(self.n_classes)]) # [class x price]
        self.conv2 = np.array([[utilities.generate_conversion_rate(self.prices2) for x in range(self.n_classes)] for y in range(len(self.discounts))]) # [class x promo x price]      
    
    def set_expected_customers(self, customers):
        self.customers = customers
    
    def set_conv_rate(self, conv1, conv2):
        self.conv1 = conv1
        self.conv2 = conv2
    
    def best_promo_per_class(self):
        N = np.array([1,1,1,1])
        weights = np.zeros((4,4))
        perm_prices = list(itertools.product(self.prices1, self.prices2))

        reward = 0

        # for each price
        for p in perm_prices:
            # for each class 
            for c in range(self.n_classes):
                # for each promos
                for j in range(len(self.discounts)):
                    weights[c,j] = p[0]*self.conv1[c, utilities.index(self.prices1, p[0])] + (1-self.discounts[j])*p[1]*self.conv2[c,j,utilities.index(self.prices2, p[1])]

            col_ind = [np.argmax(row_class) for row_class in weights]
            row_ind = list(range(self.n_classes))
            promo_reward = weights[row_ind,col_ind]
            total_reward = np.sum(promo_reward)

            #print(weights)
            #print(col_ind)
            #print(promo_reward)
            #print(total_reward)
            if (total_reward > reward):
                self.best_price, self.matched_promos, self.promo_rewards, reward = p, col_ind, promo_reward,  total_reward
        #print("=============== \n")
        #print(self.best_price)
        #print(self.matched_promos) # Columns of the weight matrix
    
    def print_coupons(self):
        enum_customers = list(enumerate(self.customers)) # [10, 23, 30, 54] --> [(0, 10), (1, 23), (2, 30), (3, 54)]
        enum_customers = [(self.matched_promos[i], cust) for i, cust in enum_customers] # [(0, 10), (1, 23), (2, 30), (3, 54)] --> [(3, 10), (1, 23), (2, 30), (0, 54)]
        self.coupons = np.concatenate([np.ones(c)*self.discounts[p] for p,c in enum_customers]) # [(3, 10), (1, 23), (2, 30), (0, 54)] --> [0,0,0,0, ... 0.05,0.05,0.05,0.05 ... ]
        

