from random import random
from mab.utilities import *
import numpy as np
import itertools

class Environment():
    def __init__(self, prices1, prices2, discounts, n_phases):
        #Shop related data
        self.prices1, self.prices2, self.discounts, self.n_phases = prices1, prices2, discounts, n_phases
        self.t = -2

        #Customers that will visit the shop during the day
        self.generate_next_day_customers()
        self.n_classes = len(self.customers)
        
        #Conversion rates
        self.true_conv1 = np.array([[generate_conversion_rate(prices1) for x in range(self.n_classes)] for p in range(n_phases)]) # [phase x class x price]
        self.true_conv2 = np.array([[[generate_conversion_rate(prices2) for x in range(self.n_classes)] for y in range(len(discounts))] for p in range(n_phases)]) # [phase x promo x class x price]      
    
    def round1(self, cust_class, price, horizon=365):
        current_phase = int(self.t / (horizon/self.n_phases))
        return np.random.binomial(1, self.true_conv1[current_phase, cust_class, index(self.prices1, price)])
    
    def round2(self, cust_class, promo, price, horizon=365):
        current_phase = int(self.t / (horizon/self.n_phases))
        return np.random.binomial(1, self.true_conv2[current_phase, promo, cust_class, index(self.prices2, price)])

    def set_conv_rates(self, cr1, cr2):
        self.true_conv1 = cr1
        self.true_conv2 = cr2

    #Adjust "means" and "variances" in order to test different distributions of customers 
    def generate_next_day_customers(self, means = ([25, 25, 25, 25]), variances =  ([10,10,10,10])):
        self.t += 1
        means = np.array(means)
        variances = np.array(variances)
        self.customers = np.array([clamp(int(np.random.normal(m, v)), int(m/2), int(3*m/2)) for m,v in zip(means, variances)])
   
    def arrival_of_a_single_customer(self):
       class_of_customer = np.random.randint(4, size=(1))
       return class_of_customer

    def get_phase(self, horizon=365):
        return int(self.t / (horizon/self.n_phases))