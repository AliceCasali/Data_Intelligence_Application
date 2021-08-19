from random import random
from utilities import *
import numpy as np
import itertools

class Environment():
    def __init__(self, prices1, prices2, discounts):
        #Shop related data
        self.prices1, self.prices2, self.discounts = prices1, prices2, discounts

        #Customers that will visit the shop during the day
        self.generate_next_day_customers()
        self.n_classes = len(self.customers)
        
        #Conversion rates
        self.true_conv1 = np.array([generate_conversion_rate(prices1) for x in range(self.n_classes)]) # [class x price]
        self.true_conv2 = np.array([[generate_conversion_rate(prices2) for x in range(self.n_classes)] for y in range(len(discounts))]) # [class x promo x price]      
    
    def round1(self, cust_class, price):
        return np.random.binomial(1, self.true_conv1[cust_class, index(self.prices1, price)])
    
    def round2(self, cust_class, promo, price):
        return np.random.binomial(1, self.true_conv2[cust_class, promo, index(self.prices2, price)])

    #Adjust "means" and "variances" in order to test different distributions of customers 
    def generate_next_day_customers(self, means = ([25, 25, 25, 25]), variances =  ([10,10,10,10])):
        means = np.array(means)
        variances = np.array(variances)
        self.customers = np.array([clamp(int(np.random.normal(m, v)), int(m/2), int(3*m/2)) for m,v in zip(means, variances)])
   
    def arrival_of_a_single_customer(self):
       class_of_customer = np.random.randint(4, size=(1))
       return class_of_customer