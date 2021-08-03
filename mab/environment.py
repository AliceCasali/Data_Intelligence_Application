import numpy as np
import statistics
from scipy.stats import truncnorm
import random

def clamp(num, min_val, max_val):
	return max(min(num, max_val), min_val)

class Environment():
    #def __init__(self, gaussian_parameters, discounts, prices, conv_rate1, conv_rate2, fractions, fraction_idxs):
        #self.probabilities = probabilities
    def __init__(self, gaussian_parameters, discounts, prices, conv_rate1, conv_rate2):
        means, variances = zip(*gaussian_parameters)
        means = np.array(means)
        variances = np.array(variances)

        self.customers = np.array([clamp(int(np.random.normal(m, v)), int(m/2), int(3*m/2)) for m,v in zip(means, variances)])
        self.discounts = discounts
        self.prices = prices
        self.conv1 = conv_rate1
        self.conv2 = conv_rate2
        self.nclasses = len(means)
        #self.fractions = fractions
        #self.fraction_idxs = fraction_idxs

    def update_customers(self, gaussian_parameters):
        means, variances = zip(*gaussian_parameters)
        self.customers = np.array([clamp(int(np.random.normal(m, v)), int(m/2), int(3*m/2)) for m,v in zip(means, variances)])

    def round1(self, pulled_arm, cust):
        return np.random.binomial(1, self.conv1[cust[0], pulled_arm[1]])
    
    def round_2(self, pulled_arm):
       return np.random.binomial(1, self.conv2[pulled_arm])
    
    def calculation_probabilities_for_update(self,fractions,graph, class_id, MODE):
        p = [ 0 for x in range(0,4)]
        index_max = np.argmax(graph[class_id][0])
        p[index_max] = graph[class_id][0][index_max] #Not so sure about this syntax...
        indices = [x for x in range(0,4) if x!= index_max]
        for i in indices:
            p[i] = graph[class_id][0][i]
        return p    
    
    def update_probabilities(self,optimal_solution):
        #given a optimal solution, it returns an array for each discount
        #for example: in optimal solution c0 is assigned to p0, it will be:
        #p0=[0.75 0.15 0.05 0.05]

        #I imagined optimal solution as an array
        #optimal solution = [1 0 2 3]
        #this means: customer 1 takes p0 and so on
        best_assignment_0 = optimal_solution[0]
        best_assignment_1 = optimal_solution[1]
        best_assignment_2 = optimal_solution[2]
        best_assignment_3 = optimal_solution[3]

        p0 = self.calculation_probabilities_for_update(best_assignment_0)
        p1 = self.calculation_probabilities_for_update(best_assignment_1)
        p2 = self.calculation_probabilities_for_update(best_assignment_2)
        p3 = self.calculation_probabilities_for_update(best_assignment_3)

        return p0,p1,p2,p3