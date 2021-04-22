import numpy as np
import statistics
from scipy.stats import truncnorm

class Environment():
    def __init__(self, gaussian_parameters, discounts, prices, conv_rate1, conv_rate2):
        #self.probabilities = probabilities
        means, variances = zip(*gaussian_parameters)
        means = np.array(means)
        variances = np.array(variances)

        myclip_a = 0
        myclip_b = 50
        a,b = (myclip_a - means)/variances, (myclip_b-means)/variances

        self.class_customer_distribution = [truncnorm(alpha,beta) for alpha,beta in zip(a,b)]
        self.discounts = discounts
        self.prices = prices
        self.conv1 = conv_rate1
        self.conv2 = conv_rate2
        self.nclasses = len(means)

    def calculation_probabilities_for_update(self,fractions,graph, class_id, MODE):
        
        p = [ 0 for x in range(0,4)]
        index_max = np.argmax(graph[class_id][0])
        p[index_max] = graph[class_id][0][index_max] #Not so sure about this syntax...
        indices = [x for x in range(0,4) if x!= index_max]
        for i in indices:
            p[i] = graph[class_id][0][i]
        return p    

<<<<<<< HEAD
    ''' def update_probabilities(self,optimal_solution):
=======
    
    def calculation_probabilities_for_update(self,fractions,graph, class_id, MODE):
        
        p = [ 0 for x in range(0,4)]
        index_max = np.argmax(graph[class_id][0])
        p[index_max] = graph[class_id][0][index_max] #Not so sure about this syntax...
        indices = [x for x in range(0,4) if x!= index_max]
        for i in indices:
            p[i] = graph[class_id][0][i]
        return p


    
#optimal_solution = [0,2,3,1]
#p0,p1,p2,p3 = env.update_probabilities(optimal_solution)
#print(p0)
#print(p1)
#print(p2)
#print(p3)

'''    def update_probabilities(self,optimal_solution):
>>>>>>> 71a16a1308a31ab9eb17f7d5641313512e800f97
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

<<<<<<< HEAD
        return p0,p1,p2,p3'''


#Customers
customerParams = zip([10,10,10,10], [1,1,1,1])
#discounts
discounts = np.array([0.0, 0.5, 0.10, 0.25])
#Prices
prices = np.array([250, 400])
#Conversion Rates for item 1
conv_rate1 = [0.3, 0.4, 0.25, 0.45]
#Conversion Rates for item 2
conv_rate2 = np.array([[0.1, 0.2, 0.15, 0.2],#p0
                    [0.15, 0.25, 0.2, 0.25],#p1
                    [0.2, 0.35, 0.25, 0.4],#p2
                    [0.4, 0.45, 0.35, 0.6]])#p3

env = Environment(customerParams, discounts, prices, conv_rate1, conv_rate2)
=======
        return p0,p1,p2,p3'''
>>>>>>> 71a16a1308a31ab9eb17f7d5641313512e800f97
