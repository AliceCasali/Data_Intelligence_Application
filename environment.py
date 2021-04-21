  
import numpy as np
import statistics
from scipy.stats import truncnorm



class Environment():
    '''
        gaussian_parameters should be a list of lists of the form:
            gaussian_parameters [0] : mean of class i
            gaussian_parameters [1] : variance of class i
    '''
    def __init__(self, gaussian_parameters,discounts):#,conversion_rate):
        #self.probabilities = probabilities
        self.mean = gaussian_parameters[0]
        self.variances = gaussian_parameters[1]

        myclip_a = 0
        myclip_b = 200
        self.class_customer_distribution = []
        
        for i in range(0,4):
            a, b = (myclip_a - gaussian_parameters[0][i]) / gaussian_parameters[1][i], (myclip_b - gaussian_parameters[0][i]) / gaussian_parameters[1][i]
            self.class_customer_distribution.append(truncnorm(a,b))
        
        '''self.conversion_rate_0 = conversion_rate[0]
        self.conversion_rate_1 = conversion_rate[1]
        self.conversion_rate_2 = conversion_rate[2]
        self.conversion_rate_3 = conversion_rate[3] '''    
    
        self.promo0 = 0
        self.promo1 = discounts[0]
        self.promo2 = discounts[1]
        self.promo3 = discounts[2]

    
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

        return p0,p1,p2,p3'''