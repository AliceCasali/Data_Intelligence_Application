import numpy as np
import statistics
from scipy.stats import truncnorm
from scipy.optimize import linear_sum_assignment
import random
import itertools
import matplotlib.pyplot as plt

def clamp(num, min_val, max_val):
	return max(min(num, max_val), min_val)

def generate_conversion_rate(prices):
	val = np.random.rand(len(prices))
	conversion_rates = np.sort(val)[::-1]
	return val

def index(array, value):
	return np.where(array == value)[0][0]

def slope(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    return m
