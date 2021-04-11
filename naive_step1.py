import numpy as np
customers = np.random.normal(250, 100, 4).astype(int)
customers_det = [250, 250, 250, 250]
prices = [250, 400]
promos = [0.0, 0.1, 0.35, 0.60]
conv_rate1 = [0.3, 0.4, 0.25, 0.45]
conv_rate2 = [[0.1, 0.2, 0.15, 0.2],#p0
              [0.15, 0.25, 0.2, 0.25],#p1
              [0.2, 0.35, 0.25, 0.4],#p2
              [0.4, 0.45, 0.35, 0.6]]#p3
promo_fractions = [0.35, 0.35, 0.20, 0.10]
print(conv_rate2[0][2])


print(customers)

rewards_from_item2 = np.zeros([4,4])

for i in range(len(conv_rate1)): # promo
    for j in range(len(conv_rate1)): # customer class
        rewards_from_item2[i][j] = customers_det[j]*conv_rate1[j]*conv_rate2[i][j]*prices[1]*(1-promos[i])

print(rewards_from_item2)
selected_promos = [-1, -1, -1, -1]
rewards_per_customer = [0, 0, 0, 0]
for j in range(len(conv_rate1)):
    temp = rewards_from_item2[:,j]
    promo_no = np.argmax(temp)
    selected_promos[j] = promo_no
    rewards_per_customer[j] = temp[promo_no]

print()
print('SELECTED PROMOS:')
print(selected_promos)
print()
print('REWARDS:')
print(rewards_per_customer)

total_reward = 0

for i in range(len(conv_rate1)):
    total_reward += prices[0]*conv_rate1[i]*customers_det[i]
    total_reward += rewards_per_customer[i]

print()
print('TOTAL REWARD:')
print(total_reward)

