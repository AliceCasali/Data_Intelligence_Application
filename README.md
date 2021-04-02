# Data Intelligence Application Exam Project

# Pricing and Matching

**Scenario**: Consider the scenario in which a shop has a number of promo codes to incentivize the customers that buy an item to buy a different item. The customers can belong to different classes and the promo codes can provide different discounts.

**Environment**: Imagine two items (referred to as first and second items; for each item we have an infinite number of units) and four customers’ classes. The daily number of customers of each class is described by a potentially different (truncated) Gaussian probability distribution. Each class is also associated with a potentially different conversion rate returning the probability that the user will buy the first item at a given price.

Once a buyer has bought the item, she/he can decide to buy the second item that can be or not promoted. There are four different promos P0, P1, P2, P3, each corresponding to a different level of discount. P0 corresponds to no discount. Given the total number of customers, the business unit of the shop decides the number of promos as a fraction of the total number of the daily customers and is fixed (use two different settings in your experiments that you are free to choose). Each customers’ class is also associated with a potentially different conversion rate returning the probability that the user will buy the second item at a given price after she/he has bought the first. The promos will affect the conversion rate as they actually reduce the price. 

Every price available is associated with a margin obtained by the sale that is known beforehand. This holds both for the first and the second item. 

The conversion rates will change during time according to some phases due to, e.g., seasonality.

# Team
- [Alice Casali] (https://github.com/AliceCasali/)
- [Anna Giovannacci] (https://github.com/annagiovannacci)
- [Ege Saygılı] (https://github.com/egesaygili)
- [Francesco Amorosini] (https://github.com/FrancescoAmorosini)
