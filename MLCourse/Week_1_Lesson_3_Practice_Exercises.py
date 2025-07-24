"""Exercise 1 - Bayes' Theorem Application"""
# Classification of Spam Emails
# Prior: P(Spam) = 0.3
# Likelihood: P("FREE" appears | Spam) = 0.8
# Likelihood: P("FREE" appears | Not Spam) = 0.1
# Calculate P(Spam | "FREE" appears)

import numpy as np

P_Spam = 0.3
P_Free_Spam = 0.8
P_Free_NotSpam = 0.1

#Shorthand S=Spam; F=Free
#P(S|F) = P(F|S)*P(S)/P(F)
#P(F|S) is P_Free_Spam
#P(S) is P_Spam
#P(F) is the probability of any email containing Free
P_Free = P_Free_Spam*P_Spam + P_Free_NotSpam*(1-P_Spam)
P_Spam_Free = (P_Free_Spam*P_Spam)/P_Free
print(f"The probability of an email being spam given that the word \"Free\" appears is {P_Spam_Free:.1%}")

"""Exercise 2 - Distribution Marketing"""
# Identify the most appropriate probability distributions:
# 1. Number of defective items in a batch of 100
# 2. Time between customer arrivals at a store
# 3. Probability of success in a marketing campaign
# 4. Heights of adult males in a population

# 1. Binomial distribution. Either a product is defective or functional. We meet the four criteria for binomial:
#       1a. Fixed number of trials (100)
#       1b. Each trial is independent
#       1c. Each trial has one of two outcomes
#       1d. The probability of success is constant (essentially).
#   X ~ Binomial(n=100,p), where p is the underlying probability of an item being defective
# 2. Exponential distribution. A Poisson distribution would model the number of events, however the exponential
#       models the time between events.
# 3. Any distribution may be representative of the probability of success in a marketing campaign, depending
#       on many factors. I would probably (pun intended) use a normal distribution, but perhaps
#       a beta distribution depending on what variables we are looking at, specifically. I could see some metrics
#       for "success" leading to a Poisson distribution (e.g. if we tried to represent the probability of sales
#       in a certain amount of time, if we sold consumables so that we knew the mean rate at which they sold and
#       the pace at which customers used them.)
#       For Beta Distribution, if we are trying to model the unknown underlying probability of succeess, then it
#           is the correct choice.
# 4. A normal distribution around the average male height is most appropriate.

"""Exercise 3 - Hypothesis Testing"""
# Design a hypothesis test to determine if a new ML model performs
#   significantly better than the baseline:
# * Define null and alternative hypothesis
# * Choose appropriate test statistic
# * Set significance level
# * Interpret results

# Null Hypothesis: H0: μ₁ <= μ₂ (the new model is worse than or equal to the old model)
# Alternative Hypothesis: H1: μ₁ > μ₂, (the new model is faster than the old model), subject to testing
#   significant statistical difference and examination of the underlying reasons for difference.
# For our hypothetical, we will be measuring the convergence speed of a ML model that minimizes a function.
# Run both models n times, take the average for each data set μ₁ and μ₂.
# Use a library function to calculate t_stat and p_value, using a one-tailed test because we are focused
#   on improvement, not on statistical difference. A paired samples t-test is best because the cross-validation
#   data will be the same.
# The most common alpha = 0.05. If p_value < alpha, then we reject the null hypothesis. The p-value represents
#   the odds that if the null hypothesis were true, there would only be a p-value percent chance of observing
#   the difference we saw based purely on random chance.
# However, for the ML optimization model I proposed, a "new" model may have a 1% improvement over an old model,
#   and this improvement would become apparent as the number of trials approached infinity. We could simply
#   perform more and more runs until the p-value demonstrated significant difference. However, in a practical
#   sense, a 1% improvement may not be helpful if we're going from 10 seconds to 9.9 seconds. It might be
#   helpful if we're going from 1000 minutes to 990 minutes, however. The context will aid us most in
#   performing a proper interpretation of the results. We could use an Effect Size metric.

"""Exercise 4 - Information Theory Application"""
# Calculate Entropy for different classification scenarios
# 1. Balanced binary classification (50/50 split)
# 2. Imbalanced binary classification (90/10 split)
# 3. Multi-class with 4 equally likely classes
# 4. Multi-class with probabilities [0.5, 0.3, 0.1, 0.1]
def Entropy(X):
    #Requires 1-D array
    if len(np.shape(X)) != 1:
        print("Array must be 1D")
        return
    entropy = 0
    for i in range(len(X)):
        entropy += X[i]*np.log2(X[i])
    entropy = entropy*-1
    return entropy

scenario_1 = np.array([0.5, 0.5])
scenario_2 = np.array([0.9, 0.1])
scenario_3 = np.array([0.25, 0.25, 0.25, 0.25])
scenario_4 = np.array([0.5, 0.3, 0.1, 0.1])

print(f"Balanced 50/50 split entropy: {Entropy(scenario_1):.3}")
print(f"Imbalanced binary 90/10 split entropy: {Entropy(scenario_2):.3}")
print(f"Multi-class with 4 equally-likely classes: {Entropy(scenario_3):.3}")
print(f"Multi-class with probabilities [0.5, 0.3, 0.1, 0.1]: {Entropy(scenario_4):.3}")



"""Exercise 5 - Confidence Intervals"""
# Given cross-validation results [0.82, 0.85, 0.83, 0.87, 0.84], calculate:
# * 95% CI for mean performance
# * What sample size would be needed for a +/- 1% margin of error

from scipy import stats

def ConfidenceInterval(data_set, alpha, verbose = False):
    if len(np.shape(data_set)) != 1:
        print("Array must be 1D")
        return
    mean = np.mean(data_set)
    std  = np.std(data_set, ddof=1)
    n = len(data_set)
    t_val = stats.t.ppf(1-alpha/2, n-1)
    margin = t_val * std / np.sqrt(n)
    ci_lower = mean - margin
    ci_upper = mean + margin
    if verbose:
        print(f"The 95% confidence interval for the data set is: [{ci_lower:.2} , {ci_upper:.2}]")
    return ci_lower, ci_upper

data = np.array([0.82, 0.85, 0.83, 0.87, 0.84])
alpha = .05
lower, upper = ConfidenceInterval(data, alpha, True)

def RequiredSampleSize(data_set, alpha, req_margin, verbose = False):
    if len(np.shape(data_set)) != 1:
        print("Array must be 1D")
        return
    mean = np.mean(data_set)
    std  = np.std(data_set, ddof=1)
    iteration_limit = 100
    n = 1
    while n <= iteration_limit:
        t_val = stats.t.ppf(1-alpha/2, n-1)
        margin = t_val * std / np.sqrt(n)
        if margin < req_margin: 
            if verbose:
                print(f"The sample size required for {req_margin:.0%} margin of error is: {n}")
            return n
        n += 1
    print("Could not find value in 100 iterations.")

MOE = .01
req_sample_size = RequiredSampleSize(data, alpha, MOE, True)