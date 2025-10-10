import numpy as np
import matplotlib.pyplot as plt


#############1a
def simulate_urn():
    urn = ['red'] * 3 + ['blue'] * 4 + ['black'] * 2
    roll = np.random.randint(1, 6)
    if roll in [2, 3, 5]:
        urn.append('black')
    elif roll == 6:
        urn.append('red')
    else:
        urn.append('blue')

    drawn = np.random.choice(urn)
    return drawn

##############1b
def estimate_red_probability(n=100000):
    red_count = 0
    for _ in range(n):
        if simulate_urn() == 'red':
            red_count += 1
    return red_count / n

red_prob = estimate_red_probability()
print(red_prob)

############1c
# P(die=2,3,5) = 3/6 = 0.5 -> +1 black
# P(die=6) = 1/6 -> +1 red
# P(die=1,4) = 2/6 = 1/3 -> +1 blue
#
# Red: 3 + (1/6)
# Blue: 4 + (1/3)
# Black: 2 + (0.5)
def theoretical_red_probability():
    expected_black = 2 + 0.5
    expected_red = 3 + 1/6
    expected_blue = 4 + 1/3
    total = expected_red + expected_blue + expected_black
    return expected_red / total

theoretical_red = theoretical_red_probability()
print(theoretical_red)

##########2.1
poisson_1 = np.random.poisson(1, 1000)
poisson_2 = np.random.poisson(2, 1000)
poisson_5 = np.random.poisson(5, 1000)
poisson_10 = np.random.poisson(10, 1000)

##########2.2
lambdas = [1, 2, 5, 10]
randomized_poisson = [np.random.poisson(lam=np.random.choice(lambdas)) for _ in range(1000)]

##########2.2a
plt.hist(poisson_1, bins=30, color='red', edgecolor='black', alpha=0.7)
plt.title('lambda=1')
plt.show()

plt.hist(poisson_2, bins=30, color='red', edgecolor='black', alpha=0.7)
plt.title('lambda=2')
plt.show()

plt.hist(poisson_5, bins=30, color='red', edgecolor='black', alpha=0.7)
plt.title('lambda=5')
plt.show()

plt.hist(poisson_10, bins=30, color='red', edgecolor='black', alpha=0.7)
plt.title('lambda=10')
plt.show()

plt.hist(randomized_poisson, bins=30, color='red', edgecolor='black', alpha=0.7)
plt.title('randomized')
plt.show()


###########2.2b
#fixed distributions -> centered around lambda
#randomized distribution -> not centered around lambda, right skewed distribution
#randomized lambda -> higher parameter uncertainty, asymmetric data, harder to train ml models
############2.2c
biased_lambdas = np.random.choice([1, 2, 5, 10], size=1000, p=[0.1, 0.1, 0.7, 0.1])
biased_poisson = [np.random.poisson(lam=lam) for lam in biased_lambdas]

plt.hist(biased_poisson, bins=30, color='red', edgecolor='black', alpha=0.7)
plt.title('lambda=5 more likely')
plt.show()

#clear peak at 5, less skewed