import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import math
from matplotlib import cm

PI=math.pi

def get_one_vote(mu, sigma, mu2, sigma2, nb_candidates):
    vals=np.random.normal(mu, sigma, nb_candidates-1)
    val1=np.random.normal(mu2, sigma2, 1)

    s=np.sum(val1)+np.sum(vals)

    all_vals=[]
    all_vals.append(val1[0]/s)
    for i in range(nb_candidates-1):
        all_vals.append(vals[i]/s)

    return np.array(all_vals)


# provides the probabilities that each value is from distribution or another
def get_probabilities_distribution(samples, mu, sigma, mu2, sigma2):
    values1=[]
    values2=[]
    for i in range(len(samples)):
        val=samples[i]
        pr2=1/(math.sqrt(2*PI*sigma**2))*math.exp(-((val-mu)**2)/(2*sigma**2))
        values2.append(pr2)
        pr1=1/(math.sqrt(2*PI*sigma2**2))*math.exp(-((val-mu2)**2)/(2*sigma2**2))
        values1.append(pr1)

    return values1, values2

#gets the probability that each value is the correct one
def get_proba_correct_values(values1, values2):
    probas=[1]*len(values1)
    for i in range(len(values1)):
        for j in range(len(values1)):
            if j!=i:
                probas[i]*=values2[j]
            else:
                probas[i]*=values1[i]

    total=np.sum(np.array(probas))
    res=[probas[i]/total for i in range(len(probas))]

    return res


epsilon=0.0001
sigma=0.1
nb_candidates=10
mu=(1-epsilon)/nb_candidates
nb_tries=1000

"""
#Inlfuence of epsilon

epsilons=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1]
scores=[]
for epsilon in epsilons:
    mu=(1-epsilon)/nb_candidates

    nb_corrects=0
    for i in range(nb_tries):
        samples=get_one_vote(mu, sigma, mu+epsilon, sigma, nb_candidates)
        values1, values2=get_probabilities_distribution(samples, mu, sigma, mu+epsilon, sigma)
        probas=get_proba_correct_values(values1, values2)
        index=np.argmax(np.array(probas))
        if index==0:
            nb_corrects+=1
    scores.append(nb_corrects)



plt.plot(epsilons, scores)
plt.xscale('log')
plt.axhline(y=100, color='red', label="ideal score")
plt.xlabel("Epsilon")
plt.ylabel("Amount of votes correctly revealed to the attackers")
plt.title("Number of votes revealed out of 1000 depending on epsilon, 10 candidates, sigma="+str(sigma))
plt.show()
"""






"""
#Influence of sigma
sigmas=[10**(-k/3) for k in range(0, 20)]
print(sigmas)
scores=[]
for sigma in sigmas:
    mu=(1-epsilon)/nb_candidates

    nb_corrects=0
    for i in range(nb_tries):
        samples=get_one_vote(mu, sigma, mu+epsilon, sigma, nb_candidates)
        values1, values2=get_probabilities_distribution(samples, mu, sigma, mu+epsilon, sigma)
        probas=get_proba_correct_values(values1, values2)
        index=np.argmax(np.array(probas))
        if index==0:
            nb_corrects+=1
    scores.append(nb_corrects)



plt.plot(sigmas, scores, label="nb votes revealed")
plt.xscale('log')
plt.axhline(y=100, color='red', label="ideal score")
plt.xlabel("Sigma")
plt.ylabel("Amount of votes correctly revealed to the attackers")
plt.title("Number of votes revealed out of 1000 depending on sigma, 10 candidates, epsilon="+str(epsilon))
plt.legend()
plt.show()
"""


fig = plt.figure()
ax = plt.axes(projection="3d")
nb_points=10
begin=6
epsilons=[(-k/5) for k in range(begin, begin+nb_points)]
sigmas=[(-k/5) for k in range(0, nb_points)]
epsilons, sigmas = np.meshgrid(epsilons, sigmas)
print(epsilons, sigmas)
scores = np.zeros((len(epsilons), len(sigmas)))
ideals=np.zeros((len(epsilons), len(sigmas)))
for x in range(len(epsilons)):
    for y in range(len(epsilons[x])):
        epsilon=10**epsilons[x][y]
        sigma=10**sigmas[x][y]

        mu=(1-epsilon)/nb_candidates

        nb_corrects=0
        for i in range(nb_tries):
            samples=get_one_vote(mu, sigma, mu+epsilon, sigma, nb_candidates)
            values1, values2=get_probabilities_distribution(samples, mu, sigma, mu+epsilon, sigma)
            probas=get_proba_correct_values(values1, values2)
            index=np.argmax(np.array(probas))
            if index==0:
                nb_corrects+=1

        scores[x][y]=nb_corrects
        ideals[x][y]=nb_tries/nb_candidates

ax.plot_surface(epsilons,sigmas,scores, cmap=cm.coolwarm)
ax.plot_surface(epsilons,sigmas,ideals, cmap='Greens')
ax.set_xlabel('log epsilons')
ax.set_ylabel('log sigmas')
ax.set_zlabel('Scores')

plt.show()
