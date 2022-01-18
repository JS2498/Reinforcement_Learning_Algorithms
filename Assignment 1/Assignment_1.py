# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:14:19 2022

@author: Keerthana_Jayanth
"""
#%%
#Import the required libraries

import numpy as np
from scipy import stats
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# from markovchain import MarkovChain

#seed for the entire code
seed = 5
random.seed(seed)

#%%

#Multinomial distribution 
"""
Problem 1 : Sample n points from Multinomial, Uniform, Gaussian, Exponential 
Distributions. Verify their distributions using histogram plot.
"""

"""
Multionomial Distribution :
    Parameters :  
        * p1, p2, ..., pk where k is the no. of outcomes 
        * n : no. of trials       
        
    Support :
        * xi with range 1 to n with sum(xi) = n
    Prob Mass Function :
        * fact(n)*sum(pi^xi)/(product(fact(xi)))
    Mean_i = npi for all i
    Variance_i = npi*(1-pi)
    
    * When n = 1 then it becomes Categorical Distribution 
    * When n = 1 and k = 2 then it becomes Bernoulli Distribution
    * When n = 1 and k > 2 then it becomes Binomial Distribution
"""



def Multinomial_samples(probs, n_list):
    # probs = [0.2, 0.4, 0.3, 0.1]
    # n_list = [100, 1000, 10000] 
    outcomes = np.arange(1,len(probs)+1)
    # x = random.uniform(0,1, n)
    # #multinomial distribution using Scipy.stats
    # multinomial_rv = stats.multinomial(n, probs, seed = seed)
    # #Draw the samples from multinomial distribution
    # samples = multinomial_rv.rvs(size = 1)  #Remove the extra dimension before plotting
    # print(x)
    
    fig = plt.figure(figsize = (len(n_list)*8,5))
    for i in range(len(n_list)):
        samples = random.multinomial(n_list[i], probs)
        fig.add_subplot(1,len(n_list),i+1)
        plt.bar(outcomes, height= samples/n_list[i])
        plt.xlabel("Outcomes")
        plt.ylabel("PMF of each Outcome")
        plt.title(f"No. of samples = {n_list[i]}")
    plt.suptitle("Multinomial Distribution")
    plt.show()


def Uniform_Samples(low,high, n_list):
    
    fig, ax = plt.subplots(1,len(n_list), figsize = [5*len(n_list), 5])
    for i in range(len(n_list)):
        samples = random.uniform(low, high, size = n_list[i])
        # fig.add_subplot(1,len(n_list),i+1)
        sns.histplot(samples, kde = True, bins=  int(n_list[i]*0.1), ax = ax[i], stat = 'density' )
        ax[i].set_xlabel("Outcomes")
        ax[i].set_ylabel("PDF of each Outcome")
        ax[i].set_title(f"No. of samples = {n_list[i]}")
        ax[i].set(xlim = (0,1), ylim = (0,None))
    plt.suptitle("Uniform Distribution")
    plt.show()
    
def Normal_Samples(mean,variance, n_list):
    
    fig, ax = plt.subplots(1,len(n_list), figsize = [5*len(n_list), 5])
    for i in range(len(n_list)):
        samples = random.normal(loc = mean, scale  = np.sqrt(variance), size = n_list[i])
        sns.histplot(samples, kde = True, bins=  int(n_list[i]*0.1), ax = ax[i], stat = 'density' )
        ax[i].set_xlabel("Outcomes")
        ax[i].set_ylabel("PDF of each Outcome")
        ax[i].set_title(f"No. of samples = {n_list[i]}")
        # fig.add_subplot(1,len(n_list),i+1)
        # plt.hist(samples, bins = int(n_list[i]*0.1), density = True)
        # plt.xlabel("Outcomes")
        # plt.ylabel("PDF of each Outcome")
        # plt.title(f"No. of samples = {n_list[i]}")
        ax[i].set(xlim = (-5*variance,5*variance), ylim = (0,None))
    plt.suptitle("Normal Distribution")
    plt.show()

def Exponential_Samples(rate, n_list):
    
    fig, ax = plt.subplots(1,len(n_list), figsize = [5*len(n_list), 5])
    for i in range(len(n_list)):
        samples = random.exponential(scale = rate, size = n_list[i])
        sns.histplot(samples, kde = True, bins=  int(n_list[i]*0.1), ax = ax[i], stat = 'density' )
        ax[i].set_xlabel("Outcomes")
        ax[i].set_ylabel("PDF of each Outcome")
        ax[i].set_title(f"No. of samples = {n_list[i]}")
        ax[i].set(xlim = (0,None), ylim = (0,None))
    plt.suptitle("Exponential Distribution")
    plt.show()
n_list = [100, 1000, 10000] #each item specifies the no. of samples to obtain from a distribution
#Plot the histogram (Density plot) of Multinomail Distribution Samples
probs = [0.2, 0.4, 0.3, 0.1]

print("--------------------------Problem 1-----------------------------------")

Multinomial_samples(probs, n_list)

#Plot the histogram (Density plot) of Uniformly Distributed Samples
low = 0
high = 1
Uniform_Samples( low, high, n_list)


#Plot the histogram (Density plot) of Gaussian Distributed Samples
mean = 0
variance = 1
Normal_Samples(mean, variance, n_list)

#Plot the histogram (Density plot) of Exponential Distributed Samples
rate = 0.5
Exponential_Samples(rate, n_list)
"""
Inference : 
    As the number of samples increases the histogram plot becomes close 
    to the true distribution as specified by the Law of Large numbers. 
    (Almost surely or Convergence in probability implies convergence in distribution)
"""



#%%
"""
Problem 2 : 
    Generate RV with mean, mu and standard deviation, sigma using uniform random variable4
    Two methods :
        1. Inverse Transform Method
        2. Box-Muller Transform
"""

mean = 10
variance = 5
n = 10000
low = 0
high = 1

print("--------------------------Problem 2-----------------------------------")


# random.seed(5)
uniform_samples = random.uniform(low, high, size = n)
#Inverse Transform method
IT_samples = np.sqrt(2*variance)*scipy.special.erfinv(2*uniform_samples-1) + mean
IT_mean = np.sum(IT_samples)/n
IT_variance = np.sum((IT_samples-IT_mean)**2)/n
print(f"Estimated Mean = {IT_mean} and variance = {IT_variance} of samples obtained by Inverse Transform Method")

#Box - Muller Transform
random.seed(5*seed)
uniform_samples_new = random.uniform(low, high, size = n)
random.seed(seed)
BM_samples = np.sqrt(-2*variance*np.log(uniform_samples))*np.cos(2*np.pi*uniform_samples_new) + mean
BM_mean = np.sum(BM_samples)/n
BM_variance = np.sum((BM_samples-BM_mean)**2)/n
print(f"Estimated Mean = {BM_mean} and variance = {BM_variance} of samples obtained by Box-Muller Algorithm ")

        
true_normal = random.normal(mean, np.sqrt(variance), size = n)
fig= plt.figure(figsize = (10,5))
x = np.linspace(mean - 5*variance, mean + 5*variance, n)
plt.plot(x, stats.norm.pdf(x, mean, np.sqrt(variance)), color = 'red')
sns.histplot(IT_samples,  bins= 100, stat = 'density', color = 'blue')
plt.legend([f"True Distribution $\mu ={mean}$, $\sigma^2 ={variance}$", "Inverse Transform Method"])
plt.xlabel("x")
# plt.ylim([0,0.5])
plt.xlim([-2*variance + mean , 2*variance + mean ])
plt.show()

fig = plt.figure(figsize = (10,5))
plt.plot(x, stats.norm.pdf(x, mean, np.sqrt(variance)), color = 'red')
sns.histplot(BM_samples,  bins= 100, stat = 'density', color = 'green' )
plt.xlabel("x")
plt.legend([f"True Distribution $\mu ={mean}$, $\sigma^2 ={variance}$", "Box-Muller Method"])
# plt.ylim([0,0.5])
plt.xlim([-2*variance + mean , 2*variance + mean ])
plt.show()

#%%
"""
Plot the given functions (sin(x), sin(x)*exp(-x**2)) and find the area under the curve.
"""
n = 100000
low = 0
high = np.pi
x = random.uniform(low, high, size = n)

#Integrate sqrt(sinx)
g = np.sqrt(np.sin(x))
scipy.integrate
area_g = np.mean(g)*(high-low)
func = lambda x : np.sqrt(np.sin(x))
numerical_area_g = scipy.integrate.quad(func,low, high)[0]

print("--------------------------Problem 3-----------------------------------")

fig = plt.figure(figsize = (8,5))
x = np.linspace(low,high, num = n)
plt.plot(x, np.sqrt(np.sin(x)))
plt.xlabel("x")
plt.ylabel("$\sqrt{(sin(x))}$")
# plt.ylim([0,0.5])
plt.xlim([low, high])
plt.show()

print(f"Area of sqrt(sin(x)) without using numerical technique is {area_g}.")
print(f"Area of sqrt(sin(x)) using numerical technique (scipy integrate) is {numerical_area_g}.")
print(f"The error in the calculation is {abs(numerical_area_g - area_g)} for no. of samples being {n}.")
print()
#Intergrate sqrt(sinx)*exp(-x^2)

g = np.sqrt(np.sin(x))*np.exp(-x**2)
scipy.integrate
area_g = np.mean(g)*(high-low)
func = lambda x : np.sqrt(np.sin(x))*np.exp(-x**2)
numerical_area_g = scipy.integrate.quad(func,low, high)[0]

fig = plt.figure(figsize = (8,5))
x = np.linspace(low,high, num = n)
plt.plot(x, np.sqrt(np.sin(x))*np.exp(-x**2))
plt.xlabel("x")
plt.ylabel("$\sqrt{(sin(x))}e^{-x^2}$")
# plt.ylim([0,0.5])
plt.xlim([low, high])
plt.show()

print(f"Area of sqrt(sin(x))*exp(-x**2) without using numerical technique is {area_g}.")
print(f"Area of sqrt(sin(x))*exp(-x**2) using numerical technique (scipy integrate) is {numerical_area_g}.")
print(f"The error in the calculation is {abs(numerical_area_g - area_g)} for no. of samples being {n}.")


#%%
"""
Markov Chain for the snake and ladder game
"""


def SnakeAndLadder(PTM, initial_state = 0, n_runs = 10000, end_state = 8):
    """
    I/p :
        PTM : Probability Transition Matrix of shape (num_states x num_states)
        initial_state : Starting state of the Markov Chain
        n_runs : Number of times the snake and ladder games has to be played
        
    Objective :
        Obtain the probability of reaching the end state (8) both analytically and using simulations
        
    """
    
    initial_dist = np.eye(9) #each row represents the inital state for the start of the game
    steps_list = []
    pi_end = 0
    for i in range(N):
        s = initial_state #initial state
        steps = 0
        while(1):
            steps += 1 
            s = np.random.choice(PTM.shape[0], p = PTM[s,:])
            if (s == 2) or (s==4):
                steps_list.append(steps)
                break
            if (s==end_state):
                pi_end +=1
                steps_list.append(steps)
                break
    pi_end_analytical = initial_dist[initial_state,:]@(np.linalg.matrix_power(PTM,2000))
    pi_end = pi_end/N
    print(f"The probability of reaching end state starting from position 0 is {pi_end} using simulations")   
    print(f"The probability of reaching end state starting from position 0 is {pi_end_analytical[8]: .5f} using analytical method")   


#Assuming that we stay back in state 7 if the outcome ohter than 1.
PTM = np.array([[4/6, 1/6, 1/6, 0, 0, 0, 0, 0, 0],
                [0, 4/6, 1/6, 1/6, 0, 0, 0, 0, 0],
                [0, 0, 0, 0 , 1, 0, 0, 0, 0],
                [0, 0, 0, 4/6, 1/6, 1/6, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0 , 4/6, 1/6, 1/6, 0],
                [0, 0, 0, 0, 0, 0, 4/6, 1/6, 1/6],
                [0, 0, 0, 0, 0 , 0, 0, 5/6, 1/6],
                [0, 0, 0, 0, 0 , 0, 0, 0, 1]])
N = 50000

print("--------------------------Problem 4-----------------------------------")

SnakeAndLadder(PTM, initial_state = 0 , n_runs = N, end_state = 8)

"""
Inference:
    * The given Markov Chain contains both transient states and recurrent states.
    * Since starting form states 2 or 4 we may not be able to reach other states (not accessible),
      we say that the Markov chain is reducible and hence the Markov chain is not ergodic.
    * Also, steady state distribution doesn't exist because of the transitions between state 2 and 4.
"""