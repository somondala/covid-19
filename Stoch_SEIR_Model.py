# -*- coding: utf-8 -*-
"""
Created on Thu Apr 02 15:27:21 2020

@author: Somnath.Mondal
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

#matplotlib inline
#from IPython.display import display, HTML
# from Stoch_Iteration import Stoch_Iteration
def update(old_pop, Parameters, change):
    
    #dt = 0.01
    #new_pop = np.array([49995, 1, 1, 0, 0, 0])
    beta = Parameters[0]
    delta1 = Parameters[1]
    gamma1 = Parameters[2]
    theta = Parameters[3]
    kappa = Parameters[4]
    mu1 = Parameters[5]
    p_s = Parameters[6]
    gamma3 = Parameters[7]
    mu2 = Parameters[8]
    N = Parameters[9]
    
    S = old_pop[0]
    E = old_pop[1]
    I = old_pop[2]
    A = old_pop[3]
    Q = old_pop[4]
    R = old_pop[5]
    
    
    rate = np.zeros([9])    
    
    rate[0] = beta*S*(I+delta1*A)/N 
    rate[1] = p_s*kappa*E
    rate[2] = (1.-p_s)*kappa*E
    rate[3] = gamma1*I
    rate[4] = theta*I
    rate[5] = mu1*I
    rate[6] = gamma1*A
    rate[7] = gamma3*Q
    rate[8] = mu2*Q
    

    
    u1 = np.random.rand()
    u2 = np.random.rand()
    
    if np.sum(rate)>0:
        dt = -np.log(u2)/(np.sum(rate))
    else:
        return()
    
    event = np.min(np.where(np.cumsum(rate)>=u1*np.sum(rate)))
    new_pop = old_pop + change[event,:]
    
    return dt, new_pop

    

def Stoch_Iteration(Time, Initial, Parameters):
    
    S=Initial[0]
    E=Initial[1]
    I=Initial[2]
    A=Initial[3]
    Q=Initial[4]
    R=Initial[5]
    
    change = np.zeros([9, 6])
    
    change[0,:]=[-1, +1, 0, 0, 0, 0]
    change[1,:]=[0, -1, +1, 0, 0, 0]
    change[2,:]=[0, -1, 0, +1, 0, 0]
    change[3,:]=[0, 0, -1, 0, 0, +1]
    change[4,:]=[0, 0, -1, 0, +1, 0]
    change[5,:]=[0, 0, -1, 0, 0, 0]
    change[6,:]=[0, 0, 0, -1, 0, +1]
    change[7,:]=[0, 0, 0, 0, -1, +1]
    change[8,:]=[0, 0, 0, 0, -1, +1]
    
    T = np.array([0.])
    #T = np.zeros([5*N,1])
    #pop = np.zeros([5*N,6])
    pop = [S, E, I, A, Q, R]
    old_pop = np.array([S, E, I, A, Q, R])  
    j = 0
    
    while (T[j]<Time[1]):
        [dt, new_pop] = update(old_pop, Parameters, change)
        j = j + 1
        T = np.append(T, [T[j-1]+dt])
        #T[j] = T[j-1]+dt
        #pop[j,:] = new_pop
        pop = np.vstack((pop, new_pop))
        old_pop = new_pop
                 
    return T, pop

if __name__ == "__main__":
        
    print( 'Updated ', dt.datetime.now())
    
    beta = 2.34      # contact rate per person per day
    kappa = 1/5.     # Incubation period = 5 days
    p_s = 0.80       # fraction of symptomatic I
    delta1 = 0.5     # fraction of reduced infectivity of asymptomatic I
    gamma1 = 1/14.   # Recovery time of Is = 14 days
    gamma2 = gamma1   # Recovery time of Ia = 14 days
    mu1 = 0.002      # 0.2% fatality rate per infected person per day
    mu2 = 0.01       # 1% fatality rate of quarantined person  
    
    theta = 0 #1/4.     # if time testing is 4 days, rate of testing and quarantine
    gamma3 = 1/14.   # recovery time in quarantine = 14 days change

    
    
    # initial total population size
    N = 20000
    
    # Intial number of infected individuals
    I0 = 10
    
    E0 = 0
    A0 = 0
    Q0 = 0
    S0 = N - I0
    R0 = N - S0 - I0 - A0- Q0
    
    # Total simulation time, in days
    maxTime = 50
    
    #k = 0
    
    Parameters = [beta, delta1, gamma1, theta, kappa, mu1, p_s, gamma3, mu2, N]
    
    
    [T, pop] = Stoch_Iteration([0, maxTime], [S0, E0, I0, A0, Q0, R0], Parameters)
    
    S = pop[:,0]
    E = pop[:,1]
    I = pop[:,2]
    A = pop[:,3]
    Q = pop[:,4]
    R = pop[:,5]
    
    plt.figure (1)
    plt.plot(T,I, label="Infected Symptomatic")
    plt.plot(T,A, label="Infected Asymptomatic")
    plt.plot(T,Q, label="Quarantined")
    plt.xlabel('Time [days]')
    plt.ylabel('Number of Individuals')
    plt.title('Simulation of SEIAQR Model')
    plt.legend(loc='upper right')
    plt.show()
    
    x=5
    
    

    

    
    
    