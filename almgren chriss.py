# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:10:15 2022

@author: Wu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

pd.options.mode.chained_assignment = None  # default='warn'

data = pd.read_csv('CompanyX.csv', index_col='Date',parse_dates=True)

train = data.iloc[:-5]
test = data.iloc[-5:]

adv = train.iloc[-30:].median()['PX_VOLUME'] # Median daily trading volume

bid_ask_spread = train.iloc[-1]['AVERAGE_BID_ASK_SPREAD']

train['Ret'] = train['PX_LAST'].diff().ewm(36).mean()
train['Vol'] = train['PX_LAST'].diff().ewm(36).std()

alpha = train.iloc[-1]['Vol']  # daily dollar change/growth
sigma = train.iloc[-1]['Vol']  # daily dollar volatility

X = 2000000  # shares, initial holdings

T = 5 # Liquidation time
N = 5 # interval
period = np.arange(1,N+1)
tau = T/N

epsilon =  0.5 * bid_ask_spread # fixed portion of temporary impact, such as half bid-ask spread plus fees

eta = bid_ask_spread/(0.01*adv) # quadratic portion of temporary impact, fluctuate with squared unit traded

gamma = bid_ask_spread/(0.1*adv) # permanent impact

eta_tilde = eta - (0.5*gamma*tau)

x=[]
y=[]
U = []

# Minimize E(x)+lambda*V(x)

def objective(nk, lam, sigma, eta, gamma, epsilon, X, eta_tilde, tau, alpha):
    nk = nk * X
    xk = [np.sum(nk[x+1:]) for x,_ in enumerate(nk)]
    E = (0.5*gamma*(X**2)) + (epsilon*np.sum(np.abs(nk))) + (eta_tilde/tau)*np.sum(np.square(nk))   #-(alpha*tau*np.sum(xk))
    V = (sigma**2) * tau * np.sum(np.square(xk))
    return E + (lam*V)

print('Minimize E(x)+lambda*V(x)')
for l in [-2E-6, 2E-6]:
    n0 = [1/T] * T
    cons = ({'type': 'eq', 'fun': lambda n0: np.sum(n0) - 1  })
    bnds = [(0, 1)] * T
    best_mix = spo.minimize(objective, n0, args=(l, sigma, eta, gamma, epsilon, X, eta_tilde, tau,alpha), method='SLSQP', constraints = cons, options={'disp':False})  
    print('Trading trajectory for risk parameter of',l,': ',best_mix.x*X)
    plt.plot(np.arange(1,T+1), X*best_mix.x, label=l)

plt.legend()
plt.title('Trade trajectory over 5 days')
plt.xlabel('Days')
plt.ylabel('Number of shares traded')

# Better still, use closed form solution

lam = 2E-6
k = (lam*(sigma**2)/eta)**0.5
xa = alpha/(2*lam*(sigma**2))  # optimal level of security holding
n_j = []
x_j = []

for j in period:
    tj = j*tau
    tj_half = (j-0.5)*tau
    xj = X*np.sinh(k*(T-tj))/np.sinh(k*T)
    nj = 2 * np.sinh(0.5*k*tau) * X * np.cosh(k*(T-tj_half)) / np.sinh(k*T)
    
    # With drift
    #xj += xa * (1-( (np.sinh(k*(T-tj))+np.sinh(k*tj))/np.sinh(k*T)  ))
    #nj += xa * (2*np.sinh(0.5*k*tau)/np.sinh(k*T)) * (np.cosh(k*tj_half)-np.cosh(k*(T-tj_half)))
    
    n_j.append(np.round(nj))
    x_j.append(xj)

print('\nOptimal trajectory using closed form solution: ',n_j)

# Expectation and variance of the optimal strategy
plt.figure()
x,y = [],[]
for lam in np.arange(10**-7,10**-4,10**-6):
    k = (lam*(sigma**2)/eta)**0.5
    E = (0.5*gamma*(X**2)) + (epsilon*X) + (eta_tilde * (X**2)*np.tanh(0.5*k*tau)*((tau*np.sinh(2*k*T))+(2*T*np.sinh(k*tau)))/(2*(tau**2)*(np.sinh(k*T)**2)))
    V = 0.5* (sigma**2)*(X**2)*((tau*np.sinh(k*T)*np.cosh(k*(T-tau)))-(T*np.sinh(k*tau)))/((np.sinh(k*T)**2)*np.sinh(k*tau))
    x.append(E)
    y.append(V)

plt.plot(x,y)
plt.title('Efficient frontier')
plt.xlabel('Variance V(x) $^2')
plt.ylabel('Expected loss E(x) $')