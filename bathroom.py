import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def min_waiting(n, k, trips_per_day=7, avg_minutes_spent=5, waking_hours=16):
    ''' n people in a house with k bathrooms. 
    lambda_ times each person on average uses the bathroom per minute. 
    mu minutes spent on average in the bathroom per use.
    Returns the number of minutes each person can expect to wait per day
    '''

    # Rate of going into bathroom, underscore as lambda is Python keyword 17/24 represents the hours we are awake
    lambda_ = trips_per_day / waking_hours / 60.0
    mu = 1/avg_minutes_spent # Rate of leaving
    
    # Transition matrix
    Q = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        if i < n:
            Q[i][i+1] = (n-i) * lambda_
        if i > 0:
            Q[i][i-1] = min(k, i) * mu

        Q[i][i] = - sum(Q[i, :])
    
    E = np.ones(shape=(n+1, n+1))
    pi = np.matmul(np.ones(shape=(1, n+1)), np.linalg.inv(Q + E))
    pi = pi.reshape(n+1)
    assert np.all(np.matmul(pi, Q) < 1e-10)

    ppl_minutes = 0
    for ppl_waiting, time in enumerate(pi[k:]):
        ppl_minutes += ppl_waiting * time
    prop_wait = ppl_minutes/n

    min_wait_per_day = prop_wait * waking_hours * 60
    return min_wait_per_day

###########################################################################
# Simulations, plot heatmap
###########################################################################
MAX_PEOPLE = 15
MAX_BATHROOM = 3

results = pd.DataFrame(0.0, index=range(MAX_PEOPLE, 1, -1), 
	columns=range(1, MAX_BATHROOM + 1))
results.index.name = 'Num_people'

for num_people in range(2, MAX_PEOPLE + 1):
    for num_bathroom in range(1, min(num_people, MAX_BATHROOM + 1)):
        results[num_bathroom][num_people] = min_waiting(num_people, num_bathroom)

print('Results calculated')
sns.heatmap(results, annot=True, annot_kws={"size": 15})
plt.xlabel('Number of bathrooms')
plt.ylabel('Number of people')
plt.title('Expected minutes per day waiting to use the bathroom')
plt.show()

plt.savefig('images/comparison.pdf')

