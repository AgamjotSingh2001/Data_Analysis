import numpy as np
import matplotlib.pyplot as plt

def roll_dice():
    return np.sum(np.random.randint(1,7,2))

print(roll_dice())

def monte_carlo_simulation(runs=1000):
    result=np.zeros(2)
    for _ in range(runs):
        if roll_dice() == 7:
            result[0] += 1
        else:
            result[1] += 1
    return result
    

#Test 2-3 times and calculate how much you will win    
print(monte_carlo_simulation())

print(monte_carlo_simulation())

print(monte_carlo_simulation())

#Part 3
#Now do it 1000 times
#Takes some time

results=np.zeros(1000)
for i in range(1000):
    results[i]=monte_carlo_simulation()[0]
print(results)

#Ploting it on graph
fig, ax = plt.subplots()
ax.hist(results, bins=15)
plt.show()

#Our win/loss
print(results.mean())           #Genral mean
print(results.mean()*5)         #What we will get as win on an average
print(1000-results.mean())      #What we will pay on an average
print(results.mean()/1000)      #Probability of the 'We will pay' result
#The last probability should be close to the theoretical probability of getting  a 7
#  a 7 when we throw two dice