import numpy as np
import matplotlib.pyplot as plt

rev_m = 170
rev_stdev = 20
iterations = 1000

# Create a normal distribution using these
revenue = np.random.normal(rev_m, rev_stdev, iterations)
print(revenue)

# Plot it
plt.figure(figsize=(15, 6))
plt.title('Revenue simulation')
plt.plot(revenue)
plt.show()


#now cost of Goods sold
#Generally it is about 60% of the revenue with 10 % SD
#We already have 1000 revenue records, use them to randomly assign COGS in the above range

COGS=(revenue*np.random.normal(0.6,0.1))
plt.figure(figsize=(15,6))
plt.title('COGS')
plt.plot(COGS)
plt.show()



Gross_Profit=revenue-COGS
plt.figure(figsize=(15,6))
plt.title('Gros profit simulation')
plt.plot(Gross_Profit)
plt.show()
#cREATE A STACKD BAR CHART

numbers= list(range(1,1001))

plt.figure(figsize=(10,6))
plt.bar(numbers, revenue, label='Revenue',color='skyblue')
plt.bar(numbers, COGS, bottom=revenue,label='COGS',color='lightcoral')
plt.bar(numbers, Gross_Profit, bottom=[r+c for r,c in zip(revenue,COGS)],label='Gross_Profit',color='lightgreen')

plt.xlabel('Number of times')
plt.ylabel('Amount ($)')
plt.title('Revenue, COGS, And Gross profit using Carlo')
plt.legend()
plt.show()