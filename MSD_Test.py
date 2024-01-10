import pandas as pd 
import numpy as np 
import datetime 
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv('MS_Dhoni_ODI_record.csv')

# Basic checks
print(df.head())
print(df.tail())

# Data cleaning - Opposition name says 'v Aus' etc, we can remove 'v '
df['opposition'] = df['opposition'].apply(lambda x: x[2:])

# Add a 'feature' - 'year' column using the match date column
# First convert date column into datetime format
df['date'] = pd.to_datetime(df['date'], dayfirst=True) 
df['year'] = df['date'].dt.year.astype(int)
#print(df.head())

# Create a column to distinguish between out and not out

# The apply method in Pandas allows you to apply a function to each element in a DataFrame or Series. In this case, the function being applied is str, which is the built-in Python function for converting values into strings. By applying str to each element in the 'score' column, we are converting the numerical or other data types in that column into string data types.
df['score'] = df['score'].apply(str) 
df['not_out'] = np.where(df['score'].str.endswith('*'), 1, 0)

# dropping the odi_number feature because it adds no value to the analysis
df.drop(columns='odi_number', inplace=True)

# dropping those innings where Dhoni did not bat and storing in a new DataFrame 
# Take all the columns, starting with runs_scored
df_new = df.loc[((df['score'] != 'DNB') & (df['score'] != 'TDNB')), 'runs_scored':]
#print(df_new.head())

# fixing the data types of numerical columns 
df_new['runs_scored'] = df_new['runs_scored'].astype(int)
df_new['balls_faced'] = df_new['balls_faced'].astype(int) 
df_new['strike_rate'] = df_new['strike_rate'].astype(float) 
df_new['fours'] = df_new['fours'].astype(int) 
df_new['sixes'] = df_new['sixes'].astype(int)

# Career stats
first_match_date = df['date'].dt.date.min().strftime('%B %d, %Y') # first match
print('First match:', first_match_date)
last_match_date = df['date'].dt.date.max().strftime('%B %d, %Y') # last match
print('Last match:', last_match_date)
number_of_matches = df.shape[0] # number of mathces played in career
print('Number of matches played:', number_of_matches)
number_of_inns = df_new.shape[0] # number of innings
print('Number of innings played:', number_of_inns)
not_outs = df_new['not_out'].sum() # number of not outs in career
print('Not outs:', not_outs)
runs_scored = df_new['runs_scored'].sum() # runs scored in career
print('Runs scored in career:', runs_scored)
balls_faced = df_new['balls_faced'].sum() # balls faced in career
print('Balls faced in career:', balls_faced)
career_sr = (runs_scored / balls_faced)*100 # career strike rate
print('Career strike rate: {:.2f}'.format(career_sr))
career_avg = (runs_scored / (number_of_inns - not_outs)) # career average
print('Career average: {:.2f}'.format(career_avg))
highest_score_date = df_new.loc[df_new.runs_scored == df_new.runs_scored.max(), 'date'].values[0]
highest_score = df.loc[df.date == highest_score_date, 'score'].values[0] # highest score
print('Highest score in career:', highest_score)
hundreds = df_new.loc[df_new['runs_scored'] >= 100].shape[0] # number of 100s
print('Number of 100s:', hundreds)
fifties = df_new.loc[(df_new['runs_scored']>=50)&(df_new['runs_scored']<100)].shape[0] #number of 50s
print('Number of 50s:', fifties)
fours = df_new['fours'].sum() # number of fours in career
print('Number of 4s:', fours)
sixes = df_new['sixes'].sum() # number of sixes in career
print('Number of 6s:', sixes)

# number of matches played against different oppositions
df['opposition'].value_counts().plot(kind='bar', title='Number of matches against different oppositions', figsize=(8, 5));
plt.show()

# Runs scored against each team
runs_scored_by_opposition = pd.DataFrame(df_new.groupby('opposition')['runs_scored'].sum())
runs_scored_by_opposition.plot(kind='bar', title='Runs scored against different oppositions', figsize=(8, 5))
plt.xlabel(None);
plt.show()

# Batting average against each team

# This line groups the DataFrame df_new by the "opposition" column and counts the number of entries (innings) for each unique value in the "opposition" column. It creates a new DataFrame innings_by_opposition with the opposition team names as the index and the number of innings as a column.
innings_by_opposition = pd.DataFrame(df_new.groupby('opposition')['date'].count())
print(innings_by_opposition)

# Similarly, this line groups the DataFrame by "opposition" and sums up the "not_out" column for each opposition team. It creates a new DataFrame not_outs_by_opposition with the opposition team names as the index and the total number of not-outs as a column.
not_outs_by_opposition = pd.DataFrame(df_new.groupby('opposition')['not_out'].sum())

# Here, we are merging two DataFrames, runs_scored_by_opposition and innings_by_opposition, based on their indices (opposition team names) to combine the runs scored and the number of innings for each opposition team into a temporary DataFrame temp.
temp = runs_scored_by_opposition.merge(innings_by_opposition, left_index=True, right_index=True)
print(temp)

average_by_opposition = temp.merge(not_outs_by_opposition, left_index=True, right_index=True)
print(average_by_opposition)

# This line renames the "date" column to "innings" in the average_by_opposition DataFrame.
average_by_opposition.rename(columns = {'date': 'innings'}, inplace=True)

# We are calculating the effective number of innings by subtracting the total not-outs from the total innings and adding this as a new column "eff_num_of_inns" in the DataFrame.
average_by_opposition['eff_num_of_inns'] = average_by_opposition['innings'] - average_by_opposition['not_out']

# We are calculating the batting average for each opposition team by dividing the total runs scored by the effective number of innings and adding this as a new column "average" in the DataFrame.
average_by_opposition['average'] = average_by_opposition['runs_scored'] / average_by_opposition['eff_num_of_inns']

# In this line, we are replacing any occurrences of positive infinity (np.inf) in the DataFrame with NaN (Not a Number). This step is likely done to handle cases where a player has not been dismissed (division by zero), resulting in an infinite average.
average_by_opposition.replace(np.inf, np.nan, inplace=True)

# This line defines a list of major cricketing nations.
major_nations = ['Australia', 'England', 'New Zealand', 'Pakistan', 'South Africa', 'Sri Lanka', 'West Indies']

# creates a new figure for the plot with a specified figure size of 8 inches in width and 5 inches in height.
plt.figure(figsize = (8, 5))

# plots the batting averages against major cricketing nations. It uses the plot function to plot the values from the "average" column of the average_by_opposition DataFrame for the specified major_nations. The marker='o' argument specifies that data points should be marked with circles.
plt.plot(average_by_opposition.loc[major_nations, 'average'].values, marker='o')

# Plots a dashed horizontal line representing the career average. [career_avg]*len(major_nations) creates a list of the career average repeated for each major nation, and -- specifies that the line should be dashed.
plt.plot([career_avg]*len(major_nations), '--')

plt.title('Average against major teams')

# Sets the x-axis tick positions and labels. It maps the positions 0 to 6 (since there are 7 major nations) to the labels in the major_nations list.
plt.xticks(range(0, 7), major_nations)

# Sets the y-axis limits, restricting the range of the y-axis between 20 and 70.
plt.ylim(20, 70)

# Adds a legend to the plot, labeling the two lines as "Avg against opposition" and "Career average."
plt.legend(['Avg against opposition', 'Career average'])

plt.show()

# Year-wise record
df['year'].value_counts().sort_index().plot(kind='bar', title='Matches played by year', figsize=(8, 5))
#plt.xticks(rotation=0)
plt.xticks(rotation=50)
plt.show()

# Runs scored year-wise
df_new.groupby('year')['runs_scored'].sum().plot(kind='line', marker='o', title='Runs scored by year', figsize=(8, 5))
years = df['year'].unique().tolist()
plt.xticks(years)
plt.xlabel(None)
plt.show()



