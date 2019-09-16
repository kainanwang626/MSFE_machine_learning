import pandas as pd
import pylab
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
df = pd.read_csv("HY_Universe_corporate bond.csv")
df2 = df.sample(n=200)

#data
coupon=df2.iloc[:,9]
bond_type=df2.iloc[:,29].astype(int)
bloomberg_rating=df2.iloc[:,8]
percent_intra_dealer=df2.iloc[:,27].astype(float)
percent_uncapped=df2.iloc[:,28].astype(float)
Client_Trade_Percentage=df2.iloc[:,30].astype(float)

#number of rows
print(df.shape[0])
#number of columns
print(df.shape[1])

#probability plot
stats.probplot(percent_intra_dealer,dist="norm",plot=pylab)
pylab.show()

#data summary
print(df.head())
print(df.tail())
print(df.describe())

#scatter plot
plt.scatter(percent_intra_dealer, percent_uncapped,alpha=0.5)
plt.xlabel("percent intra dealer")
plt.ylabel("percent uncapped")
plt.show()

#histogram
n_data=len(percent_intra_dealer)
nbins=int(np.sqrt(n_data))
plt.hist(percent_intra_dealer,bins=nbins)
plt.show()

#swarmplot
_ =sns.swarmplot(bond_type, Client_Trade_Percentage)
_ = plt.xlabel('bond type')
_ = plt.ylabel('Client Trade Percentage')
# Show the plot
plt.show()

# ECDF
def ecdf(data):
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

x_cli, y_cli = ecdf(Client_Trade_Percentage)
x_dea, y_dea = ecdf(percent_intra_dealer)

# Generate plot
_ =plt.plot(x_cli,y_cli,marker='.', linestyle ='none')
_ =plt.plot(x_dea,y_dea,marker='.', linestyle ='none')

# Label the axes
plt.legend(('Client Trade Percentage', 'percent intra dealer'), loc='lower right')
_ = plt.ylabel('ECDF')


# Display the plot
plt.show()

# Create box plot with Seaborn's default settings
_ =sns.boxplot(bond_type,Client_Trade_Percentage)

# Label the axes
_ = plt.xlabel('bond type')
_ = plt.ylabel('Client trade percentage')


# Show the plot
plt.show()

#heat map
from pandas import DataFrame
corMat = DataFrame(df.corr())
#visualize correlations using heatmap
plt.pcolor(corMat)
plt.show()

print("My name is {type your name here}")
print("My NetID is: {type your NetID here}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
