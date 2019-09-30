import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from pandas import DataFrame
import seaborn as sns

df = pd.read_csv("hw5_treasury_yield.csv")
df = df.dropna()
df = df.iloc[:,1:32]
df.head()

# Plot heatmap of correlation matrix
corr= df.corr()
sns.heatmap(corr, annot=False)
plt.yticks(rotation=0); plt.xticks(rotation=90)  # fix ticklabel directions
plt.tight_layout()  # fits plot area to the plot, "tightly"
plt.show()  # show the plot
plt.clf()  # clear the plot area

cols = ['SVENF01','SVENF03','SVENF05','Adj_Close']
sns.pairplot(df[cols],height=2.5)
plt.tight_layout()
plt.show()

# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,pca)

# Fit the pipeline to 'samples'
pipeline.fit(df)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_)
plt.xlabel('PCA feature')
plt.ylabel('variance ratio')
plt.xticks(features)
plt.show()

print("the explained variances ratios are: ",pca.explained_variance_ratio_)
cum_var_exp=np.cumsum(pca.explained_variance_)
plt.step(features,cum_var_exp,where='mid')
plt.show()
print("the cumulativie variances are: ",cum_var_exp)

#PCA of n=3 components

# Create a PCA model with 3 components: pca
pca = PCA(n_components=3)

# Fit the PCA instance to the scaled samples
pca.fit(df)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,pca)

# Fit the pipeline to 'samples'
pipeline.fit(df)


# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_)
plt.xlabel('PCA feature')
plt.ylabel('variance ratio')
plt.xticks(features)
plt.show()


print("the explained variances ratio are: ",pca.explained_variance_ratio_)

cum_var_exp=np.cumsum(pca.explained_variance_)
plt.step(features,cum_var_exp,where='mid')
plt.show()
print("the cumulativie variances are: ",cum_var_exp)

#train test split
y = df['Adj_Close']
X = df.drop('Adj_Close', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=42)

# Linear Model

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred_train = reg_all.predict(X_train)
y_pred_test = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2 for train set: {}".format(reg_all.score(X_train, y_train)))
print("R^2 for test set: {}".format(reg_all.score(X_test, y_test)))
rmse_test = np.sqrt(mean_squared_error(y_test,y_pred_test))
rmse_train = np.sqrt(mean_squared_error(y_train,y_pred_train))
print("Root Mean Squared Error for train set: {}".format(rmse_train))
print("Root Mean Squared Error for test set: {}".format(rmse_test))

