import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from pandas import DataFrame
import seaborn as sns



df = pd.read_csv("housing2.csv")
df=df.dropna()
df.head()
df.describe()
df.info()



# Plot heatmap of correlation matrix
corr= df.corr()
sns.heatmap(corr, annot=False)
plt.yticks(rotation=0); plt.xticks(rotation=90)  # fix ticklabel directions
plt.tight_layout()  # fits plot area to the plot, "tightly"
plt.show()  # show the plot
plt.clf()  # clear the plot area



# Create a scatter plot of the most highly correlated variable with the target
cols = ['ZN','CHAS','RM','DIS','TAX','MEDV']
sns.pairplot(df[cols],size=2.5)
plt.tight_layout()
plt.show()



#Linear Model
y = df['MEDV']
X = df.drop('MEDV', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)


# Coefficient and Intercept
print("Coefficient: ",reg_all.coef_)
print("Intercept: ",reg_all.intercept_)
# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))

plt.scatter(y_pred, y_pred-y_test, c='limegreen', marker='s',label='Test data')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.legend(loc='upper left')
plt.xlabel('Residual plot for Linear Regression')
plt.show()



def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()




#Ridge Regression
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

y = df['MEDV']
X = df.drop('MEDV', axis=1)

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

   # Specify the alpha value to use: ridge.alpha
   ridge.alpha = alpha
   
   # Perform 10-fold CV: ridge_cv_scores
   ridge_cv_scores = cross_val_score(ridge,X,y,cv=10)
   
   # Append the mean of ridge_cv_scores to ridge_scores
   ridge_scores.append(np.mean(ridge_cv_scores))
   
   # Append the std of ridge_cv_scores to ridge_scores_std
   ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)
index=ridge_scores.index(np.max(ridge_scores))
best_alpha = alpha_space[index]
print('best alpha for ridge regression:',alpha_space[index])


ridge= Ridge(alpha=best_alpha, normalize = True)
ridge.fit(X,y)
y_pred = ridge.predict(X_test)

ridge.coef=ridge.fit(X,y).coef_
print('Coefficients of ridge regression:',ridge.coef)
print("The intercept of ridge regression: ",ridge.intercept_)
# Compute and print R^2 and RMSE
print("R^2: {}".format(ridge.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))



#Plot Residual Error
plt.scatter(y_pred, y_pred-y_test, c='steelblue', marker='s',label='Test data')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.legend(loc='upper left')
plt.xlabel('residual error of Ridge regression')
plt.show()


# Import Lasso
from sklearn.linear_model import Lasso

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
lasso_scores = []
lasso_scores_std = []

# Create a ridge regressor: ridge
Lasso = Lasso(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    lasso.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    lasso_cv_scores = cross_val_score(lasso,X,y,cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    lasso_scores.append(np.mean(lasso_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    lasso_scores_std.append(np.std(lasso_cv_scores))

# Display the plot
display_plot(lasso_scores, lasso_scores_std)

#Find optimal alpha 
index=lasso_scores.index(np.max(lasso_scores))
best_alpha2 = alpha_space[index]
print('best alpha for alpha regression:',alpha_space[index])



# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=best_alpha2, normalize = True)

# Fit the regressor to the data
lasso.fit(X,y)

y_pred = lasso.predict(X_test)

lasso.coef=lasso.fit(X,y).coef_
print('Coefficients of lasso regression:',lasso.coef)
print("The intercept of lasso regression: ",lasso.intercept_)
# Compute and print R^2 and RMSE
print("R^2: {}".format(lasso.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))



#Plot Residual Error
plt.scatter(y_pred, y_pred-y_test, c='steelblue', marker='s',label='Test data')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.legend(loc='upper left')
plt.xlabel('residual error of Lasso regression')
plt.show()
