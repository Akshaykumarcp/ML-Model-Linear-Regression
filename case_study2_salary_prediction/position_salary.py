""" 
Summary of the polynominal regression program:
1. Read dataset 
2. observe top 5 data points
3. split the dataset into independent and dependent feature/variable
4. fit linear regression model with polynomial degree of 1 (default i,e linear regression)
5. fit linear regression model with polynomial degree of 2 and 4
6. plot the linear regressions fit to observe how model has fitted on data
"""

# import lib's
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

# import the dataset
dataset= pd.read_csv('case_study2_position_salary/Position_Salaries.csv') # the full namespace of the file can be provided if the file is not in the same directory as the .ipynb or.py file

dataset.head()   # to see how the imported data looks like
""" 
            Position  Level  Salary
0   Business Analyst      1   45000
1  Junior Consultant      2   50000
2  Senior Consultant      3   60000
3            Manager      4   80000
4    Country Manager      5  110000

- Here, it can be seen that there are 3 columns in the dataset. 
- The problem statement here is to predict the salary based on the Position and Level of the employee. 
- But we may observe that the Position and the level are related or level is one other way of conveying the position 
        of the employee in the company. So, essentially Position and Level are conveying the same kind of information.
        As Level is a numeric column, 
        let's use that in our Machine Learning Model. Hence, Level is our feature or X variable. 
        And, Salary is Label or the Y variable """

x=dataset.iloc[:,1:2].values

# x=dataset.iloc[:,1].values
# this is written in this way to make x as a matrix as the machine learning algorithm.
# if we write 'x=dataset.iloc[:,1].values', it will return x as a single-dimensional array which is not desired 
x
""" array([[ 1],
       [ 2],
       [ 3],
       [ 4],
       [ 5],
       [ 6],
       [ 7],
       [ 8],
       [ 9],
       [10]], dtype=int64) """

y=dataset.iloc[:,2].values
y
""" array([  45000,   50000,   60000,   80000,  110000,  150000,  200000,
        300000,  500000, 1000000], dtype=int64) """

""" 
Generally, we divide our dataset into two parts
1) The training dataset to train our model. And, 
2) The test dataset to test our prepared model. 

- Here, as the dataset has a limited number of entries, we won't do a split. 
- Instead of that, we'd use direct numerical values to test the model. 
- Hence, the code above is kept commented. But, train test split can also be done, if you desire so:)

- To learn Polynomial Regression, we'd follow a comparative approach. 
- First, we'll try to create a Linear Model using Linear Regression and then we'd prepare a Polynomial 
                Regression Model and see how do they compare to each other
 """

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
# LinearRegression()

# Visualising the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

""" 
- Here, the red dots are the actual data points and, the blue straight line is what our model has created. 
- It is evident from the diagram above that a Linear model does not fit our dataset well.
- So, let's try with a Polynomial Model.
 """

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)  # trying to create a 2 degree polynomial equation. It simply squares the x as shown in the output
X_poly = poly_reg.fit_transform(x)

print(X_poly)
""" 
[[  1.   1.   1.]
 [  1.   2.   4.]
 [  1.   3.   9.]
 [  1.   4.  16.]
 [  1.   5.  25.]
 [  1.   6.  36.]
 [  1.   7.  49.]
 [  1.   8.  64.]
 [  1.   9.  81.]
 [  1.  10. 100.]] """

poly_reg.fit(X_poly, y)
# PolynomialFeatures()

# doing the actual polynomial Regression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
LinearRegression()

""" 
- It can be noted here that for Polynomial Regression also, we are using the Linear Regression Object.

Why is it so?
- It is because the Linear in Linear Regression does not talk about the degree of the Polynomial equation in terms 
                of the dependent variable(x). 
                Instead, it talks about the degree of the coefficients. Mathematically,
                y=a+bx+cx2+...+nxn+...

- It's not talking about the power of x, but the powers of a,b,c etc. 
- And as the coefficients are only of degree 1, hence the name Linear Regression.
 """

# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

""" 
- Still, a two degree equation is also not a good fit. 
- Now, we'll try to increase the degree of the equation i.e. we'll try to see that whether we get a good fit at a 
                higher degree or not. 
                After some hit and trial, we see that the model get's the best fit for a 4th degree polynomial equation.
 """

# Fitting Polynomial Regression to the dataset
poly_reg1 = PolynomialFeatures(degree = 4)
X_poly1 = poly_reg1.fit_transform(x)
poly_reg1.fit(X_poly, y)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly1, y)
# LinearRegression()

# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_3.predict(poly_reg1.fit_transform(x)), color = 'blue')
plt.title('Polynomial Regression of Degree 4')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

""" 
- Here, we can see that our model now accurately fits the dataset. 
- This kind of a fit might not be the case with the actual business datasets. 
- we are getting a brilliant fit as the number of datapoints are a few. """