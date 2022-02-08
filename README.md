## Table of content

1. What is regression analysis ?
2. Usage of regression
3. Linear regression
4. Simple linear regression
5. Regularization
6. Linear regression using loss minimization
7. R2 statistics
8. Prediction using model
9. Model confidence
10. Feature selection
11. Polynomial Regression
12. Various cases of linear regression
    - 12.1 Imbalanced dataset in linear regression
    - 12.2 Feature Importance and interpretability in linear regression
    - 12.3 Outliers in linear regression
13. Acknowledgements
14. Connect with me
---

## 1. What is Regression Analysis?
- Regression in statistics is the process of predicting a Label(or Dependent Variable) based on the features(Independent Variables) at hand. 
- Regression is used for time series modelling and finding the causal effect relationship between the variables and forecasting. 
- For example, the relationship between the stock prices of the company and various factors like customer reputation and company annual performance etc. can be studied using regression.
- Regression analysis is an important tool for analysing and modelling data. 
- Here, we fit a curve/line to the data points, in such a manner that the differences between the distance of the actual data points from the plotted curve/line is minimum. 


## 2. Usage of Regression
- Regression analyses the relationship between two or more features. 

Let’s take an example:

Let’s suppose we want to make an application which predicts the chances of admission a student to a foreign university. In that case, the

The benefits of using Regression analysis are as follows:

- It shows the significant relationships between the Lable (dependent variable) and the features(independent variable).
- It shows the extent of the impact of multiple independent variables on the dependent variable.
- It can also measure these effects even if the variables are on a different scale.
- These features enable the data scientists to find the best set of independent variables for predictions.

## 3. Linear Regression

Building blocks of a Linear Regression Model are:
- Discreet/continuous independent variables
- A best-fit regression line

Continuous dependent variable. i.e., A Linear Regression model predicts the dependent variable using a regression line based on the independent variables. The equation of the Linear Regression is:

```
   Y = a + b * X + e

Where, 
a is the intercept
b is the slope of the line
X is the input and 
e is the error term. 

The equation above is used to predict the value of the target variable based on the given predictor variable(s)
```

- Linear regression can be derived from same ideas of logistic regression with Naive bayes assumption and assumption of yi (target feature) is gaussian distributed.
- Linear regression is sister of Generalized Linear Models

In geometric interpretation, Optimization problem of  linear regression is shown below:

![OP](https://eed3si9n.com/images/ml2.jpg)
[image source](https://eed3si9n.com/images/ml2.jpg)

Linear Regression often reffered as Ordinary least squares (OLS) or Linear least squares (LLS)

## 4. Simple Linear Regression
- Simple Linear regression is a method for predicting a quantitative response using a single feature ("input variable"). 

The mathematical equation is:
```
y = β0 + β1 x

y is the response or the target variable
x is the feature
β1 is the coefficient of x
β0 is the intercept
β0 and β1 are the model coefficients. 
```
To create a model, we must "learn" the values of these coefficients. And once we have the value of these coefficients, we can use the model to predict!

#### Estimating ("Learning") Model Coefficients
- The coefficients are estimated using the least-squares criterion, i.e., the best fit line has to be calculated that minimizes the sum of squared residuals (or "sum of squared errors").

#### Let’s see the underlying assumptions: -

- The regression model is linear in terms of coefficients and error term.
- The mean of the residuals is zero.
- The error terms are not correlated with each other, i.e. given an error value; we cannot predict the next error value.
- The independent variables(x) are uncorrelated with the residual term, also termed as exogeneity. This, in layman term, generalises that in no way should the error term be predicted given the value of independent variables.
- The error terms have a constant variance, i.e. homoscedasticity.
- No Multicollinearity, i.e. no independent variables should be correlated with each other or affect one another. If there is multicollinearity, the precision of prediction by the OLS model decreases.
- The error terms are normally distributed.
- The general equation of a straight line is:
    ```
    y=mx+b
    ```
    It means that if we have the value of m and b, we can predict all the values of y for corresponding x. During construction of a Linear Regression Model, the computer tries to calculate the values of m and b to get a straight line.

## 5. Regularization

- we can use L2 or L1 regularization or elastic net.
- Similar to logisic regression, only change is squared loss.

## 6. Linear regression using Loss minimization
Loss minimization interpretation:

- Loss function as logistic loss gives Logistic regression
- Loss function as hinge loss gives SVM
- Loss function as exponential loss gives Adaboost
- Loss function as squared loss gives Linear regression

## 7. R2  statistics
- The R-squared statistic provides a measure of fit. 
- It takes the form of a proportion—the proportion of variance explained—and so it always takes on a value between 0 and 1. 
- In simple words, it represents how much of our data is being explained by our model. 
- For example, R2 statistic = 0.75, it says that our model fits 75 % of the total data set. 
- Similarly, if it is 0, it means none of the data points is being explained and a value of 1 represents 100% data explanation.
- The closer the value of R2 is to 1 the better the model fits our data. 
- If R2 comes below 0 (ex: -1) that means the model is so bad that it is performing even worse than the average best fit line.
- Google for r square formula.

#### Issue with R2
- As we increase the number of independent variables in our equation, the R2 increases as well. 
- But that doesn’t mean that the new independent. variables have any correlation with the output variable. 
- In other words, even with the addition of new features in our model, it is not necessary that our model will yield better results but R2 value will increase.
- R2 always increases with an increase in the number of independent variables. 
- Thus, it doesn’t give a better picture and so we need Adjusted R2 value to keep this in check.
- we use Adjusted R2 value which penalises excessive use of such features which do not correlate with the output data.
- Google for adjusted r square formula.
-  In adjusted r square formula, when p = 0, adjusted R2 becomes equal to R2. 
- Thus, adjusted R2 will always be less than or equal to R2, and it penalises the excess of independent variables which do not affect the dependent variable.

## 8. Prediction using the model

If the expense on TV ad is $50000, what will be the sales prediction for that market?
```
y=β0+β1x
```
y= 7.032594 + 0.047537 × 50

## calculate the prediction
```
7.032594 + 0.047537 * 50

9.409444 # y i,e response
```

## 9. Model Confidence

**Question:** Is linear regression a low bias/high variance model or a high bias/low variance model?

**Answer:** 
- It's a High bias/low variance model. 
- Even after repeated sampling, the best fit line will stay roughly in the same position (low variance), but the average of the models created after repeated sampling won't do a great job in capturing the perfect relationship (high bias). 
- Low variance is helpful when we don't have less training data!

## 10. Feature Selection

#### How do I decide which features have to be included in a linear model? Here's one idea:

- Try different models, and only keep predictors in the model if they have small p-values.
- Check if the R-squared value goes up when you add new predictors to the model.

#### What are the drawbacks in this approach? 
- If the underlying assumptions for creating a Linear model(the features being independent) are violated(which usually is the case),p-values and R-squared values are less reliable.

- Using a p-value cutoff of 0.05 means that adding 100 predictors to a model that are pure noise, still 5 of them (on average) will be counted as significant.
- R-squared is susceptible to model overfitting, and thus there is no guarantee that a model with a high R-squared value will generalise. 
- Selecting the model with the highest value of R-squared is not a correct approach as the value of R-squared shall always increase whenever a new feature is taken for consideration even if the feature is unrelated to the response.
- The alternative is to use adjusted R-squared which penalises the model complexity (to control overfitting), but this again generally under-penalizes complexity.

- A better approach to feature selection isCross-validation. 
- It provides a more reliable way to choose which of the created models will best generalise as it better estimates of out-of-sample error. 
- An advantage is that the cross-validation method can be applied to any machine learning model and the scikit-learn package provides extensive functionality for that.

## 11. Polynomial Regression

- For understanding Polynomial Regression, let's first understand a polynomial. 
- Merriam-webster defines a polynomial as: "A mathematical expression of one or more algebraic terms each of which consists of a constant multiplied by one or more variables raised to a non-negative integral power (such as a + bx + cx^2)". 
- Simply said, poly means many. So, a polynomial is an aggregation of many monomials(or Variables). A simple polynomial equation can be written as:
    ```
    y = a + b x + c x2 +...+ nxn +...
    ```
- So, Polynomial Regression can be defined as a mechanism to predict a dependent variable based on the polynomial relationship with the independent variable.

In the equation,
```
y = a + b x + c x2 +...+ nxn +...
```
- the maximum power of 'x' is called the degree of the polynomial equation. 
- For example, if the degree is 1, the equation becomes
    ```
    y=a+bx
    ```
    which is a simple linear equation. 
    
    if the degree is 2, the equation becomes
    ```
    y=a+bx+cx2
    ```
    which is a quadratic equation and so on.

#### When to use Polynomial Regression?
- Many times we may face a requirement where we have to do a regression, but when we plot a graph between a dependent and independent variables, the graph doesn't turn out to be a linear one. 
- A linear graph typically looks like: https://images.app.goo.gl/KmQbCxi84g6RzwkE9
- But what if the relationship looks like: https://images.app.goo.gl/cfjFpC9MjTcFX1m58

- It means that the relationship between X and Y can't be described Linearly. Then comes the time to use the Polynomial Regression.

## 12. Various cases of linear regression

#### 12.1 Imbalanced dataset in linear regression
- Perform up/down sampling

#### 12.2 Feature Importance and interpretability in linear regression
- When features are not multicollinear, then we can use simple feature weights {wj}
- Top values of w weight vector

#### 12.3 Outliers in linear regression

- In logistic regression, sigmoid limited the impact of outlier
- In linear regression, squared loss is being used. Squared loss can be impacted by outliers.

    ##### **How to handle outliers ?**
    - Use all training dataset for finding out:
        - optimal w* and w in optimization problem
        - find the points/observations that are far away from the plain (pi)
    - Remove the points as outliers
    - Create new dataset
    - Repeat the process!
    - This technique is called as RANSAC in statistics


## 13. Acknowledgements :handshake:
- [Google Images](https://www.google.co.in/imghp?hl=en-GB&tab=ri&authuser=0&ogbl)
- [Ineuron](https://ineuron.ai/)
- [Appliedai](https://www.appliedaicourse.com/)
- Other google sites

## 14. Connect with me  :smiley:
[<img align="left" alt="" width="22px" src="https://simpleicons.org/icons/linkedin.svg" />](https://www.linkedin.com/in/akshay-kumar-c-p/)
[<img align="left" alt="" width="22px" src="https://simpleicons.org/icons/youtube.svg" />](https://www.youtube.com/channel/UC3l8RTE3zBRzUrHbSXpx-qA)
[<img align="left" alt="" width="22px" src="https://simpleicons.org/icons/github.svg" />](https://github.com/Akshaykumarcp)
[<img align="left" alt="" width="22px" src="https://simpleicons.org/icons/medium.svg" />](https://medium.com/@akshai.148)
