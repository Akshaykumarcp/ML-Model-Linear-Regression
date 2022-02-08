""" 
Summary of the linear regression program:
1. Read dataset
2. Basic EDA
    2.1 observe top 5 data points
    2.2 describing dataset
    2.3 check for null values and null values imputation
    2.4 drop unecessary columns
    2.5 plot distribution for dependent and independent features/variables 
    2.6 features transformation
3. check for multi-collinearity
4. model fitting 
5. test model using r2 and adjusted r2
6. check for model overfitting in train data and test data using:
    6.1 lasso 
    6.2 ridge
    6.3 elasticnet
    """
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data =pd.read_csv('case_study1_admission_prediction/Admission_Prediction.csv')

data.head()
""" 
   Serial No.  GRE Score  TOEFL Score  University Rating  SOP  LOR  CGPA  Research  Chance of Admit
0           1      337.0        118.0                4.0  4.5  4.5  9.65         1             0.92
1           2      324.0        107.0                4.0  4.0  4.5  8.87         1             0.76
2           3        NaN        104.0                3.0  3.0  3.5  8.00         1             0.72
3           4      322.0        110.0                3.0  3.5  2.5  8.67         1             0.80
4           5      314.0        103.0                2.0  2.0  3.0  8.21         0             0.65 """

data.describe(include='all')
""" 
       Serial No.   GRE Score  TOEFL Score  University Rating         SOP        LOR        CGPA    Research  Chance of Admit
count  500.000000  485.000000   490.000000         485.000000  500.000000  500.00000  500.000000  500.000000        500.00000
mean   250.500000  316.558763   107.187755           3.121649    3.374000    3.48400    8.576440    0.560000          0.72174
std    144.481833   11.274704     6.112899           1.146160    0.991004    0.92545    0.604813    0.496884          0.14114
min      1.000000  290.000000    92.000000           1.000000    1.000000    1.00000    6.800000    0.000000          0.34000
25%    125.750000  308.000000   103.000000           2.000000    2.500000    3.00000    8.127500    0.000000          0.63000
50%    250.500000  317.000000   107.000000           3.000000    3.500000    3.50000    8.560000    1.000000          0.72000
75%    375.250000  325.000000   112.000000           4.000000    4.000000    4.00000    9.040000    1.000000          0.82000
max    500.000000  340.000000   120.000000           5.000000    5.000000    5.00000    9.920000    1.000000          0.97000 """

# check for missing values in the data
data.isnull().sum()
""" 
Serial No.            0
GRE Score            15
TOEFL Score          10
University Rating    15
SOP                   0
LOR                   0
CGPA                  0
Research              0
Chance of Admit       0
dtype: int64 """

# fill missing values with mode and mean
data['University Rating'] = data['University Rating'].fillna(data['University Rating'].mode()[0])
data['TOEFL Score'] = data['TOEFL Score'].fillna(data['TOEFL Score'].mean())
data['GRE Score']  = data['GRE Score'].fillna(data['GRE Score'].mean())

# check for missing values in the data after missing values are addressed
data.isnull().sum()
""" 
Serial No.           0
GRE Score            0
TOEFL Score          0
University Rating    0
SOP                  0
LOR                  0
CGPA                 0
Research             0
Chance of Admit      0
dtype: int64 """

data.describe()
""" 
       Serial No.   GRE Score  TOEFL Score  University Rating         SOP        LOR        CGPA    Research  Chance of Admit
count  500.000000  500.000000   500.000000         500.000000  500.000000  500.00000  500.000000  500.000000        500.00000
mean   250.500000  316.558763   107.187755           3.118000    3.374000    3.48400    8.576440    0.560000          0.72174
std    144.481833   11.103952     6.051338           1.128993    0.991004    0.92545    0.604813    0.496884          0.14114
min      1.000000  290.000000    92.000000           1.000000    1.000000    1.00000    6.800000    0.000000          0.34000
25%    125.750000  309.000000   103.000000           2.000000    2.500000    3.00000    8.127500    0.000000          0.63000
50%    250.500000  316.558763   107.000000           3.000000    3.500000    3.50000    8.560000    1.000000          0.72000
75%    375.250000  324.000000   112.000000           4.000000    4.000000    4.00000    9.040000    1.000000          0.82000
max    500.000000  340.000000   120.000000           5.000000    5.000000    5.00000    9.920000    1.000000          0.97000 """

""" 
- data looks good and there are no missing values. 
- Also, the first cloumn is just serial numbers, so we don' need that column. 

Let's drop it from data and make it more clean. """

data= data.drop(columns = ['Serial No.'])
data.head()
""" 
    GRE Score  TOEFL Score  University Rating  SOP  LOR  CGPA  Research  Chance of Admit
0  337.000000        118.0                4.0  4.5  4.5  9.65         1             0.92
1  324.000000        107.0                4.0  4.0  4.5  8.87         1             0.76
2  316.558763        104.0                3.0  3.0  3.5  8.00         1             0.72
3  322.000000        110.0                3.0  3.5  2.5  8.67         1             0.80
4  314.000000        103.0                2.0  2.0  3.0  8.21         0             0.65 """

# Let's visualize the data and analyze the relationship between independent and dependent variables:

# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber<=16 :
        ax = plt.subplot(4,4,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.tight_layout()
plt.show()

""" 
- The data distribution looks decent enough and there doesn't seem to be any skewness. Great let's go ahead!
- Let's observe the relationship between independent variables and dependent variable.
 """

y = data['Chance of Admit']
X =data.drop(columns = ['Chance of Admit'])
plt.figure(figsize=(20,30), facecolor='white')
plotnumber = 1

for column in X:
    if plotnumber<=15 :
        ax = plt.subplot(5,3,plotnumber)
        plt.scatter(X[column],y)
        plt.xlabel(column,fontsize=20)
        plt.ylabel('Chance of Admit',fontsize=20)
    plotnumber+=1
plt.tight_layout()
plt.show()

""" 
- Great, the relationship between the dependent and independent variables look fairly linear. 
- Thus, our linearity assumption is satisfied.

- Let's move ahead and check for multicollinearity.
 """
scaler =StandardScaler()
X_scaled = scaler.fit_transform(X)

from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = X_scaled

variables

""" 
array([[ 1.84274116e+00,  1.78854223e+00,  7.82009548e-01, ...,
         1.09894429e+00,  1.77680627e+00,  8.86405260e-01],
       [ 6.70814288e-01, -3.10581135e-02,  7.82009548e-01, ...,
         1.09894429e+00,  4.85859428e-01,  8.86405260e-01],
       [ 5.12433309e-15, -5.27312752e-01, -1.04622593e-01, ...,
         1.73062093e-02, -9.54042814e-01,  8.86405260e-01],
       ...,
       [ 1.21170361e+00,  2.11937866e+00,  1.66864169e+00, ...,
         1.63976333e+00,  1.62785086e+00,  8.86405260e-01],
       [-4.10964364e-01, -6.92730965e-01,  7.82009548e-01, ...,
         1.63976333e+00, -2.42366993e-01, -1.12815215e+00],
       [ 9.41258951e-01,  9.61451165e-01,  7.82009548e-01, ...,
         1.09894429e+00,  7.67219636e-01, -1.12815215e+00]]) """

# we create a new data frame which will include all the VIFs
# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
# we do not include categorical values for mulitcollinearity as they do not provide much information as numerical ones do
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]

# Finally, I like to include names so it is easier to explore the result
vif["Features"] = X.columns

vif
""" 
        VIF           Features
0  4.152735          GRE Score
1  3.793345        TOEFL Score
2  2.517272  University Rating
3  2.776393                SOP
4  2.037449                LOR
5  4.654369               CGPA
6  1.459411           Research """

""" 
- Here, we have the correlation values for all the features. 
- As a thumb rule, a VIF value greater than 5 means a very severe multicollinearity. 
- We don't any VIF greater than 5 , so we are good to go.

- Great. Let's go ahead and use linear regression and see how good it fits our data. 
- But first. let's split our data in train and test.
 """
x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.25,random_state=355)

y_train
""" 
378    0.56
23     0.95
122    0.57
344    0.47
246    0.72
       ...
51     0.56
291    0.56
346    0.47
130    0.96
254    0.85
Name: Chance of Admit, Length: 375, dtype: float64 """

regression = LinearRegression()

regression.fit(x_train,y_train)
""" 
LinearRegression() """

# saving the model to the local file system
import pickle

filename = 'final_model.pickle'

pickle.dump(regression, open(filename, 'wb'))

# prediction using the saved model
loaded_model = pickle.load(open(filename, 'rb'))

a=loaded_model.predict(scaler.transform([[300,110,5,5,5,10,1]]))

a
# array([0.92190162])

regression.score(x_train,y_train)
# 0.8415250484247909

# Let's create a function to create adjusted R-Squared
def adj_r2(x,y):
    r2 = regression.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2

adj_r2(x_train,y_train)
# 0.8385023654247188

""" 
- Our r2 score is 84.15% and adj r2 is 83.85% for our training , so looks like we are not being penalized by use 
        of any feature. 
        
- Let's check how well model fits the test data.

- Now let's check if our model is overfitting our data using regularization."""

regression.score(x_test,y_test)
# 0.7534898831471066

adj_r2(x_test,y_test)
# 0.7387414146174464

""" 
- So it looks like our model r2 score is less on the test data.

Let's see if our model is overfitting our training data. """

# Lasso Regularization
# LassoCV will return best alpha and coefficients after performing 10 cross validations
lasscv = LassoCV(alphas = None,cv =10, max_iter = 100000, normalize = True)
lasscv.fit(x_train, y_train)
""" 
LassoCV(cv=10, max_iter=100000, normalize=True) """

# best alpha parameter
alpha = lasscv.alpha_

alpha
# 3.0341655445178153e-05
# now that we have best parameter, let's use Lasso regression and see how well our data has fitted before

lasso_reg = Lasso(alpha)
lasso_reg.fit(x_train, y_train)
# Lasso(alpha=3.0341655445178153e-05)

lasso_reg.score(x_test, y_test)
# 0.7534654960492284

""" 
- our r2_score for test data (75.34%) comes same as before using regularization. 
- So, it is fair to say our OLS model did not overfit the data. """

# Using Ridge regression model
# RidgeCV will return best alpha and coefficients after performing 10 cross validations. 
# We will pass an array of random numbers for ridgeCV to select best alpha from them

alphas = np.random.uniform(low=0, high=10, size=(50,))
ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)
ridgecv.fit(x_train, y_train)
""" 
RidgeCV(alphas=array([7.71160025, 4.89704475, 2.26073619, 4.72218538, 3.29671112,
       6.77254816, 1.30378135, 8.87189262, 8.59222773, 1.65793652,
       0.47857702, 1.81721786, 9.71075685, 2.98434191, 6.26482878,
       4.20661968, 4.06495692, 5.53712557, 0.81006518, 2.76571898,
       1.17702301, 6.35949763, 7.56435146, 8.0937448 , 2.8872764 ,
       8.39004522, 3.55549035, 5.76323149, 8.72174929, 3.06687108,
       1.4551871 , 9.77827694, 2.42087102, 7.8311565 , 9.94779198,
       9.03373019, 5.57054305, 7.99294792, 7.68984049, 1.61585639,
       4.60946789, 5.75690071, 0.65660216, 7.97103371, 6.01714369,
       5.67094673, 6.37615597, 4.11351162, 7.98963922, 7.16156557]),
        cv=10, normalize=True) """

ridgecv.alpha_
# 0.8432446610176114

ridge_model = Ridge(alpha=ridgecv.alpha_)
ridge_model.fit(x_train, y_train)
# Ridge(alpha=0.47857702367108623)

ridge_model.score(x_test, y_test)
# 0.7538937537809315

# we got the same r2 square using Ridge regression as well. So, it's safe to say there is no overfitting.

# Elastic net
elasticCV = ElasticNetCV(alphas = None, cv =10)
elasticCV.fit(x_train, y_train)
# ElasticNetCV(cv=10)

elasticCV.alpha_
# 0.0011069728449315508
# l1_ration gives how close the model is to L1 regularization, below value indicates we are giving equal
# preference to L1 and L2

elasticCV.l1_ratio
# 0.5

elasticnet_reg = ElasticNet(alpha = elasticCV.alpha_,l1_ratio=0.5)
elasticnet_reg.fit(x_train, y_train)
""" 
ElasticNet(alpha=0.0011069728449315508) """

elasticnet_reg.score(x_test, y_test)
# 0.7531695370639867

""" 
- So, we can see by using different type of regularization, we still are getting the same r2 score. 
That means our OLS model has been well trained over the training data and there is no overfitting. """