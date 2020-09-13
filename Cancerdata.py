# CancerData
#Using Data Visualization and ML models on Cancer Data


# here we will import the libraries used for machine learning
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph. I like it most for plot
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.model_selection import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV # for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model
# Any results you write to the current directory are saved as output.
# dont worry about the error if its not working then instead use of model_selection we can use cross_validation

data = pd.read_csv(r"C:\\Users\\Siddharth\\Downloads\\CancerData.csv")
print(data.head(3))

# now lets look at the type of data we have. We can use
#print(data.info())

# now we can drop this column Unnamed: 32
data.drop("Unnamed: 32",axis=1,inplace=True) # in this process this will change in our data itself
# if you want to save your old data then you can use below code
# data1=data.drop("Unnamed:32",axis=1)
# here axis 1 means we are droping the column.

# here you can check the column has been droped
#print(data.columns) # this gives the column name which are persent in our data no Unnamed: 32 is not now there

#print(len(data.columns))

# like this we also don't want the Id column for our analysis
data.drop("id",axis=1,inplace=True)

#print(data.columns)
#print(len(data.columns))

# As said above the data can be divided into three parts.lets divied the features according to their category
features_mean= list(data.columns[1:11])
features_se= list(data.columns[11:20])
features_worst=list(data.columns[21:31])
#print(features_mean)
#print("-----------------------------------")
#print(features_se)
#print("------------------------------------")
#print(features_worst)

# lets now start with features_mean
# now as ou know our diagnosis column is a object type so we can map it to integer value
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

#print(data.describe()) # this will describe the all statistical function of our data

# lets get the frequency of cancer stages
sns.countplot(data['diagnosis'],label="Count")
#plt.show()  # from this graph we can see that there is a more number of bengin stage of cancer which can be cure

# now lets draw a correlation graph so that we can remove multi colinearity it means the columns are
# dependenig on each other so we should avoid it because what is the use of using same column twice
# lets check the correlation between features
# now we will do this analysis only for features_mean then we will do for others and will see who is doing best
corr = data[features_mean].corr() # .corr is used for find corelation
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= features_mean, yticklabels= features_mean,
           cmap= 'coolwarm')
#plt.show()

# observation
#
# the radius, parameter and area are highly correlated as expected from their relation so from these we will use anyone of them
# compactness_mean, concavity_mean and concavepoint_mean are highly correlated so we will use compactness_mean from here
# so selected Parameter for use is perimeter_mean, texture_mean, compactness_mean, symmetry_mean*

prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
# now these are the variables which will use for prediction

#now split our data into train and test
train, test = train_test_split(data, test_size = 0.3)# in this our main data is splitted into train and test
# we can check their dimension
#print(train.shape)
#print(test.shape)

train_X = train[prediction_var]# taking the training data input
train_y=train.diagnosis# This is output of our training data
# same we have to do for test
test_X= test[prediction_var] # taking test data inputs
test_y =test.diagnosis   #output value of test data

model=RandomForestClassifier(n_estimators=100)# a simple random forest model

model.fit(train_X,train_y)# now fit our model for traiing data

prediction=model.predict(test_X)# predict for the test data
print(prediction)
print("\n")
# prediction will contain the predicted value by our model predicted values of dignosis column for test inputs

accurate_score = metrics.accuracy_score(prediction,test_y) # to check the accuracy
# here we will use accuracy measurement between our predicted value and our test output values

print(len(prediction))
print("\n")

print(accurate_score*100) # Here the Accuracy for our model is 91 % which seems good*
print("\n")

########################### lets now try with SVM Model #########################################################

model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
svm_accurate_score = metrics.accuracy_score(prediction,test_y)

print(prediction)
print("\n")

print(svm_accurate_score)
print("\n")

# SVM is giving only around 0.85 which we can improve by using different techniques i will improve it till
# then beginners can understand how to model a data and they can have a overview of ML
#
# *Now lets do this for all feature_mean so that from Random forest we can get the feature which are important**

prediction_var = features_mean # taking all features
train_X= train[prediction_var]
train_y= train.diagnosis

test_X = test[prediction_var]
test_y = test.diagnosis

model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)

print(prediction)
print("\n")

print(len(prediction))

print(metrics.accuracy_score(prediction,test_y))
print("\n")

# by taking all features accuracy increased but not so much so according to Razor's rule simpler method is better
# by the way now lets check the importan features in the prediction

##### VERY IMPORTANT THING TO TAKE A NOTE OF IT

featimp = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)
print(featimp)
# this is the property of Random Forest classifier that it provide us the importance
# of the features used

####### NOW USING ALL FEATURES WITH SVM MODEL AND SEE WHAT COMES OUT.

# first lets do with SVM also using all features

model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
svm_score_with_all_features = metrics.accuracy_score(prediction,test_y)

print(svm_score_with_all_features)

# # as you can see the accuracy of SVM decrease very much
# # now lets take only top 5 important features given by RandomForest classifier

prediction_var=['concave points_mean','perimeter_mean' , 'concavity_mean' , 'radius_mean','area_mean']


train_X= train[prediction_var]
train_y= train.diagnosis
test_X = test[prediction_var]
test_y = test.diagnosis

model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)

print(metrics.accuracy_score(prediction,test_y))

####
model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)

print(metrics.accuracy_score(prediction,test_y))

# looking at the accuracy scores now I think for simplicity the Randomforest will be better for prediction

# Now explore a little bit more
# now from features_mean i will try to find the variable which can be use for classify
# so lets plot a scatter plot to identify those variable who have a separable boundary between two class
#of cancer

# # Lets start with the data analysis for features_mean
# # Just try to understand which features can be used for prediction
# # I will plot scatter plot for the all features_mean for both of diagnosis Category
# # and from it we will find which are easily can used for differenciate between two category

from pandas.plotting import scatter_matrix
color_function = {0: "blue", 1: "red"} # Here Red color will be 1 which means M and blue foo 0 means B
colors = data["diagnosis"].map(lambda x: color_function.get(x))# mapping the color fuction with diagnosis column
pd.plotting.scatter_matrix(data[features_mean], c=colors, alpha = 0.5, figsize = (15, 15)); # plotting scatter plot matrix
plt.show()

# Observation
#
# ** 1. Radius, area and perimeter have a strong linear relationship as expected 2 As graph shows
# the features like as texture_mean, smoothness_mean, symmetry_mean and fractal_dimension_mean can
# not be used for classify two category because both category are mixed there is no separable plane
#
# So we can remove them from our prediction_var

# currently predicton features are :

print(features_mean)

# So new predicton features will be
predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']

# # Now with these variable we will try to explore a liitle bit we will move to how to use cross validiation
# # for a detail on cross validation use this link https://www.analyticsvidhya.com/blog/2015/11/improve-model-performance-cross-validation-in-python-r/

def model(model,data,prediction,outcome):
    # This function will be used for to check accuracy of different model
    # model is the m
    kf = KFold(data.shape[0], n_folds=10) # if you have refer the link then you must understand what is n_folds

prediction_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']

# so those features who are capable of classify classe will be more useful.



