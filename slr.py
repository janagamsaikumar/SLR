# import the lib first 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
# read the dataset and store into the object or varibale 
dataset=pd.read_csv(r'C:\Users\saikumar\Desktop\AMXWAM data science\class 16 & 17_oct 17, 2020\1.SIMPLE LINEAR REGRESSION\salary_data.csv')
# this dataset contains experience and salary 
# used case was predict the salary based on the experience when a new employee join
#----------------------
# now seperate the independent and dependent variables 
X=dataset.iloc[:,:-1] # iloc index location of (r,c) -1 will eliminates the last row of the dataset and make it split into dependent(y) and independent varibales (X)
y=dataset.iloc[:,-1]
 # as we see thre are no missing values and outliers are inthe data frame so we directly train and test the dataset
 # and also no catergorical data in the dataset
from sklearn.model_selection import train_test_split
 # this fuction will split the data frame into train and test by the amount of percentage we mentioned generally we take 80% to train the model and 20% for testing 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0) # randon state should be 0 always by default when testing the model it predicts the different accuracy for each attempt
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train) # fitting our training to set to linearreg model
y_pred=reg.predict(X_test)# now its time to test my dataset with y test and observe the numbers what my model has predicted.
plt.title('exp vs salary')
plt.xlabel('experience')
plt.ylabel('salary')
plt.scatter(X_train,y_train,color='red') # graphical representation would be more better to understand the slr model
plt.plot(X_train,reg.predict(X_train),color='blue') # .predict will give the predicted line by the model observe the graph and take it into  consideration
plt.show()
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='blue')
plt.title('exp vs salary')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()
 # almost all the data points of my trained and test  dataset touches the predicted line and much closer to the line means very low error rate 