import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
train=pd.read_csv('C:\\Users\\RAJIV MISHRA\\Desktop\\Arpita\\sampleproblemdataset\\trainingdata.csv')
test=pd.read_csv('C:\\Users\\RAJIV MISHRA\\Desktop\\Arpita\\sampleproblemdataset\\testdata.csv')
train['Type']='Train' #Create a flag for Train and Test Data set
test['Type']='Test'
fullData = pd.concat([train,test],axis=0) #Combined both Train and Test Data set
print(fullData.columns)# This will show all the column names
print(fullData.head(10)) # Show first 10 records of dataframe
print(fullData.describe()) #You can look at summary of numerical fields by using describe() function
ID_col = ['Loan_ID']
target_col = ["Loan_Status"]
cat_cols = ['Education','Gender','Loan_Status','Married','Property_Area','Self_Employed','Dependents']
num_cols= list(set(list(fullData.columns))-set(cat_cols)-set(ID_col)-set(target_col))
other_col=['Type'] #Test and Train Data set identifier
print(fullData.isnull().any())
num_cat_cols = num_cols+cat_cols
for var in num_cat_cols:
    if fullData[var].isnull().any()==True:
        fullData[var+'_NA']=fullData[var].isnull()*1
fullData[num_cols] = fullData[num_cols].fillna(fullData[num_cols].mean(),inplace=True)
print(fullData[num_cols])
#Impute categorical missing values with -9999
fullData[cat_cols] = fullData[cat_cols].fillna(value = -9999)
print(fullData[cat_cols])
for var in cat_cols:
 number = LabelEncoder()
 fullData[var] = number.fit_transform(fullData[var].astype('str'))
#Target variable is also a categorical so convert it
fullData["Loan_Status"] = number.fit_transform(fullData["Loan_Status"].astype('str'))
train=fullData[fullData['Type']=='Train']
test=fullData[fullData['Type']=='Test']
train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
Train, Validate = train[train['is_train']==True], train[train['is_train']==False]
features=list(set(list(fullData.columns))-set(ID_col)-set(target_col)-set(other_col))
x_train = Train[list(features)].values
y_train = Train["Loan_Status"].values
x_validate = Validate[list(features)].values
y_validate = Validate["Loan_Status"].values
x_test=test[list(features)].values
random.seed(100)
#lg = LogisticRegression()
#lg.fit(x_train, y_train)
#status = lg.predict_proba(x_validate)
#fpr, tpr, _ = roc_curve(y_validate, status[:,1])
#roc_auc = auc(fpr, tpr)
#print(roc_auc)
#final_status = lg.predict_proba(x_test)
dtree = DecisionTreeClassifier()
dtree.fit(x_train,y_train)
#status = dtree.predict_proba(x_validate)
final_status = dtree.predict_proba(x_test)
test["Loan_Status"]=final_status[:,1]
test.to_csv('C:\\Users\\RAJIV MISHRA\\Desktop\\Arpita\\sampleproblemdataset\\model_output3.csv',columns=['Loan_ID','Loan_Status'])