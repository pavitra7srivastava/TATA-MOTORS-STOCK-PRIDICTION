import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC 
from xgboost import XGBClassifier 
from sklearn import metrics















# we are taking tata mortors 5 yeaars stocks from 17th july 2019 to 17th july 2024
df = pd.read_csv(r'C:\Users\Pavitra\Downloads\TATAMOTORS.NS.csv')
df.head()
print(df.head())

#we will get to know how many  rows and columns are there
df.shape
print(df.shape)

df.describe()
print(df.describe())

df.info()
print(df.info())


#graph visualization of tata motors close price over time
plt.figure(figsize=(20,10)) 
plt.plot(df['Close']) 
plt.title('TataMotors Close price.', fontsize=10) 
plt.ylabel('dollars') 
plt.show()

#checking if there is any null value or not
df.isnull().sum()
print(df.isnull().sum())



#crrating subplots
features = ['Open', 'High', 'Low', 'Close', 'Volume'] 
plt.subplots(figsize=(20,10)) 
for i, col in enumerate(features): 
    plt.subplot(2,3,i+1) 
    sb.distplot(df[col])
plt.show()

#creating boxplots
plt.subplots(figsize=(20,10)) 
for i, col in enumerate(features): 
    plt.subplot(2,3,i+1) 
    sb.boxplot(df[col]) 
plt.show()


#ading some more col to the charts to get some more info

splitted = df['Date'].str.split('-', expand=True)
print(splitted)

df['year'] = splitted[0].astype('int') 
df['month'] = splitted[1].astype('int') 
df['day'] = splitted[2].astype('int')

df['is_quarter_end'] = np.where(df['month']%3==0,1,0) 

df.head()
print(df.head())
df.info()
print(df.info())



#convert date into datetime formate
df['Date'] = pd.to_datetime(df['Date'])

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day




#graph representation by grouping mean of years
data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20,10)) 

for i, col in enumerate(['Open', 'High', 'Low', 'Close']): 
    plt.subplot(2,2,i+1) 
    data_grouped[col].plot.bar() 
plt.show()

df.groupby('is_quarter_end').mean()



#creating new columns and storing new values
df['open-close'] = df['Open'] - df['Close'] 
df['low-high'] = df['Low'] - df['High'] 
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# creating pie chat
plt.pie(df['target'].value_counts().values, 
		labels=[0, 1], autopct='%1.1f%%') 
plt.show()



#creating heatmap
plt.figure(figsize=(10, 10)) 
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False) 
plt.show()


#data training and validition and normalization
features = df[['open-close', 'low-high', 'is_quarter_end']] 
target = df['target'] 

scaler = StandardScaler() 
features = scaler.fit_transform(features) 

X_train, X_valid, Y_train, Y_valid = train_test_split( 
	features, target, test_size=0.1, random_state=2022) 
print(X_train.shape, X_valid.shape) 



#model evolution
models = [LogisticRegression(), SVC( 
kernel='poly', probability=True), XGBClassifier()] 

for i in range(3): 
    models[i].fit(X_train, Y_train) 

print(f'{models[i]} : ') 
print('Training Accuracy : ', metrics.roc_auc_score( 
	Y_train, models[i].predict_proba(X_train)[:,1])) 
print('Validation Accuracy : ', metrics.roc_auc_score( 
	Y_valid, models[i].predict_proba(X_valid)[:,1])) 
print()
for model in models:
    

    # Compute confusion matrix
    cm = confusion_matrix(Y_valid, model.predict(X_valid))
    print(f'Confusion Matrix for {type(model).__name__}:')
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sb.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {type(model).__name__}')
    plt.show()











