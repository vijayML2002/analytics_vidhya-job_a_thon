import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import tensorflow as tf

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

rem_list = ['ID','Region_Code']

def preprocessing_data(data,removal_list):
  column = data.columns.to_list()
  for rem in removal_list:
    column.remove(rem)
  data = pd.DataFrame(data,columns=column)
  return data

def split_feature_target(data,target):
    column = data.columns.to_list()
    try:
        column.remove(target)
        df_features = pd.DataFrame(data, columns = column)
        df_target = pd.DataFrame(data, columns = [target])
        return df_features,df_target
    except:
        print('The given target is not found in the dataframe')
        return None,None

def define_model(n_input):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.InputLayer(input_shape=n_input))
	model.add(tf.keras.layers.Dense(10, activation='relu'))
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='sgd')
	return model
      
def create_model(train):
    data = preprocessing_data(train,rem_list)
    y = data["Response"]
    x = data.drop("Response",axis=1)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    model = define_model((11,))
    model.fit(train_x,train_y,epochs=100) 
    pred = model.predict(test_x)
    print( accuracy_score(test_y, pred) )
    print(np.unique(pred))
    print(classification_report(test_y, pred))
    print(confusion_matrix(test_y,pred))
    return model,pred

def file_create(train,test):
  model,pred = create_model(train)
  test_column = test.columns.to_list()
  test_column.remove("ID")
  df_ID = pd.DataFrame(test,columns=["ID"])
  data = pd.DataFrame(test,columns=test_column)
  prediction = model.predict(data)
  data_ID = df_ID.to_numpy()
  data_id = np.reshape(data_ID,(21805,))
  df = pd.DataFrame({'ID':data_id, 'Response':prediction})
  df.to_csv('prediction.csv',index=False)


def up_sample(df):
  df_majority = df[df.Response==0]
  df_minority = df[df.Response==1]
   
  df_minority_upsampled = resample(df_minority, 
                                   replace=True,     
                                   n_samples=len(df_majority),    
                                   random_state=123) 
   
  df_upsampled = pd.concat([df_majority, df_minority_upsampled])
  df_upsampled.Response.value_counts()
  return df_upsampled

def down_sample(df):
  df_majority = df[df.Response==0]
  df_minority = df[df.Response==1]
   
  df_majority_downsampled = resample(df_majority, 
                                   replace=False,    
                                   n_samples=len(df_minority),     
                                   random_state=123) 
   
  df_downsampled = pd.concat([df_majority_downsampled, df_minority])
  df_downsampled.Response.value_counts()
  return df_downsampled

print('UNBALANCED DATA') 
model,pred = create_model(train_data)

print('UP SAMPLED DATA')
model,pred = create_model(up_sample(train_data))

print('DOWN SAMPLED DATA')
model,pred = create_model(down_sample(train_data))
