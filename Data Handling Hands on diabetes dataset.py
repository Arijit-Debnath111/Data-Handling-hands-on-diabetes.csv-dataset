#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


data=pd.read_csv('diabetes.csv')


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# In[7]:


data.describe()


# In[8]:


data.columns


# In[9]:


for x in data.columns:
    count=(data[x]==0).sum()
    print('Count of zeros in',x,'is',count)


# In[10]:


import numpy as np


# In[11]:


data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)


# In[12]:


data.isnull().sum()


# In[13]:


for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
    data[col]=data[col].fillna(data[col].median())


# In[14]:


data.isnull().sum()


# In[15]:


for x in data.columns:
    count=(data[x]==0).sum()
    print('Count of zeros in',x,'is',count)


# In[16]:


data.head()


# In[17]:


data['Outcome'].value_counts()


# In[18]:


x=data.iloc[:,:-1]


# In[19]:


x.head()


# In[20]:


y=data.iloc[:,-1]


# In[21]:


y.head()


# In[22]:


y.head()
from sklearn.model_selection import train_test_split


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


# In[28]:


x_train.head()


# In[29]:


x_train.shape


# In[30]:


x_test.shape


# In[31]:


from sklearn.ensemble import RandomForestClassifier


# In[32]:


r_f=RandomForestClassifier(n_estimators=10)


# In[33]:


r_f.fit(x_train,y_train)  # trained model


# In[34]:


pred=r_f.predict(x_test)


# In[35]:


x_test.head()


# In[37]:


pred


# In[38]:


y_test


# In[39]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[40]:


print(accuracy_score(y_test,pred))


# In[41]:


print(confusion_matrix(y_test,pred))


# In[42]:


r_f_1=RandomForestClassifier(n_estimators=350,criterion='entropy',max_features='sqrt',min_samples_leaf=11,random_state=2)


# In[43]:


r_f_1.fit(x_train,y_train)


# In[44]:


pred_1=r_f_1.predict(x_test)


# In[45]:


print(accuracy_score(y_test,pred_1))


# In[46]:


from sklearn.model_selection import RandomizedSearchCV


# In[47]:


n_estimators=[int(a) for a in np.linspace(200,2000,10) ]


# In[48]:


n_estimators


# In[53]:


max_features=['auto','sqrt','log2']
max_depth=[int(b) for b in np.linspace(10,1000,10)]


# In[54]:


d=['entropy','gini']


# In[55]:


RandomForestClassifier() 


# In[56]:


min_samples_split=[2,5,10,14]
min_samples_leaf=[1,2,4,6,8]

thunder={'n_estimators':n_estimators,'max_features':max_features,'min_samples_split':min_samples_split,'max_depth':max_depth,'min_samples_leaf':min_samples_leaf,'criterion':d}


# In[57]:


thunder


# In[58]:


t_h=RandomForestClassifier()


# In[59]:


t_h_random=RandomizedSearchCV(estimator=t_h,param_distributions=thunder,n_iter=100,cv=3,verbose=2,random_state=100,n_jobs=-1)


# In[62]:


t_h_random.fit(x_train,y_train)


# In[63]:


t_h_random.best_params_


# In[67]:


n_estimarors:400,max_features:'sqrt',min_samples_split:5,'max_depth':120,min_samples_leaf:6,criterion: 'entropy'


# In[68]:


# Grid search .

from sklearn.model_selection import GridSearchCV

sky={'n_estimators':[550,600,650],'Criterion':['entropy'],'max_depth':[120,125,110],'min_samples_leaf':[1,2],'max_features': ['sqrt'],'min_samples_split':[2,3,4]}


# In[69]:


sky


# In[70]:


best_rf=RandomForestClassifier()


# In[71]:


grid_search=GridSearchCV(estimator=best_rf,param_grid=sky,cv=10,n_jobs=-1,verbose=2)


# In[ ]:




