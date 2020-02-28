#!/usr/bin/env python
# coding: utf-8

# In[395]:


#LIBRARIES
import numpy as np
import pandas as pd


# In[396]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[397]:


#DATASSET
df = pd.read_csv('cancer.csv')


# In[398]:


df.head()


# In[399]:


from sklearn.model_selection import train_test_split


# In[400]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[401]:


df.shape


# In[402]:


df.info()


# In[403]:


#TO CHECK NULL VALUES, IF ANY 
df.isnull().any()


# In[404]:


df.diagnosis.value_counts()


# In[405]:


df.diagnosis.value_counts().plot(kind='bar')


# In[406]:


#df.columns


# In[407]:


#VISUALIZATION 
df.hist(bins=10,figsize=(20,20),grid=False)


# In[408]:


import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[409]:


init_notebook_mode(connected=True)


# In[410]:


cf.go_offline()


# In[411]:


#df=pd.DataFrame(np.random.randn(100,11),columns='radius_worst texture_worst perimeter_worst area_worst smoothness_worst compactness_worst concavity_worst concave points_worst symmetry_worst fractal_dimension_worst'.split()) 


# In[412]:


#df.head()


# In[413]:


#df.iplot(kind='box')


# In[414]:


cancer_mapping = {'B':0, 'M':1}
df.diagnosis=df.diagnosis.map(cancer_mapping)


# In[415]:


df.isnull().any()


# In[416]:


#to check duplicacy
df.duplicated()


# In[417]:


#unique identity=id
df.id.value_counts().unique()


# In[418]:


df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)


# In[419]:


df.columns


# In[420]:


#Dimensional reduction
y=df['diagnosis']
X=df[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]


# In[421]:


#correlation map
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(X.corr(),annot=True,linewidth=.5)


# In[422]:


#high correlation,threshold=90% , 0.90
high_corr_pts=X[['radius_mean','perimeter_mean','area_mean','radius_worst','perimeter_worst']]


# In[423]:


#for transparent plot,we need low alpha value
plt.scatter(df.diagnosis,df.radius_mean,alpha=0.1)
plt.title('Radius mean w.r.t Diagnosis')


# In[424]:


plt.scatter(df.diagnosis,df.perimeter_mean,alpha=0.1)
plt.title('Perimeter mean w.r.t Diagnosis')


# In[425]:


plt.scatter(df.diagnosis,df.area_mean,alpha=0.1)
plt.title('Area mean w.r.t Diagnosis')


# In[426]:


plt.scatter(df.diagnosis,df.perimeter_worst,alpha=0.1)
plt.title('Perimeter worst w.r.t Diagnosis')


# In[427]:


#Splitting the dataset
#40% test and 60% train
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4)


# In[428]:


sns.pairplot(high_corr_pts)


# In[429]:


import xgboost as xgb


# In[430]:


model_all = xgb.XGBClassifier()


# In[431]:


model_all.fit(X_train,y_train)


# In[432]:


pred=model_all.predict(X_test)


# In[435]:


accuracy_score(y_test,pred)


# In[436]:


print(classification_report(y_test,pred))


# In[437]:


confusion_matrix(y_test,pred)


# In[438]:


#accuracy=95.61%


# In[439]:


#Optimal features
from sklearn.feature_selection import RFECV


# In[440]:


#Accuracy scoring is proportional to number of correct classifciations
clf=xgb.XGBClassifier()
rfecv=RFECV(estimator=clf,step=1,cv=5,scoring='accuracy') #5-fold cross validation
rfecv=rfecv.fit(X_train,y_train)

print('Optimal number of features:' , rfecv.n_features_)
print('Best features:',X_train.columns[rfecv.support_])


# In[441]:


#VS. cross validation scores
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel('Number of features selected')
plt.ylabel('Cross validation score of number of selected features')
plt.plot(range(1,len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show


# In[442]:


#ranking features according to their importance
importances=(model_all.feature_importances_)
indices=np.argsort(importances)[::-1]
print('Feature ranking:')
for f in range(X_train.shape[1]):
    print('%d. feature %d (%f)' % (f + 1,indices[f], importances[indices[f]])) 


# In[443]:


#plotting important features
plt.title('Feature importances')
plt.bar(range(X_train.shape[1]),importances[indices], color='g',align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()
plt.show()


# In[444]:


#train and analyse model with optimal features
X_train_optimal=X_train[['texture_mean', 'area_mean', 'concavity_mean', 'radius_se',
       'perimeter_se', 'area_se', 'smoothness_se', 'concavity_se',
       'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
       'smoothness_worst', 'compactness_worst', 'concavity_worst',
       'concave points_worst', 'symmetry_worst']]
X_test_optimal=X_test[['texture_mean', 'area_mean', 'concavity_mean', 'radius_se',
       'perimeter_se', 'area_se', 'smoothness_se', 'concavity_se',
       'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
       'smoothness_worst', 'compactness_worst', 'concavity_worst',
       'concave points_worst', 'symmetry_worst']]
model_optimal=xgb.XGBClassifier()
model_optimal.fit(X_train_optimal,y_train)
pred = model_optimal.predict(X_test_optimal)
accuracy_score(y_test,pred)


# In[445]:


#KNN
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot

cf.go_offline()
df = pd.read_csv('cancer.csv')
df.replace('?', 99, inplace=True)
df.drop(['id'], 1, inplace=True)
df.head()
df.iplot(kind='box')

y=df['diagnosis']
X=df[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)


# In[461]:



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting the Logistic Regression Algorithm to the Training Set
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X_train, y_train)
#95.8 Acuracy

#Fitting SVM
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 0)
#classifier.fit(X_train, y_train) 
#accuracy=classifier.score(X_test, y_test)
#print(accuracy)
#97.2 Acuracy

#Fitting K-SVM
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(X_train, y_train)
#96.5 Acuracy

#Fitting Naive_Bayes
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train, y_train)
#91.6 Acuracy

#Fitting Decision Tree Algorithm
#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#classifier.fit(X_train, y_train)
#95.8 Acuracy

#Fitting Random Forest Classification Algorithm
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
#classifier.fit(X_train, y_train)
#98.6 Acuracy

#Fitting K-NN Algorithm
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#classifier.fit(X_train, y_train)
#95.1 Acuracy


#predicting the Test set results
#y_pred = classifier.predict(X_test)

#Creating the confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#c = print(cm[0, 0] + cm[1, 1])


# In[473]:


#Fitting SVM
import numpy as np
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
y=df['diagnosis']
X=df[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]
cf.go_offline()
df = pd.read_csv('cancer.csv')
df.replace('?', 99, inplace=True)
df.drop(['id'], 1, inplace=True)
df.head()
df.iplot(kind='box')
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train) 
accuracy=classifier.score(X_test, y_test)
print(accuracy)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[470]:


#Fitting K-NN Algorithm
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
y=df['diagnosis']
X=df[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]
cf.go_offline()
df = pd.read_csv('cancer.csv')
df.replace('?', 99, inplace=True)
df.drop(['id'], 1, inplace=True)
df.head()
df.iplot(kind='box')
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
accuracy=classifier.score(X_test,y_test)
print(accuracy)
#95.1 Acuracy


# In[471]:


#Fitting Random Forest Classification Algorithm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
y=df['diagnosis']
X=df[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]
cf.go_offline()
df = pd.read_csv('cancer.csv')
df.replace('?', 99, inplace=True)
df.drop(['id'], 1, inplace=True)
df.head()
df.iplot(kind='box')
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
accuracy=classifier.score(X_test,y_test)
print(accuracy)
#98.6 Acuracy


# In[472]:


#Fitting K-SVM
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
y=df['diagnosis']
X=df[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]
cf.go_offline()
df = pd.read_csv('cancer.csv')
df.replace('?', 99, inplace=True)
df.drop(['id'], 1, inplace=True)
df.head()
df.iplot(kind='box')
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
accuracy=classifier.score(X_test,y_test)
print(accuracy)
#96.5 Acuracy


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




