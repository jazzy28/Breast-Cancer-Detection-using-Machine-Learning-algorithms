#!/usr/bin/env python
# coding: utf-8

# In[1]:


#LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#DATASSET
df = pd.read_csv('cancer.csv')


# In[3]:


df.head()


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[5]:


df.info()


# In[6]:


df.isnull().any()


# In[7]:


df.diagnosis.value_counts()


# In[8]:


df.diagnosis.value_counts().plot(kind='bar')


# In[9]:


df.hist(bins=10,figsize=(20,20),grid=False)


# In[10]:


import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()


# In[11]:


cancer_mapping = {'B':0, 'M':1}
df.diagnosis=df.diagnosis.map(cancer_mapping)


# In[12]:


df.isnull().any()


# In[13]:


df.duplicated()


# In[14]:


df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)


# In[15]:


df.columns


# In[16]:


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


# In[17]:


#correlation map
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(X.corr(),annot=True,linewidth=.5)


# In[18]:


#high correlation,threshold=90% , 0.90
high_corr_pts=X[['radius_mean','perimeter_mean','area_mean','radius_worst','perimeter_worst']]


# In[19]:


#for transparent plot,we need low alpha value
plt.scatter(df.diagnosis,df.radius_mean,alpha=0.1)
plt.title('Radius mean w.r.t Diagnosis')


# In[20]:


plt.scatter(df.diagnosis,df.perimeter_mean,alpha=0.1)
plt.title('Perimeter mean w.r.t Diagnosis')


# In[21]:


plt.scatter(df.diagnosis,df.area_mean,alpha=0.1)
plt.title('Area mean w.r.t Diagnosis')


# In[22]:


plt.scatter(df.diagnosis,df.perimeter_worst,alpha=0.1)
plt.title('Perimeter worst w.r.t Diagnosis')


# In[23]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)


# In[24]:


sns.pairplot(high_corr_pts)


# In[25]:


import xgboost as xgb


# In[26]:


model_all = xgb.XGBClassifier()


# In[27]:


model_all.fit(X_train,y_train)


# In[28]:


pred=model_all.predict(X_test)


# In[29]:


accuracy_score(y_test,pred)


# In[30]:


confusion_matrix(y_test,pred)


# In[31]:


from sklearn.feature_selection import RFECV


# In[32]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[33]:


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
#0.972 Acuracy


# In[34]:


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


# In[35]:


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


# In[36]:


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


# In[37]:


#Fitting Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
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
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
accuracy=classifier.score(X_test,y_test)
print(accuracy)
#95.8 Acuracy


# In[110]:


#from tkinter import *
#from PIL import Image,ImageTk
#import cv2

#root=Tk()


#root.title("Breast Cancer Detection")
#root.geometry("800x800") 

#Label1
#label1=Label(root,text="Select the Machine Learning Algorithm",width=35,font="Helvetica 20 bold")
#label1.grid(row=0,sticky=W,padx=10,pady=20) #col=0


#label2=Label(root,text="Select the Model",width=35,font="Helvetica 20 bold")
#label2.grid(row=0,column=1,pady=20)

#Label2
#var2=StringVar()
#label3=Label(root,textvariable=var2)
#label3.grid(row=3,column=1)

#Image of Results
#image=Image.open("cancer_awareness.png")
#resized=image.resize((700,500),Image.ANTIALIAS)
#photo=ImageTk.PhotoImage(resized)
#label=Label(root,image=photo)
#label.image=photo
#label.grid(row=20,columnspan=4,sticky=W,padx=10,pady=10)

#mb_var=StringVar()
#mb_var.set("Model Selection")
#mb=OptionMenu(root,mb_var,())
#mb.configure(width=20)
#mb.grid(row=1,column=1)


#def reset_option_menu(options,index=None):
    
   # menu=mb["menu"]
   # menu.delete(0,"end")
   # for string in options:
    #    menu.add_command(label=string,command=lambda value=string:mb_var.set(value))
    #if index is not None:
     #   mb_var.set(options[index])

#def a():
 #   reset_option_menu(["SVM","KNN","Random Forest Tree","K-SVM","Decision Tree"],0)
    
#def default():
 #   reset_option_menu([""],0)

#def c():
 #   var="The Selected Model is "+mb_var.get()
  #  var2.set(var)
    
#def d():
 #   var=mb_var.get()

  #  if var == "SVM":
    
   #     image1=Image.open("SVM.png")
    #    resized=image1.resize((700,500),Image.ANTIALIAS)
     #   photo1=ImageTk.PhotoImage(resized)
      #  label.configure(image=photo1)
       # label.image=photo1
        
    
   # elif var == "KNN":
    #    image2=Image.open("KNN.png")
     #   resized=image2.resize((700,500),Image.ANTIALIAS)
      #  photo2=ImageTk.PhotoImage(resized)
      #  label.configure(image=photo2)
      #  label.image=photo2
    
    #elif var == "Random Forest Tree":
     #   image3=Image.open("Random_Forest_Tree.png")
      #  resized=image3.resize((700,500),Image.ANTIALIAS)
      #  photo3=ImageTk.PhotoImage(resized)
      #  label.configure(image=photo3)
      #  label.image=photo3
    
    
    #elif var == "K-SVM":
     #   image4=Image.open("K-SVM.png")
     #   resized=image4.resize((700,500),Image.ANTIALIAS)
     #   photo4=ImageTk.PhotoImage(resized)
     #   label.configure(image=photo4)
     #   label.image=photo4
    
   # elif var == "Decision Tree":
    #    image5=Image.open("Decision_Tree.png")
    #    resized=image5.resize((700,500),Image.ANTIALIAS)
    #    photo5=ImageTk.PhotoImage(resized)
    #    label.configure(image=photo5)
    #    label.image=photo5
    


#Radio Button
#var1=IntVar()
#R1=Radiobutton(root,text="Supervised Learning",variable=var1,value=1,command=a)
#R1.grid(row=1,sticky=W,padx=20)

#default()

#B=Button(root,text="Set Model",font="Helvetica 14",relief=RAISED,command=c)
#B.grid(row=3,sticky=W,padx=10,pady=10)

#B=Button(root,text="Calculate Result",font="Helvetica 14",relief=RAISED,command=d)
#B.grid(row=4,column=1,sticky=W,padx=10,pady=10)

#var2=IntVar()
#r1=Radiobutton(root,text="Wisconsin Dataset",variable=var2,value=3,command=e)


#root.mainloop()


# In[ ]:





# In[3]:


#import statements
from tkinter import * 
import tkinter as tk
import tkinter.messagebox
from tkinter import font
from PIL import ImageTk,Image
import cv2
root = Tk()

root.title("Breast Cancer Detection")
root.geometry('800x800')
label1=Label(root,text='Select the Machine Learning Algorithm and Dataset',width=45,font="Helvetica 20 bold")
label1.grid(column=0,row=0)

label2=Label(root,text='Select the Model',width=35,font="Helvetica 20 bold")
label2.grid(column=5,columnspan=50,sticky=NE)
mb_var=StringVar()
mb_var.set("Model Selection")
mb=OptionMenu(root,mb_var,())
mb.configure(width=20)
mb.grid(column=5,columnspan=33,sticky=NE,padx=10,pady=10)


B1=Button(root,text="Set Model",font="Helvetica 14",relief=RAISED,command=c)
B1.grid(sticky=E,columnspan=20,column=5,row=5,padx=10,pady=10)

B2=Button(root,text="Calculate Result",font="Helvetica 14",relief=RAISED,command=d)
B2.grid(sticky=E,columnspan=37,column=10,row=5,padx=10,pady=10)    
        

image = Image.open("cancer_awareness.png")
photo = ImageTk.PhotoImage(image,master=root)
label = tk.Label(root, image=photo)
label.image = photo
label.grid()

def reset_option_menu(options,index=None):
    
 menu=mb["menu"]
 menu.delete(0,"end")
 for string in options:
    menu.add_command(label=string,command=lambda value=string:mb_var.set(value))
 if index is not None:
     mb_var.set(options[index])

def a():
    reset_option_menu(["SVM","KNN","Random Forest Tree","K-SVM","Decision Tree"],0)
    
def default():
    reset_option_menu([""],0)

def c():
    var="The Selected Model is "+mb_var.get()
    var2.set(var)
def d():
        
    var=mb_var.get()

    if var == "SVM":
        
        image1=Image.open("SVM.png")
        resized=image1.resize((700,500),Image.ANTIALIAS)
        photo1=ImageTk.PhotoImage(resized)
        label.configure(image=photo1)
        label.image=photo1
        
        
        #label_Accuarcy.config(text="Accuracy is: ")
    
    elif var == "KNN":
        image2=Image.open("KNN.png")
        resized=image2.resize((700,500),Image.ANTIALIAS)
        photo2=ImageTk.PhotoImage(resized)
        label.configure(image=photo2)
        label.image=photo2
        
    
    elif var == "Random Forest Tree":
        image3=Image.open("Random_Forest_Tree.png")
        resized=image3.resize((700,500),Image.ANTIALIAS)
        photo3=ImageTk.PhotoImage(resized)
        label.configure(image=photo3)
        label.image=photo3
        
    
    
    elif var == "K-SVM":
        image4=Image.open("K-SVM.png")
        resized=image4.resize((700,500),Image.ANTIALIAS)
        photo4=ImageTk.PhotoImage(resized)
        label.configure(image=photo4)
        label.image=photo4
        
    
    elif var == "Decision Tree":
        image5=Image.open("Decision_Tree.png")
        resized=image5.resize((700,500),Image.ANTIALIAS)
        photo5=ImageTk.PhotoImage(resized)
        label.configure(image=photo5)
        label.image=photo5



#Radio Button
var1=IntVar()
R1=Radiobutton(root,text="Supervised Learning",variable=var1,value=1,command=a)
R1.grid(sticky=NW,column=0,row=1,padx=20,pady=10) 

var2=IntVar()
R2=Radiobutton(root,text="Wisconsin Dataset",variable=var2,value=1,command=a)
R2.grid(sticky=NW,column=0,row=2,padx=20,pady=10) 







root.mainloop()


# In[ ]:


#Position a widget in the parent widget in a grid. Use as options:
#column=number - use cell identified with given column (starting with 0)
#columnspan=number - this widget will span several columns
#in=master - use master to contain this widget
#in_=master - see 'in' option description
#ipadx=amount - add internal padding in x direction
#ipady=amount - add internal padding in y direction
#padx=amount - add padding in x direction
#pady=amount - add padding in y direction
#row=number - use cell identified with given row (starting with 0)
#rowspan=number - this widget will span several rows
#sticky=NSEW - if cell is larger on which sides will this
             # widget stick to the cell boundary


# In[ ]:





# In[107]:


#from tkinter import *
#from PIL import Image,ImageTk
#import cv2

#root=Tk()


#root.title("Breast Cancer Detection")
#root.geometry("800x800") 

#labelaccuracy
#label_Accuarcy=Label(root,text="",width=25,font="Helvetica 20 bold")
#label_Accuarcy.grid(row=20,column=500,sticky=W,padx=10,pady=10)

#Label1
#var=StringVar()
#label1=Label(root,textvariable=var,width=35,font="Helvetica 20 bold")
#var.set("Select the Machine Learning Algorithm")
#label1.grid(row=0,sticky=W,padx=10,pady=20) #col=0

#var=StringVar()
#label2=Label(root,textvariable=var,width=35,font="Helvetica 20 bold")
#var.set("Select the Model")
#label2.grid(row=0,column=1,pady=20)

#Label2
#var2=StringVar()
#label3=Label(root,textvariable=var2)
#label3.grid(row=3,column=1)

#Image of Results
#image=Image.open("cancer_awareness.png")
#resized=image.resize((700,500),Image.ANTIALIAS)
#photo=ImageTk.PhotoImage(resized)
#label=Label(root,image=photo)
#label.image=photo
#label.grid(row=20,columnspan=4,sticky=W,padx=10,pady=10)

#mb_var=StringVar()
#mb_var.set("Model Selection")
#mb=OptionMenu(root,mb_var,())
#mb.configure(width=20)
#mb.grid(row=1,column=1)

#def reset_option_menu(options,index=None):
    
# menu=mb["menu"]
# menu.delete(0,"end")
# for string in options:
 #   menu.add_command(label=string,command=lambda value=string:mb_var.set(value))
 #if index is not None:
  #  mb_var.set(options[index])

#def a():
 #   reset_option_menu(["SVM","KNN","Random Forest Tree","K-SVM","Decision Tree"],0)
    
#def default():
 #   reset_option_menu([""],0)

#def c():
   # var="The Selected Model is "+mb_var.get()
   # var2.set(var)
    
#def d():
 #   var=mb_var.get()

  #  if var == "SVM":
    
   #     image1=Image.open("SVM.png")
   #     resized=image1.resize((700,500),Image.ANTIALIAS)
   #     photo1=ImageTk.PhotoImage(resized)
   #     label.configure(image=photo1)
    #    label.image=photo1
       # label_Accuarcy.config(text="Accuracy is: ",)
    
    #elif var == "KNN":
    #    image2=Image.open("KNN.png")
    #    resized=image2.resize((700,500),Image.ANTIALIAS)
    #    photo2=ImageTk.PhotoImage(resized)
    #    label.configure(image=photo2)
    #    label.image=photo2
    
   # elif var == "Random Forest Tree":
    #    image3=Image.open("Random_Forest_Tree.png")
    #    resized=image3.resize((700,500),Image.ANTIALIAS)
    #    photo3=ImageTk.PhotoImage(resized)
    #    label.configure(image=photo3)
    #    label.image=photo3
    
    
    #elif var == "K-SVM":
     #   image4=Image.open("K-SVM.png")
     #   resized=image4.resize((700,500),Image.ANTIALIAS)
     #   photo4=ImageTk.PhotoImage(resized)
     #   label.configure(image=photo4)
     #   label.image=photo4
    
    #elif var == "Decision Tree":
      #  image5=Image.open("Decision_Tree.png")
      #  resized=image5.resize((700,500),Image.ANTIALIAS)
      #  photo5=ImageTk.PhotoImage(resized)
      #  label.configure(image=photo5)
      #  label.image=photo5
    


#Radio Button
#var1=IntVar()
#R1=Radiobutton(root,text="Supervised Learning",variable=var1,value=1,command=a)
#R1.grid(row=1,sticky=W,padx=20)

#default()

#B=Button(root,text="Set Model",font="Helvetica 14",relief=RAISED,command=c)
#B.grid(row=3,sticky=W,padx=10,pady=10)

#B=Button(root,text="Calculate Result",font="Helvetica 14",relief=RAISED,command=d)
#B.grid(row=4,column=1,sticky=W,padx=10,pady=10)

#var2=IntVar()
#r1=Radiobutton(root,text="Wisconsin Dataset",variable=var2,value=3,command=e)


#root.mainloop()


# In[ ]:




