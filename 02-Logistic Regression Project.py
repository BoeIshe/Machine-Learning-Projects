#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Logistic Regression Project 
# 
# In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Import Libraries
# 
# **Import a few libraries you think you'll need (Or just import them as you go along!)**

# In[2]:


import pandas as pd
import numpy as np


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**

# In[9]:


ad_data = pd.read_csv('advertising.csv')


# In[ ]:





# **Check the head of ad_data**

# In[10]:


ad_data.head()


# In[ ]:





# In[ ]:





# ** Use info and describe() on ad_data**

# In[11]:


ad_data.info()


# In[12]:


ad_data.describe()


# In[ ]:





# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# Try recreating the plots shown below!
# 
# ** Create a histogram of the Age**

# In[16]:


sns.set_style('whitegrid')
ad_data['Age'].plot.hist(bins=30)


# In[ ]:





# **Create a jointplot showing Area Income versus Age.**

# In[18]:


sns.jointplot(x= 'Age', y= 'Area Income', data= ad_data )


# In[ ]:





# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

# In[39]:


sns.jointplot(x= 'Age', y= 'Daily Time Spent on Site', data= ad_data, kind= 'kde', color='red', palette='seismic' )


# In[ ]:





# In[ ]:





# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# In[46]:


sns.jointplot(x= 'Daily Time Spent on Site', y= 'Daily Internet Usage', data= ad_data, color= 'green')


# In[ ]:





# ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

# In[49]:


sns.pairplot(ad_data, hue= 'Clicked on Ad', palette= 'coolwarm')


# In[ ]:





# # Logistic Regression
# 
# Now it's time to do a train test split, and train our model!
# 
# You'll have the freedom here to choose columns that you want to train on!

# ** Split the data into training set and testing set using train_test_split**

# In[119]:


X = ad_data.drop(['Clicked on Ad', 'Ad Topic Line', 'City','Country','Timestamp','Clicked on Ad'], axis=1)


# In[120]:


y= ad_data['Clicked on Ad']


# In[121]:


from sklearn.model_selection import train_test_split 


# In[122]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)    


# In[ ]:





# ** Train and fit a logistic regression model on the training set.**

# In[123]:


from sklearn.linear_model import LogisticRegression


# In[124]:


logmodel = LogisticRegression()


# In[125]:


logmodel.fit(X_train, y_train)


# In[ ]:





# ## Predictions and Evaluations
# ** Now predict values for the testing data.**

# In[126]:


predictions = logmodel.predict(X_test)


# In[ ]:





# ** Create a classification report for the model.**

# In[127]:


from sklearn.metrics import classification_report


# In[128]:


print(classification_report(y_test, predictions))


# In[ ]:





# In[95]:




