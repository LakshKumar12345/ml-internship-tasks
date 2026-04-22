#!/usr/bin/env python
# coding: utf-8

# # 1. Import Libraries
# 
# Libraries like Pandas, Seaborn, and Matplotlib were imported to handle data and create visualizations.

# In[57]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## 2. Load Dataset

# In[58]:


df = pd.read_csv("iris.csv")


#  the Iris dataset was loaded using read_csv() to bring the data into a structured DataFrame for analysis.

# # 3. Dataset Shape

# In[59]:


df.shape


#  Checked dataset dimensions (rows and columns) to understand dataset size and structure.

# # 4. print first 5 rows

# In[60]:


df.head()


#  Previewed first 5 rows to verify dataset loading and understand data format.

# # 5. Column Names

# In[61]:


print(pd.Series(df.columns))


# We displayed column names to understand what features are available in the dataset.

# # 6.Dataset Info

# In[62]:


df.info()


# we checked data types, missing values, and dataset structure to understand data quality.

# # 7.Dataset Describe

# In[63]:


print(df.describe())


# We generated statistical summary to understand mean, min, max, and distribution of features.

# ## 8. Scatter plot

# In[64]:


import matplotlib.pyplot as plt
sns.scatterplot(x="SepalLengthCm",y="SepalWidthCm",hue="Species",data=df)
plt.show()


# 
# We used scatter plot to understand relationship between sepal features and species separation.

# In[65]:


import matplotlib.pyplot as plt
sns.scatterplot(x="PetalLengthCm",y="PetalWidthCm",hue="Species",data=df)
plt.show()


# We used scatter plot to understand relationship between Petal features and species separation.

# # 9.histogram

# In[80]:


df.hist(figsize=(10,8), bins=20)
plt.suptitle("Histogram of Iris Features")
plt.show()


# we used histograms to understand distribution and frequency of each feature

# # 10. Boxplot  for outliers

# In[68]:


plt.figure(figsize=(10,6))

sns.boxplot(data=df.iloc[:,1:])  # sirf features

plt.title("Boxplot for Outlier Detection")
plt.show()


# We used boxplot to detect outliers and understand data spread using quartiles and median values.

# # 11.  Pair plot

# In[81]:


sns.pairplot(df.iloc[:,1:], hue="Species")
plt.show()


# We used pairplot to visualize relationships between all features and observe class separation in one view.

# #  Conlusion
# 

# This exploratory data analysis helped us understand dataset structure, feature distribution, relationships, and outliers. Visualizations clearly show that the Iris dataset is well-suited for classification tasks.

# In[ ]:




