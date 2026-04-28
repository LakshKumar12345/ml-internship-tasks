#!/usr/bin/env python
# coding: utf-8

# # 🏠 House Price Prediction 

# ## 1. Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## 2. Load Dataset

# In[2]:


df = pd.read_csv("House_Price_Prediction_Dataset.csv")
df.head()


# ## 3. Data Exploration

# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# ## 4. Visualization

# In[7]:


df_sample = df.sample(200)
plt.scatter(df_sample["Area"], df_sample["Price"],)
plt.ticklabel_format(style='plain')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()


# In[10]:


sns.barplot(x="Location", y="Price", data=df,estimator="mean")
plt.title("Location vs Price")
plt.show()


# In[13]:


df_sample = df.sample(100)
sns.scatterplot(x="Area", y="Price", hue="Location", data=df_sample)
plt.ticklabel_format(style='plain')
plt.title("Area vs Price with Location")
plt.show()


# ## 5. Data Preprocessing

# In[14]:


df.drop("Id", axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)


# ## 6. Feature Selection

# In[15]:


X = df.drop("Price", axis=1)
y = df["Price"]


# ## 7. Train Test Split

# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## 8. Linear Regression

# In[23]:


from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)


# In[24]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

print("Linear MAE:", lr_mae)
print("Linear RMSE:", lr_rmse)


# 

# In[27]:


import matplotlib.pyplot as plt

plt.scatter(y_test, lr_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.ticklabel_format(style='plain')
plt.show()


# In[ ]:





# In[33]:


comparsion=pd.DataFrame({
    "Actual Price":y_test,
    "Predicted Price":lr_pred
})
comparsion["error"]=comparsion["Actual Price"]-comparsion["Predicted Price"]
comparsion.head(5)


# ### Insight
# 
# Most predictions are close to actual values, showing that the model has learned the underlying patterns well. However, some larger errors indicate variability in housing data.

# ## 9. Gradient Boosting

# In[19]:


from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.02, max_depth=2)
gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)


# In[20]:


gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))

print("GB MAE:", gb_mae)
print("GB RMSE:", gb_rmse)


# In[34]:


errors = y_test - gb_pred
sns.histplot(errors, kde=True)
plt.title("Gradient Boosting Error Distribution")
plt.show()


# In[45]:


importance = pd.Series(gb_model.feature_importances_, index=X.columns)
k=importance.sort_values(ascending=False)
plt.barh(features,k)
plt.title("Feature Importance (Gradient Boosting)")
plt.show()


# ### Feature importance shows that Area is the most influential factor in the model. However, Location impact is distributed across multiple encoded features, which reduces its individual importance score.

# ## 10. Comparison

# In[21]:


print("Linear MAE:", lr_mae)
print("Gradient Boosting MAE:", gb_mae)


# ### both have almost same scores  

# ## 11. Conclusion

# 
# 
# In this project, we built machine learning models to predict house prices using features such as area, bedrooms, bathrooms, location, and condition.
# 
# ### 📊 Key Findings:
# -  Area is the most influential factor in the model. However, Location impact is distributed across multiple encoded features, which reduces its individual importance score. -
# 
# ### Comparison
# - Both Linear Regression and Gradient Boosting models were applied
# - Both models performed similarly with no significant improvement from the complex model
# 
# ### 🧠 Final Insight:
# This indicates that the dataset has mostly linear relationships, and a simple Linear Regression model is sufficient for this prediction task.
# 
# ### 🚀 Overall:
# The model successfully captures the main patterns in the data and provides reasonably accurate price predictions. 
# 

# In[ ]:




