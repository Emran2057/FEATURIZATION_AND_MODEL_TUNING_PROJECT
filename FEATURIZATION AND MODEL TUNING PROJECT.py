#!/usr/bin/env python
# coding: utf-8

#    # FEATURIZATION AND MODEL TUNING PROJECT
# 
# ******************************************************************************
# 
# **Data Description**
# 
# The actual concrete compressive strength(MPa) for a given mixture under a specific age (days) was determined from laboaratory. Data is in raw form(not scaled). The data has 8 quantitative input variables, and 1 quantitaive output variable and 1030 instances (observations)
# 
# **Domain**
# 
# Cement Manufacturing
# 
# **Context**
# 
# Concrete is the most important in civil enginerring, The concrete compressive strength is a highly non linear function of age and ingredients. These ingredients include cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, and fine aggreagate.
# 
# **Attribute** **Information**
# 
# * Cement           : measured in kg in a m3 mixture
# * Blast            : measured in kg in a m3 mixture
# * Fly ash          : measured in kg in a m3 mixture
# * Water            : measured in kg in a m3 mixture
# * Superplasticizer : measured in kg in a m3 mixture
# * Coarse Aggregate : measured in kg in a m3 mixture
# * Fine Agrregate   : measured in kg in a m3 mixture
# * Age              : day (1~365)
# * Concrete compressive strength measured in Mpa
# 
# **Objective**
# 
# Modeling of strength of high performance concrete using Machine Learning
# 
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# 
# **importing the libraries**

# In[1]:


import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler , FunctionTransformer , OneHotEncoder, PowerTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# **importing the data**

# In[2]:


cement = pd.read_csv(r"C:\Users\emran\Downloads\concrete.csv")


# In[3]:


cement.head()


# **Summary of data**

# In[4]:


cement.shape


# In[5]:


cement.info()


# In[6]:

# # Univariate Analysis
cement.describe()


# *Conclusions that can be made out of it:*
# 
# * all the features given in the data are numeric
# * There are no NULL or NaN values present in the data
# * In total there are 8 features and 1030 observations
# * We shall be using strength as the dependent variable and all other features as independent variable as it makes the best sense out of the problem objective.
# * Age value ranges from 1 to 365

# In[7]:


fig = px.box(cement, x="cement")
fig.show()


# In[8]:


fig = px.box(cement, x="slag")
fig.show()


# In[9]:


fig = px.box(cement, x="ash")
fig.show()


# In[10]:


fig = px.box(cement, x="water")
fig.show()


# In[11]:


fig = px.box(cement, x="superplastic")
fig.show()


# In[12]:


fig = px.box(cement, x="coarseagg")
fig.show()


# In[13]:


fig = px.box(cement, x="fineagg")
fig.show()


# In[14]:


fig = px.box(cement, x="age")
fig.show()


# In[15]:


fig = px.box(cement, x="strength")
fig.show()


# * from the plotly we can plotted and individual attribute's boxplots which were plotted before are the evidences that there are outliers present in the attributes: slag, ash, superplastic, fineagg, age

# In[16]:

# # Multivariate analysis
sns.pairplot(cement)


# *Conclusion that can be drawn from bi-variate analysis:*
# 
# * cement and strength are the 2 attributes which looks more normal than other attributes
# * slag, ash, superplastic, age are the attributes which are rightly skewed.
# * Except cement and strength all other attributes have multiple/ more than one Gaussians.

# In[17]:

# # Split the data
from sklearn.model_selection import train_test_split

cement["cement_cat"]=pd.cut(cement['cement'], bins=[100, 150, 200, 250, 300, 350, 400, 450, 500, np.inf], 
                              labels=[1, 2, 3, 4, 5, 6, 7, 8, 9])
cement.head()


# In[18]:


strat_train_set, strat_test_set = train_test_split(cement, test_size=0.3, stratify=cement["cement_cat"], random_state=42)


# In[19]:


strat_train_set.drop("cement_cat", axis=1, inplace=True)
strat_test_set.drop("cement_cat", axis=1, inplace=True)


# In[20]:


strat_train_set


# In[21]:


strat_test_set


# *Conclusion that can be drawn from split data:*
# 
# * cement attribute is more useful or important attribute to predict the strength of concrete.
# * training data shold be representative of data.
# * so we used stratified sampling and also split the data into 70% for training data and 30% for testing data.

# # Relationship between dependent and independent variables

# In[22]:


cement = strat_train_set.copy()


# In[23]:


cement.corr()


# In[25]:


plt.figure(figsize=(10,6))
sns.heatmap(cement.corr(), annot= True, cmap='BuPu', linecolor='black')
plt.title("Correlation between the variables")


# *Conclusion that can be drawn from relationship between dependent and independent variables:*
# 
# * Strength vs Cement: is highly positively correlated, it's also linearly related
# * Strength vs Slag: is positively correalted with less degree of correaltion
# * Strength vs Ash: is negatively correlated of almost -0.11
# * Strength vs Water:- Negatively correalted and poorly realted to dependent variable
# * Strength vs Superplastic: positive and fairly correalted of almost 0.37
# * Strength vs coarseagg/fineagg : negatively correalted
# * Strength vs age: positively and fairly correalted

# *--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*

# **Make each attributes into Gausian distributions**

# In[29]:


sy = PowerTransformer("yeo-johnson")
df1 = sy.fit_transform(cement)


# In[30]:


sy.lambdas_


# **Check how much outliers is in the training set**

# In[45]:


df1 = cement[(cement["slag"]>350) | (cement["water"]>228) | (cement["superplastic"]>23.4) | 
       ((cement["fineagg"]>945.0)) | (cement["age"]>120)]


# In[46]:


df1.shape


# * There total 54 outliers in the training set.
# * But we don't remove this outliers because it is right information and also it will use full for predictions.

# *------------------------------------------------------------------------------------------------------------------------------------------------------------*

# # Creating pipeline for data preprocessing

# In[47]:


cement = strat_train_set.drop("strength", axis=1)
cement_labels = strat_train_set["strength"].copy()


# In[51]:


num_pipeline = make_pipeline(SimpleImputer(strategy="median"), PowerTransformer("yeo-johnson"), StandardScaler())
preprocessing = ColumnTransformer([("num", num_pipeline, ['cement', 'slag', 'ash', 'water', 'superplastic', 
                                                          'coarseagg', 'fineagg', 'age']),])


# In[52]:


cement_prepared = preprocessing.fit_transform(cement)
cement_prepared.shape


# In[53]:


preprocessing.get_feature_names_out()


# *In this pipeline we used imputation if there is filling value, converting attributes into gauisan distribution and scaling variables.*

# *-----------------------------------------------------------------------------------------------------------------------------------------------------------------------*

# # Select and train model

# **Algorithm**:- *Linear Regression*

# In[54]:


from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(cement, cement_labels)


# In[57]:


lin_predictions = lin_reg.predict(cement)
lin_predictions[:5].round(2)


# In[58]:


cement_labels.iloc[:5]


# In[61]:


lin_rmse = mean_squared_error(cement_labels, lin_predictions, squared=False)
print("RMSE: ", lin_rmse)


# In[62]:


from sklearn import metrics
lin_reg_r2 = metrics.r2_score(cement_labels, lin_predictions)
print("R^2 Score:", lin_reg_r2)
print("Mean Squared Error:", metrics.mean_squared_error(cement_labels, lin_predictions))


# **Algorithm**:- *Decision Tree Regressor*

# In[63]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor())
tree_reg.fit(cement, cement_labels)


# In[64]:


tree_predictions = tree_reg.predict(cement)
tree_predictions[:5].round(2)


# In[65]:


tree_rmse = mean_squared_error(cement_labels, tree_predictions, squared=False)
print("RMSE: ", tree_rmse)


# *RMSE is 1.03 but training data overfit the model so we used validation set i.e cross validation method for evaluating the model.*

# In[66]:


tree_rmses = -cross_val_score(tree_reg, cement, cement_labels, cv=10, scoring="neg_root_mean_squared_error")


# In[67]:


pd.Series(tree_rmses).describe()


# In[68]:


tree_reg_r2 = metrics.r2_score(cement_labels, tree_predictions)
print("R^2 Score:", tree_reg_r2)
print("Mean Squared Error:", metrics.mean_squared_error(cement_labels, tree_predictions))


# **Algorithm**:- *Random Forest Regressor*

# In[74]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_reg.fit(cement, cement_labels)


# In[75]:


forest_predictions = forest_reg.predict(cement)
forest_predictions[:5].round(2)


# In[76]:


forest_rmse = mean_squared_error(cement_labels, forest_predictions, squared=False)
print("RMSE: ", forest_rmse)


# In[80]:


forest_rmses = -cross_val_score(forest_reg, cement, cement_labels, cv=10, scoring="neg_root_mean_squared_error")
pd.Series(forest_rmses).describe()


# In[78]:


forest_reg_r2 = metrics.r2_score(cement_labels, forest_predictions)
print("R^2 Score:", forest_reg_r2)
print("Mean Squared Error:", metrics.mean_squared_error(cement_labels, forest_predictions))


# In[81]:


lin_rmses = -cross_val_score(lin_reg, cement, cement_labels, cv=10, scoring="neg_root_mean_squared_error")
pd.Series(lin_rmses).describe()


# **Algorithm**:- *Support Vector Regressor*

# In[83]:


from sklearn.svm import SVR

svr_reg = make_pipeline(preprocessing, SVR())
svr_reg.fit(cement, cement_labels)


# In[84]:


svr_predictions = svr_reg.predict(cement)
svr_predictions[:5].round(2)


# In[85]:


svr_rmses = -cross_val_score(svr_reg, cement, cement_labels, cv=10, scoring="neg_root_mean_squared_error")
pd.Series(svr_rmses).describe()


# Conclusion that can be drawn from model selection:*
# 
# * Random Forest Regressor is the best model for this because mean of rmses is very low compare to other.
# * So we will hyperparameter tune to random forest regressor.

# *------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*

# **Hyper Parameter Tuning of Random Forest Regressor**

# In[89]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

full_pipeline = Pipeline([("preprocessing", preprocessing), ("random_forest", RandomForestRegressor(random_state=42))])
param_distribs = {'random_forest__max_features': randint(low=2, high=9), 
                 'random_forest__n_estimators': randint(low=0, high=300),
                 'random_forest__bootstrap':[True, False],
                 'random_forest__max_depth': randint(low=0, high=300),
                 'random_forest__criterion': ['squared_error', 'absolute_error']}

rnd_search = RandomizedSearchCV(full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3, 
                               scoring="neg_root_mean_squared_error")
rnd_search.fit(cement, cement_labels)


# In[91]:


-rnd_search.best_score_


# In[92]:


rnd_search.best_params_


# *We used randomizedsearchcv because it help to find the best parameters for fixed range.*

# *------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*

# **Final model**

# In[95]:


final_model = rnd_search.best_estimator_
final_model


# In[96]:


feature_importances = final_model["random_forest"].feature_importances_
feature_importances.round(2)


# In[97]:


sorted(zip(feature_importances, final_model["preprocessing"].get_feature_names_out()), reverse=True)


# # Evaluate in test set

# In[98]:


X_test = strat_test_set.drop("strength", axis=1)
y_test = strat_test_set["strength"].copy()

final_predictions = final_model.predict(X_test)

final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
print("RMSE:", final_rmse)


# In[100]:


final_model_r2 = metrics.r2_score(y_test, final_predictions)
print("R^2 Score:", final_model_r2)


# In[103]:


print(f"Accuracy of Hyperparameter tuned Random Forest Regressor is {100*final_model_r2}.")


# *We can conclude that RMSE is low for test data so it show that it is generalized and R squared is 0.90 so we can say that 90% is the accuracy of our hyperparameter tune Random Forest Regressor model.*


from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))

# *We can say that RMSE lies between [4.60862946, 5.92716145] at 95% cofidence level.*
