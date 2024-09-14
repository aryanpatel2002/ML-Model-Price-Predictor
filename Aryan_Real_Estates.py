#!/usr/bin/env python
# coding: utf-8

# ## Aryan Real Estate Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


housing.hist(bins=50, figsize=(20,15))


# ## Train-Test Splitting

# In[7]:


import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:] 
    return data.iloc[train_indices], data.iloc[test_indices]


# In[8]:


train_set, test_set = split_train_test(housing, 0.2)


# In[9]:


print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[10]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[11]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[12]:


strat_test_set['CHAS'].value_counts()


# In[13]:


strat_train_set['CHAS'].value_counts()


# ## Correlations

# In[14]:


corr_matrix = housing.corr()


# In[15]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[16]:


from pandas.plotting import scatter_matrix
attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
scatter_matrix(housing[attributes], figsize = (12, 8))


# In[17]:


housing.plot(kind = "scatter", x="RM", y="MEDV", alpha=0.8)


# In[18]:


housing = strat_train_set.drop("MEDV", axis = 1)
housing_labels =  strat_train_set["MEDV"].copy()


# ## Treating Missing Values

# In[19]:


##1. Removing missing values.
##2. Removing attribute
##3. Treating with mean or median.


# In[20]:


housing.dropna(subset=['RM']) #option1


# In[21]:


housing.drop('RM', axis=1) #option2


# In[22]:


median  = housing["RM"].median()


# In[23]:


housing["RM"].fillna(median) #option3


# In[24]:


from sklearn.impute import SimpleImputer #option3alternate
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[25]:


imputer.statistics_


# In[26]:


X = imputer.transform(housing)


# In[27]:


housing_tr = pd.DataFrame(X, columns=housing.columns)


# In[28]:


housing_tr.describe()


# ## Pipeline

# In[29]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])


# In[30]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[31]:


housing_num_tr


# ## Selecting a model for our company

# In[93]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
#model = LinearRegression()
#model = DecisionTreeRegressor()
model.fit(housing_num_tr, housing_labels)


# In[94]:


some_data = housing.iloc[:5]


# In[95]:


some_labels = housing_labels.iloc[:5]


# In[96]:


prepared_data = my_pipeline.transform(some_data)


# In[97]:


model.predict(prepared_data)


# In[98]:


list(some_labels)


# ## Evaluating our model

# In[99]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[100]:


rmse


# ## Cross Validation

# In[101]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)


# In[102]:


rmse_scores


# In[103]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())


# In[104]:


print_scores(rmse_scores)


# ## Saving the model

# In[118]:


from joblib import dump, load
dump(model, 'Aryan.joblib')


# ## Testing model on test data

# In[119]:


X_test = strat_test_set.drop('MEDV', axis=1)
Y_test = strat_test_set['MEDV'].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
#print(final_predictions, list(Y_test))


# In[120]:


final_rmse


# In[123]:


prepared_data[0]


# ## Using the model

# In[124]:


from joblib import dump, load
import numpy as np
model = load('Aryan.joblib')
feature = np.array([[-5.43942006,  2.12628155, -4.12165014, -0.27288841, -1.42262747,
       -0.23979304, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(feature)

