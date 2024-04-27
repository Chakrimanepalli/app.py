#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# The objective is to develop a predictive model for energy production in a combined-cycle power plant. 
# 
# The plant comprises gas turbines, steam turbines, and heat recovery steam generators, where electricity is generated through a combined cycle of gas and steam turbines, with energy transfer between turbines. 
# 
# The model will be based on exhaust vacuum and ambient variables. The aim is to enhance the power plant's performance by accurately predicting energy production.

# # Data Set Details

# This project focuses on predicting the variable of energy production. 
# The dataset comprises 9568 observations collected over six years from a combined-cycle power plant operating at full load.

# The dataset includes five variables:

# Temperature: measured in degrees Celsius.
# 
# Exhaust Vacuum: measured in cm Hg.
# 
# Ambient Pressure: measured in millibar.
# 
# Relative Humidity: measured in percentage.
# 
# Energy Production: measured in MW, representing the net hourly electrical energy output.
# 

# # IMPORT LIBRARIES

# In[1]:


# Imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stat
import warnings
warnings.filterwarnings('ignore')

# Specific imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler 
import statsmodels.formula.api as smf 
from scipy.stats import shapiro
import pylab


# # IMPORT DATASET

# In[2]:


filepath="C:\\Users\\chakri\\Downloads\\energy_production.csv"
df=pd.read_csv(filepath,sep=';')


# In[3]:


data=df.copy()


# # Exploratory Data Analysis (EDA)

# In[4]:


df.head()


# # CHECKING FOR DATA TYPES

# In[5]:


df.info()


# # DESCRIPTIVE ANALYSIS

# In[6]:


df.describe()


# # CHECKING FOR MISSING VALUES

# In[7]:


df.isnull().sum()


# These values indicate that there are no null (missing) values in any of the variables. All variables have a count of 0 null values, meaning the dataset is complete with no missing data for any of the variables.

# # VISUALIZING MISSING VALUES

# In[8]:


plt.figure(figsize=(12,8))
sns.heatmap(df.isnull(),cmap='viridis')


# In[9]:


df[df.values==0.0]


# In[10]:


df.shape


# In[11]:


# Sweetviz


# In[12]:


pip install pandas_profiling[notebook]


# In[13]:


get_ipython().system('pip install sweetviz')


# In[14]:


import sweetviz as sv
sweet_report=sv.analyze(df)
sweet_report.show_html("energy_production")


# # CHECKING OF DUPLICATE VALUES

# In[15]:


df[df.duplicated()].shape


# In[16]:


df[df.duplicated()]


# # REMOVING DUPLICATE VALUES

# In[17]:


df1 = df.drop_duplicates()


# In[18]:


df1[df1.duplicated()]


# In[19]:


sweet_report=sv.analyze(df1)
sweet_report.show_html("energy_production")


# In[20]:


# Skewness
skewness = df1.skew()
print("Skewness:")
print(skewness)


# Temperature and Ambient Pressure have slightly negative skewness.
# 
# Exhaust Vacuum has slightly positive skewness.
# 
# Relative Humidity has more pronounced negative skewness.
# 
# Energy Production has positive skewness.

# In[21]:


# Kurtosis
kurtosis = df.kurtosis()
print("\nKurtosis:")
print(kurtosis)


# Temperature, Exhaust Vacuum, and Energy Production have negative kurtosis, indicating distributions with lighter tails than a normal distribution.
# 
# Ambient Pressure has positive kurtosis, indicating a distribution with slightly heavier tails than a normal distribution.
# 
# Relative Humidity has slightly negative kurtosis, suggesting a distribution with lighter tails.

# # EXPLORATORY DATA ANALYSIS

# In[22]:


plt.figure(figsize=(10, 8))
sns.histplot(data=df1, x='energy_production', bins=30, kde=True)
plt.title('Distribution of Energy Production')
plt.xlabel('Energy Production (MW)')
plt.ylabel('Frequency')
plt.show()


# In[23]:


plt.figure(figsize=(12, 8))
for i, column in enumerate(df1.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df1[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# In[24]:


sns.pairplot(data=df1)


# In[25]:


df1.corr()


# Temperature and Energy Production have a strong negative correlation (-0.947908).
# 
# Exhaust Vacuum and Energy Production also have a strong negative correlation (-0.869900).
# 
# Ambient Pressure and Energy Production have a moderate positive correlation (0.518687).
# 
# Relative Humidity and Energy Production have a moderate positive correlation (0.391175).
# 
# Correlation coefficients measure the strength and direction of the linear relationship between two variables. A value close to 1 or -1 indicates a strong correlation, while a value close to 0 indicates a weak correlation. 

# In[26]:


# Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df1.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[27]:


def find_outliers(df):
    # Calculate the first and third quartiles
    Q1 = np.percentile(df1, 25)
    Q3 = np.percentile(df1, 75)
    
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    
    # Define the lower and upper bounds for outliers detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find outliers
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    
    return outliers


# In[28]:


# Box plots to check for outliers
plt.figure(figsize=(12, 6))

# Box plot for temperature
plt.subplot(2, 2, 1)
sns.boxplot(x=df1['temperature'], color='skyblue')
plt.title('Box Plot for Temperature')


# In[29]:


# Box plot for exhaust vacuum
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 2)
sns.boxplot(x=df1['exhaust_vacuum'], color='salmon')
plt.title('Box Plot for Exhaust Vacuum')


# In[30]:


# Box plot for ambient pressure
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 3)
sns.boxplot(x=df1['amb_pressure'], color='lightgreen')
plt.title('Box Plot for Ambient Pressure')


# In[31]:


# Box plot for relative humidity
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 4)
sns.boxplot(x=df1['r_humidity'], color='gold')
plt.title('Box Plot for Relative Humidity')


# In[32]:


#box plot for 'energy_production'
plt.figure(figsize=(8, 6))
sns.boxplot(x=df1['energy_production'], color='skyblue')
plt.title('Box Plot of Energy Production')
plt.xlabel('Energy Production (MW)')
plt.show()


# In[33]:


first_quartile = df1.quantile(0.25)
print("First Quartile (25th percentile):", first_quartile)


# In[34]:


second_quartile = df1.quantile(0.50)
print("Second Quartile (Median or 50th percentile):", second_quartile)


# In[35]:


third_quartile = df1.quantile(0.75)
print("Third Quartile (75th percentile):", third_quartile)


# In[36]:


iqr = third_quartile - first_quartile
print("Inter-Quartile Range (IQR):", iqr)


# In[37]:


upper_whisker = third_quartile + 1.5 * iqr
print("Upper Whisker:", upper_whisker)


# In[38]:


lower_whisker = first_quartile - 1.5 * iqr
print("Lower Whisker:", lower_whisker)


# # Model Training and Evaluation

# 
# #  Data Preparation:

# In[39]:


# Split the dataset into features (X) and target variable (y)
X = df1[['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity']]
y = df1['energy_production']


# # Model Training

# In[40]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[41]:


# Initialize regression models
linear_reg = LinearRegression()
lasso_reg = Lasso(alpha=0.1)
ridge_reg = Ridge(alpha=0.1)
elastic_net_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
random_forest_reg = RandomForestRegressor(n_estimators=100)
decision_tree_reg = DecisionTreeRegressor()
xgboost_reg = XGBRegressor()
gradient_boosting_reg = GradientBoostingRegressor()


# In[42]:


# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Once the model is trained, you can use it to make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance using appropriate metrics
from sklearn.metrics import mean_squared_error, r2_score

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared
r_squared = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared:", r_squared)


# # OLS MODEL

# In[43]:


#  the independent variables (features)
X = df1[['exhaust_vacuum', 'amb_pressure', 'r_humidity', 'temperature']]

# Add a constant term to the features
X = sm.add_constant(X)

# the dependent variable
y = df1['energy_production']

# OLS regression model
raw_model = sm.OLS(y, X).fit()
raw_model.summary()


# In[44]:


raw_model.params


# # Let's try Median Imputation to handle Outlier

# In[45]:


df3=df1.copy()


# In[46]:


df_median_imputed = df3.copy()


# In[47]:


median_energy_production = df1['r_humidity'].median()


# In[48]:


median_r_humidity = df3['r_humidity'].median()
print("Median r_humidity:", median_r_humidity)


# In[49]:


sns.boxplot(df1['r_humidity'])
plt.title('r_humidity before median imputation')


# In[50]:


for i in df3['r_humidity']:
    q1 = np.quantile(df.r_humidity,0.25)
    q3 = np.quantile(df.r_humidity,0.75)
    med = np.median(df.r_humidity)
    iqr = q3 - q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    if i > upper_bound or i < lower_bound:
        df3['r_humidity'] = df3['r_humidity'].replace(i, np.median(df3['r_humidity']))
sns.boxplot(df3['r_humidity'])
plt.title('r_humidity after median imputation')
plt.show()


# In[51]:


# Calculate the median ambient pressure
median_amb_pressure = df3['amb_pressure'].median()

# Create a copy of the DataFrame


# Display median humidity
print("Median Amb_pressure:", median_amb_pressure)

# Display boxplot before median imputation
sns.boxplot(df3['amb_pressure'])
plt.title('Ambient Pressure before median imputation')
plt.show()


# In[52]:


# Perform median imputation for outliers in the 'amb_pressure' column
for i in df3['amb_pressure']:
    q1 = np.quantile(df.amb_pressure, 0.25)
    q3 = np.quantile(df.amb_pressure, 0.75)
    med = np.median(df.amb_pressure)
    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    if i > upper_bound or i < lower_bound:
        df3['amb_pressure'] = df3['amb_pressure'].replace(i, np.median(df3['amb_pressure']))


# In[53]:


sns.boxplot(df3['amb_pressure'])
plt.title('Ambient Pressure after median imputation')
plt.show()


# Observation: As you can see after median imputation the model is not performing well

# In[54]:


X = df3[['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity']]
y = df3['energy_production']


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[56]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[57]:


y_pred = model.predict(X_test)


# In[58]:


mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r_squared)


# In[59]:


df_median_imputed=df3
# Define the independent variables (features)
X_median = df_median_imputed[['exhaust_vacuum', 'amb_pressure', 'r_humidity', 'temperature']]

# Add a constant term to the features
X_median = sm.add_constant(X_median)

# Define the dependent variable
y_median = df_median_imputed['energy_production']

# Fit OLS regression model
model_median_imputation = sm.OLS(y_median, X_median).fit()
model_median_imputation.summary()


# In[60]:


model_median_imputation.params


# # Let's try Mean Imputation to handle Outlier¶

# In[61]:


mean_amb_pressure = df1['amb_pressure'].mean()

print("Mean Amb_pressure:", mean_amb_pressure)


# In[62]:


#  boxplot before mean imputation
sns.boxplot(df1['amb_pressure'])
plt.title('Ambient Pressure before mean imputation')
plt.show()


# In[63]:


# Perform mean imputation for outliers in the 'amb_pressure' column
for i in df3['amb_pressure']:
    q1 = np.quantile(df.amb_pressure, 0.25)
    q3 = np.quantile(df.amb_pressure, 0.75)
    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    if i > upper_bound or i < lower_bound:
        df3['amb_pressure'] = df3['amb_pressure'].replace(i, mean_amb_pressure)


# In[64]:


# Display boxplot after mean imputation
sns.boxplot(df3['amb_pressure'])
plt.title('Ambient Pressure after mean imputation')
plt.show()


# In[ ]:





# In[65]:


# Split the dataset into features (X) and target variable (y)
X = df3[['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity']]
y = df3['energy_production']


# In[66]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[67]:


# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[68]:


# Predict the target variable on the test set
y_pred = model.predict(X_test)


# In[69]:


# Calculate model performance metrics
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r_squared)


# In[70]:


import statsmodels.api as sm
df_mean_imputed=df3
# Define the independent variables (features)
X_mean = df_mean_imputed[['exhaust_vacuum', 'amb_pressure', 'r_humidity', 'temperature']]

# Add a constant term to the features
X_mean = sm.add_constant(X_mean)

# Define the dependent variable
y_mean = df_mean_imputed['energy_production']

# Fit OLS regression model
mean_imputed_model = sm.OLS(y_mean, X_mean).fit()
mean_imputed_model.summary()


# In[71]:


mean_imputed_model.params


# In[72]:


df3.columns


# Observation: As you can see after mean imputation the model is not performing well

# # Visualizing the Distribution of Independent Features with the help of Histograms

# In[73]:


feature = ['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity']
def plot_data(df3, feature):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    df3[feature].hist()
    plt.subplot(1, 2, 2)
    stat.probplot(df3[feature], dist='norm', plot=pylab)


# In[74]:


plot_data(df3, 'temperature')
plt.show()
plot_data(df3, 'exhaust_vacuum')
plt.show()
plot_data(df3, 'amb_pressure')
plt.show()
plot_data(df3, 'r_humidity')
plt.show()


# # The Shapiro-Wilk test is a test of normality. It is used to determine whether or not a sample comes from a normal distribution.

# In[75]:


from scipy.stats import shapiro
statistic, p_value = shapiro(df3)

print("Shapiro-Wilk Test Statistic:", statistic)
print("P-value:", p_value)

alpha = 0.05  # Significance level
if p_value > alpha:
    print("The data follows a normal distribution (fail to reject H0)")
else:
    print("The data does not follow a normal distribution (reject H0)")


# Observation: Since the p-values are not less than .05, we fail to reject the null hypothesis.
# We do not have sufficient evidence to say that the sample data does not come from a normal distribution.

# # Visualizing the Relation between each independent Feature with respect to the Dependent Feature

# In[76]:


import seaborn as sns
import matplotlib.pyplot as plt

for column in df3.columns:
    if column != 'energy_production':  
        plt.figure(figsize=(8, 6))
        sns.regplot(x=column, y='energy_production', data=df3, scatter_kws={'alpha':0.5})
        plt.title(f'Relationship between {column} and Energy Production')
        plt.xlabel(column)
        plt.ylabel('Energy Production')
        plt.show()


# temperature feature has a good linear relation with energy production as compare to other features

# In[77]:


df3.corr()


# In[78]:


plt.figure(figsize=(12,8))
sns.heatmap(
    df3.corr(),
    annot=True)


# # Plotting Correlation on a Pair Plot

# In[79]:


sns.set_style(style='darkgrid')
sns.pairplot(df3)


# # Applying some Data Transformation to increase the linear realtionship and improve our model prediction as well it scores
# 

# # Applying Standard Scaler

# In[80]:


from sklearn.preprocessing import StandardScaler
df_standard_scaled = df3.copy()
continuous_feature = ['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity','energy_production']
features = df_standard_scaled[continuous_feature]
scaler = StandardScaler()


# In[81]:


df_standard_scaled[continuous_feature] = scaler.fit_transform(features.values)
df_standard_scaled.head()


# Now if we check the mean and standard deviation of our scaled data it should have a Mean '0' and Standard deviation '1'

# In[82]:


print('Mean' '\n',np.round(df_standard_scaled.mean(),1),'\n' 'Standard Devaition','\n',np.round(df_standard_scaled.std()),1)


# # VIF

# In[83]:


import statsmodels.formula.api as smf

# Calculate R-squared for temperature
rsq_temperature = smf.ols('temperature ~ exhaust_vacuum + amb_pressure + r_humidity', data=df1).fit().rsquared
vif_temperature = 1 / (1 - rsq_temperature)

# Calculate R-squared for exhaust_vacuum
rsq_exhaust_vacuum = smf.ols('exhaust_vacuum ~ temperature + amb_pressure + r_humidity', data=df1).fit().rsquared
vif_exhaust_vacuum = 1 / (1 - rsq_exhaust_vacuum)

# Calculate R-squared for amb_pressure
rsq_amb_pressure = smf.ols('amb_pressure ~ temperature + exhaust_vacuum + r_humidity', data=df1).fit().rsquared
vif_amb_pressure = 1 / (1 - rsq_amb_pressure)

# Calculate R-squared for r_humidity
rsq_r_humidity = smf.ols('r_humidity ~ temperature + amb_pressure + exhaust_vacuum', data=df1).fit().rsquared
vif_r_humidity = 1 / (1 - rsq_r_humidity)

# Store VIF values in a DataFrame
d1 = {'variables': ['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity'], 'VIF': [vif_temperature, vif_exhaust_vacuum, vif_amb_pressure, vif_r_humidity]}
VIF_frame = pd.DataFrame(d1)

VIF_frame


# In[84]:


import matplotlib.pyplot as plt

# Define variables and their corresponding VIF values
variables = ['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity']
vif_values = [5.968554, 3.935221, 1.451793, 1.709384]

# Define unique attractive colors
colors = ['dodgerblue', 'limegreen', 'lightcoral', 'orange']

# Create a bar plot with colors
plt.figure(figsize=(10, 6))
bars = plt.bar(variables, vif_values, color=colors)

# Add data labels to each bar
for bar, value in zip(bars, vif_values):
    plt.text(bar.get_x() + bar.get_width() / 2, 
             bar.get_height() - 0.1, 
             round(value, 2),
             ha='center', va='top', color='black')

plt.xlabel('Variables')
plt.ylabel('Variance Inflation Factor (VIF)')
plt.title('VIF Values for Each Variable')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# # Evaluation Metrics:

# # Mean Square Error

# In[85]:


X = df3[['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity']]
y = df3['energy_production']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[86]:


# Initialize regression models
linear_reg = LinearRegression()
lasso_reg = Lasso(alpha=0.1)
ridge_reg = Ridge(alpha=0.1)
elastic_net_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
random_forest_reg = RandomForestRegressor(n_estimators=100)
decision_tree_reg = DecisionTreeRegressor()
xgboost_reg = XGBRegressor()
gradient_boosting_reg = GradientBoostingRegressor()


# In[87]:


# Fit the models
linear_reg.fit(X_train, y_train)


# In[88]:


lasso_reg.fit(X_train, y_train)


# In[89]:


ridge_reg.fit(X_train, y_train)


# In[90]:


elastic_net_reg.fit(X_train, y_train)


# In[91]:


random_forest_reg.fit(X_train, y_train)


# In[92]:


decision_tree_reg.fit(X_train, y_train)


# In[93]:


xgboost_reg.fit(X_train, y_train)


# In[94]:


gradient_boosting_reg.fit(X_train, y_train)


# In[ ]:





# In[95]:


from sklearn.metrics import mean_squared_error

def calculate_mse(model, X_test, y_test):
    y_pred = model.predict(X_test)  # Predicted values
    mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
    return mse
mse_values = {}

models = {
    "Linear Regression": linear_reg,
    "Lasso Regression": lasso_reg,
    "Ridge Regression": ridge_reg,
    "Elastic Net Regression": elastic_net_reg,
    "Random Forest Regression": random_forest_reg,
    "Decision Tree Regression": decision_tree_reg,
    "XGBoost Regression": xgboost_reg,
    "Gradient Boosting Regression": gradient_boosting_reg
}


for model_name, model in models.items():
    mse_values[model_name] = calculate_mse(model, X_test, y_test)

for model_name, mse in mse_values.items():
    print(f"{model_name} MSE:", mse)


# In[96]:


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    return mse, r_squared


# In[97]:


models = {
    "Linear Regression": linear_reg,
    "Lasso Regression": lasso_reg,
    "Ridge Regression": ridge_reg,
    "Elastic Net Regression": elastic_net_reg,
    "Random Forest Regression": random_forest_reg,
    "Decision Tree Regression": decision_tree_reg,
    "XGBoost Regression": xgboost_reg,
    "Gradient Boosting Regression": gradient_boosting_reg
}


# In[98]:


for name, model in models.items():
    mse, r_squared = evaluate_model(model, X_test, y_test)
    print(f"{name}:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r_squared}")
    print("\n")


# In[99]:


# Mean Squared Error for each model
mse_scores = []
for name, model in models.items():
    mse, _ = evaluate_model(model, X_test, y_test)
    mse_scores.append(mse)

# Create a DataFrame to store model names and MSE scores
model_table = pd.DataFrame({
    "Model": list(models.keys()),
    "Mean Squared Error": mse_scores
})

model_table


# In[100]:


import matplotlib.pyplot as plt

# Model names
models = ['Linear Regression', 'Lasso Regression', 'Ridge Regression', 
          'Elastic Net Regression', 'Random Forest Regression', 
          'Decision Tree Regression', 'XGBoost Regression', 
          'Gradient Boosting Regression']

# Mean Squared Error values
mse_values = [20.830005827239283, 20.83458659006245, 20.83000541416773, 
              20.83183248328101, 11.22390304236097, 18.62283310598111, 
              10.101216130841028, 15.75042262754841]

# Unique attractive colors for each bar
colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 
          'royalblue', 'plum', 'gold', 'lightpink']


# In[101]:


plt.figure(figsize=(12, 6))

# Plot Mean Squared Error
plt.bar(models, mse_values, color=colors)
plt.title('Mean Squared Error Comparison')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.xticks(rotation=45)
for i in range(len(models)):
    plt.text(i, mse_values[i], round(mse_values[i], 2), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[102]:


import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Initialize models
models = {
    "Linear Regression": linear_reg,
    "Lasso Regression": lasso_reg,
    "Ridge Regression": ridge_reg,
    "Elastic Net Regression": elastic_net_reg,
    "Random Forest Regression": random_forest_reg,
    "Decision Tree Regression": decision_tree_reg,
    "XGBoost Regression": xgboost_reg,
    "Gradient Boosting Regression": gradient_boosting_reg
}

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_test, y_pred)
    adj_r_squared = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    return mse, rmse, r_squared, adj_r_squared

# Initialize lists to store results
mse_list = []
rmse_list = []
rsquared_list = []
adj_rsquared_list = []

# Evaluate each model and store the results
for name, model in models.items():
    mse, rmse, r_squared, adj_r_squared = evaluate_model(model, X_test, y_test)
    mse_list.append(mse)
    rmse_list.append(rmse)
    rsquared_list.append(r_squared)
    adj_rsquared_list.append(adj_r_squared)
    
model_results = pd.DataFrame({
    "Model": list(models.keys()),
    "RMSE": rmse_list,
    "MSE": mse_list,
    "R-squared": rsquared_list,
    "Adjusted R-squared": adj_rsquared_list
})

# Display DataFrame
model_results


# In[103]:


import matplotlib.pyplot as plt

# Plotting the evaluation metrics
plt.figure(figsize=(12, 8))

# RMSE
plt.subplot(2, 2, 1)
plt.bar(model_results["Model"], model_results["RMSE"], color='blue')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error (RMSE)')
for i, v in enumerate(model_results["RMSE"]):
    plt.text(i, v + 0.1, str(round(v, 2)), ha='center', va='bottom')
    plt.tight_layout()
plt.show()


# In[104]:


# MSE

# Plotting the evaluation metrics
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 2)
plt.bar(model_results["Model"], model_results["MSE"], color='green')
plt.xlabel('Model')
plt.ylabel('MSE')
plt.title('Mean Squared Error (MSE)')
for i, v in enumerate(model_results["MSE"]):
    plt.text(i, v + 0.1, str(round(v, 2)), ha='center', va='bottom')
    plt.tight_layout()
plt.show()


# In[105]:


# R-squared

# Plotting the evaluation metrics
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 3)
plt.bar(model_results["Model"], model_results["R-squared"], color='orange')
plt.xlabel('Model')
plt.ylabel('R-squared')
plt.title('R-squared')
for i, v in enumerate(model_results["R-squared"]):
    plt.text(i, v + 0.01, str(round(v, 2)), ha='center', va='bottom')
    plt.tight_layout()
plt.show()


# In[106]:


# Adjusted R-squared

# Plotting the evaluation metrics
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 4)
plt.bar(model_results["Model"], model_results["Adjusted R-squared"], color='red')
plt.xlabel('Model')
plt.ylabel('Adjusted R-squared')
plt.title('Adjusted R-squared')
for i, v in enumerate(model_results["Adjusted R-squared"]):
    plt.text(i, v + 0.01, str(round(v, 2)), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[107]:


# Linear Regression
y_pred_linear = linear_reg.predict(X_test)

# Lasso Regression
y_pred_lasso = lasso_reg.predict(X_test)

# Ridge Regression
y_pred_ridge = ridge_reg.predict(X_test)

# Elastic Net Regression
y_pred_elastic_net = elastic_net_reg.predict(X_test)

# Random Forest Regression
y_pred_rf = random_forest_reg.predict(X_test)

# Decision Tree Regression
y_pred_dt = decision_tree_reg.predict(X_test)

# XGBoost Regression
y_pred_xgb = xgboost_reg.predict(X_test)

# Gradient Boosting Regression
y_pred_gb = gradient_boosting_reg.predict(X_test)


# In[108]:


# Creating DataFrames for each model to display actual and predicted values
results_linear = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_linear})
results_lasso = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_lasso})
results_ridge = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_ridge})
results_elastic_net = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_elastic_net})
results_rf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf})
results_dt = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_dt})
results_xgb = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_xgb})
results_gb = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_gb})


# In[109]:


print("Linear Regression:")
results_linear


# In[110]:


print("\nLasso Regression:")
results_lasso


# In[111]:


print("\nRidge Regression:")
results_ridge


# In[112]:


print("\nElastic Net Regression:")
results_elastic_net


# In[113]:


print("\nRandom Forest Regression:")
results_rf


# In[114]:


print("\nDecision Tree Regression:")
results_dt


# In[115]:


print("\nXGBoost Regression:")
results_xgb


# In[116]:


print("\nGradient Boosting Regression:")
results_gb


# In[117]:


results = {
    'Linear Regression': (y_test, y_pred_linear),
    'Lasso Regression': (y_test, y_pred_lasso),
    'Ridge Regression': (y_test, y_pred_ridge),
    'Elastic Net Regression': (y_test, y_pred_elastic_net),
    'Random Forest Regression': (y_test, y_pred_rf),
    'Decision Tree Regression': (y_test, y_pred_dt),
    'XGBoost Regression': (y_test, y_pred_xgb),
    'Gradient Boosting Regression': (y_test, y_pred_gb)
}


# In[118]:


mse_results = {model: mean_squared_error(actual, predicted) for model, (actual, predicted) in results.items()}


# In[119]:


# Find the model with the lowest mean squared error
best_model = min(mse_results, key=mse_results.get)
best_mse = mse_results[best_model]


# In[120]:


best_model, best_mse


# In[ ]:





# In[129]:


import xgboost as xgb


# In[128]:


import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


# In[122]:


data = df3


# In[124]:


df3.columns


# In[125]:


X = df3[['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity']]
y = df3['energy_production']


# In[126]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[130]:


xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)


# In[131]:


# Define a function to predict energy output using XGBoost
def predict_energy_output_xgb(temperature, exhaust_vacuum, ambient_pressure, relative_humidity):
    input_data = [[temperature, exhaust_vacuum, ambient_pressure, relative_humidity]]
    energy_output = xgb_model.predict(input_data)
    return energy_output[0]


# In[132]:


# Streamlit app
st.title('Energy Output Prediction')


# In[133]:


# Add input widgets
temperature = st.slider('Temperature (°C)', -10.0, 40.0, 20.0, 0.1)
exhaust_vacuum = st.slider('Exhaust Vacuum (cm Hg)', 25.0, 82.0, 50.0, 0.1)
ambient_pressure = st.slider('Ambient Pressure (millibar)', 990.0, 1040.0, 1010.0, 0.1)
relative_humidity = st.slider('Relative Humidity (%)', 20.0, 100.0, 50.0, 0.1)


# In[134]:


# Add a button to trigger prediction
if st.button('Predict'):
    prediction_xgb = predict_energy_output_xgb(temperature, exhaust_vacuum, ambient_pressure, relative_humidity)
    st.write('Predicted Energy Output (MW) using XGBoost:', prediction_xgb)


# In[ ]:




