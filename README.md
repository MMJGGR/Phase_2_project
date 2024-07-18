* [Slideshow Presentation](https://github.com/MMJGGR/Phase_2_project/blob/main/phase2-presentation.pdf)
* [Jupyter Notebook with Cleaning & Questions](https://github.com/MMJGGR/Phase_2_project/blob/main/index.ipynb)
* [Notebook PDF](https://github.com/MMJGGR/Phase_2_project/blob/main/index.pdf)
* [Data Sources](https://github.com/MMJGGR/Phase_2_project/tree/main/Data)
* [Individual Group Member Contributions](https://github.com/MMJGGR/Phase_2_project/tree/main/work_in_progress)

# 1. Introduction

In the King County housing market, multiple dynamics play a major role in shaping the prosperity of Real Estate companies. Real estate development companies face significant challenges in navigating many complexities when it comes to deciding on the most important features to include in homes. 

The project aims to help a Real Estate Development Company looking to enter the King County housing market. This will be done by providing a comprehensive analysis that focuses on the key factors that influence housing prices. By leveraging data-driven approaches, we aim to uncover insights that will empower the stakeholder to make informed decisions.

With the insights gained from the data from the King County House Sales dataset, we hope to advise the Real Estate Development Company on the most important characteristics/features to focus on when constructing homes, in order to generate more income and be as profitable as possible. 





# 2. Problem Statement- Business Problem
The Real Estate Development Company is venturing into the King County Housing market and would like to develop a sales strategy that makes it profitable by targeting houses that have the highest prices in the market.<br>


# 3. Objectives
#### 1. Explore the Relationship Between Number of bathrooms and its price
- Investigate the relationship between the number of bathrooms (`bathrooms`) in a house and its price (`price`). Determine if houses with more bathrooms are priced higher.

#### 2. Explore the relationship between the square footage of the home and its price
- Investigate the relationship between the square footage of the home (`sqft_living`) and its price (`price`). Determine if houses with more square footage are priced higher.
#### 3. Explore the impact of the King County housing grading system on the pricing of homes in the market.
- Investigate the relationship between the grades awarded to a house (`grade`) and its price. Determine if houses with higher grades are priced higher. 
#### 4. Explore the impact of a home's neighborhood on its price
- Investigate the relationship between square footage of interior housing living space for the nearest 15 neighbors of a house and its price. Determine if the size of homes within a neighborhood affects the price of a house.

#### 5. Develop a Linear Regression Model to Predict Housing Prices
- Build and evaluate a linear regression model using features `bathrooms`, `sqft_living`, and `grade`to predict house prices (`price`). Provide the stakeholder with a predictive tool for estimating housing prices and supporting strategic decision-making in housing development.
# 4. Data Understanding:
This project uses the King County House Sales dataset whose size, descripritive statistics for all the features used in the analysis, and justification of the inclusion of features based on their properties and relevance for the project have been given in the Exploratory Data Analysis part.<br>

The data contains all the relevant features that are needed to meet our objectives as described above.<br>

However, to explain the relationship fully, additional features are needed.

### Limitations of the data that have implications for the project
- The data is not normally distributed, which may impact the reliability of the model especially in the extreme of the price distribution. The data was scaled to deal with non-normality. <br>
- There are some missing values for some columns, which will have to be dealt with.


## Imports and Data Loading
#imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
data = pd.read_csv('./Data/kc_house_data.csv')
data.head(4)


## Exploratory data analysis
data.shape
There are 21 columns and 25,597 rows
data.info()
data.columns
These are the columns and their descriptions: 
* **id** - unique identified for a house
* **dateDate** - house was sold
* **pricePrice** -  is prediction target
* **bedroomsNumber** -  of Bedrooms/House
* **bathroomsNumber** -  of bathrooms/bedrooms
* **sqft_livingsquare** -  footage of the home
* **sqft_lotsquare** -  footage of the lot
* **floorsTotal** -  floors (levels) in house
* **waterfront** - House which has a view to a waterfront
* **view** - Has been viewed
* **condition** - How good the condition is ( Overall )
* **grade** - overall grade given to the housing unit, based on King County grading system
* **sqft_above** - square footage of house apart from basement
* **sqft_basement** - square footage of the basement
* **yr_built** - Built Year
* **yr_renovated** - Year when house was renovated
* **zipcode** - zip
* **lat** - Latitude coordinate
* **long** - Longitude coordinate
* **sqft_living15** - The square footage of interior housing living space for the nearest 15 neighbors
* **sqft_lot15** - The square footage of the land lots of the nearest 15 neighbors
A data description of the numerical columns
data.describe()
# 5. Data Cleaning
### Checking for duplicates
data.duplicated().sum()
There are no duplicates
### Looking for null values
data.isna().sum()
The waterfront, view, and year renovated columns have null values<br>
    
- waterfront, view, yr_renovated: 2376, 63, 3842 missing values respectively<br>
- is,date, price, bedrooms, sqft_living, sqft_lot, floors, condition, grade, sqft_above, sqft_basement, yr_built, zipcode, lat, long, sqft_living15, sq_lot15: No missing values.<br>
# We will fill null values in the waterfront and view columns with the mode since these are categorical,
# We will also convert them to the string datatype, which works for categorical data

# and for the year renovated, fill with the median since there are possibly outliers
data['view'] = data['view'].astype(str)
data['waterfront'] = data['waterfront'].astype(str)

wf_mode = data.waterfront.mode()
view_mode = data.view.mode()
median_year = data.yr_renovated.median()

data.waterfront.fillna(wf_mode, inplace = True)
data.view.fillna(view_mode, inplace = True)
data.yr_renovated.fillna(median_year, inplace = True)
data.isna().sum()
There are now no null values
# 6. Data Preparation
#### Correlation Heatmap
A correlation heatmap to obtain numbers for an easier reading of the correlation of the relationship between columns
plt.subplots(figsize=(20,15))
sns.heatmap(data.corr(),cmap="coolwarm",annot=True);
We have already determined that our predicted column is price, and will, therefore, be looking at the correlation of the rest of the columns with price<br>
Choose the features that have the highest collinearity with the price. These are:<br>
- Bathrooms
- sqft_living
- grade
- sqft_above
- sqft_living15

Based on the correlation coefficients with price from the KC Housing Data dataset, the most important features are:

    Bathrooms- Number of bathrooms
        Justification: Bathrooms RM has a strong positive correlation with price (0.53). This indicates that as the number of bathrooms in a home in increases, its price tends to increase as well.
        

    sqft_living - Footage of the home
        Justification: sqft_living has a very strong positive correlation with price (0.7). This indicates that as the square footage of the home increases, its price increases.

    grade - overall grade given to the housing unit, based on King County grading system
        Justification: Grade has a strong positive correlation with price (0.67). This indicates that as the grade of the home increases, its price increases.

    sqft_above - square footage of house apart from basement
        Justification: sqft_above has a strong correlation with price (0.61). This indicates that as the square footage of house apart from basement increase, its price increases.

    sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors
        Justification: sqft_living 15 has a strong correlation with price (0.59). This indicates that as the square footage of interior housing living space for the nearest 15 neighbors increases, the home's price increases.

### Checking the correlations for the selected important features
Create a pairplot with a regression line to show the best fit line
data2 = data[['bathrooms','sqft_living','grade','sqft_above','sqft_living15', 'price']]
data2.head()

sns.pairplot(data2, kind = "reg");

Check for outliers
# Plotting box plots for each column
plt.figure(figsize=(10, 6))
sns.boxplot(data = data2)
plt.title('Box plots of Selected Columns')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show();
# Plotting box plots for each column
plt.figure(figsize=(10, 6))
sns.boxplot(data = data2[['bathrooms', 'grade', 'sqft_above', 'sqft_living', 'sqft_living15']])
plt.title('Box plots of Selected Columns')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show();
In this case, outliers are part of the distribution and removing them presents a false representation of the problem<br>
Based on the box plots, the price is what seems to have a lot of outliers, which is good since it gives a proper representation of the market.
# 7. Data Analysis
### Analysis 1: What is the relationship between bathrooms and price?
sns.regplot(x = data2['bathrooms'], y = data2['price'],
            scatter_kws = {"color": "black", "alpha": 0.5},
            line_kws = {"color": "red"});
**Trend Observation:**

 In the scatter plot with a regression line of bathrooms (number of bathrooms in house) against price, you can observe a positive trend. As the average number of bathrooms increases, the price of the house tends to increase as well.
 
**Implication:**

 This suggests that houses with more bathrooms generally have higher prices in the King County housing market. They are priced higher.
### Analysis 2: What is the Relationship between sqft_living and price?
sns.regplot(x = data2['sqft_living'], y = data2['price'],
            scatter_kws = {"color": "green", "alpha": 0.5},
            line_kws = {"color": "red"});
**Trend Observation:**

 In the scatter plot with a regression line of square footage of the home (sqft_living) against price, you can observe a positive trend. As the average square footage of the home increases, the price of the house tends to increase as well.
 
**Implication:**

 This suggests that houses with a higher square footage have higher prices in the King County housing market. They are priced higher.
### Analysis 3: What is the relationship between Grade and Price
sns.regplot(x = data2['grade'], y = data2['price'],
            scatter_kws = {"color": "yellow", "alpha": 0.5},
            line_kws = {"color": "red"});
**Trend Observation:**

 In the scatter plot with a regression line of grade (overall grade given to the housing unit, based on King County grading system) against price, you can observe a positive trend. As the grade given to the home increases, the price of the house tends to increase as well.
 
**Implication:**

 This suggests that houses with a higher grade have higher prices in the King County housing market. They are priced higher.
### Analysis 4: What is the relationship between sqft_living15 and Price?
sns.regplot(x = data2['sqft_living15'], y = data2['price'],
            scatter_kws = {"color": "orange", "alpha": 0.5},
            line_kws = {"color": "red"});
**Trend Observation:**

 In the scatter plot with a regression line of sqft_living15 (The square footage of interior housing living space for the nearest 15 neighbors) against price, you can observe a positive trend. As the average square footage of the homes within the neighborhood increases, the price of the house tends to increase as well.
 
**Implication:**

 This suggests that houses within a neighborhood that has homes with a higher square footage have higher prices in the King County housing market. They are priced higher.
 
### Analysis 5: Modeling 
#### Feature Selection
First create the first iteration of our linear model with all the selected features
# Create a model to test using all the selected features
from statsmodels.formula.api import ols


formula = 'price ~ bathrooms + sqft_living + grade + sqft_above + sqft_living15'
model = ols(formula, data2).fit()
model_summary = model.summary()

model_summary
# Check for multicollinearity
predictors_data2 = data2[['bathrooms', 'sqft_living', 'grade', 'sqft_above', 'sqft_living15']]
predictors_data2.corr()

abs(predictors_data2.corr()) > 0.75
Sqft_living seems to have very strong correlation with all the other predictors, so we will drop it and see the impact on our model
# Create a second model after dropping sqft_living from the 
from statsmodels.formula.api import ols


formula2_without_sqft_living = 'price ~ bathrooms + grade + sqft_above + sqft_living15'
model2_without_sqft_living = ols(formula2_without_sqft_living, data2 ).fit()
model2_summary_without_sqft_living = model2_without_sqft_living.summary()

model2_summary_without_sqft_living
This actually lowers the R Squared value (0.544 versus 0.482), so dropping the sqft_living column does not improve the model
Let's drop the sqft_living15 column from the original model since it has a strong collinearity with multiple columns and see the impact that it has on the model
# Create a second model after dropping sqft_living15 from the 
from statsmodels.formula.api import ols


formula2_without_sqft_living15 = 'price ~ bathrooms + sqft_living + grade + sqft_above'
model2_without_sqft_living15 = ols(formula2_without_sqft_living15, data2).fit()
model2_summary_without_sqft_living15 = model2_without_sqft_living15.summary()

model2_summary_without_sqft_living15
The R squared value is the same as the first model, meaning that sqft_living15 did not actually contribute to the model. We can leave it as is.
Next, let's try removing sqft_above from the model since it also has relationships with multiple columns
# Create a second model after dropping sqft_living15 from the 
from statsmodels.formula.api import ols


formula2_without_sqft_above = 'price ~ bathrooms + sqft_living + grade'
model2_without_sqft_above = ols(formula2_without_sqft_above, data2).fit()
model2_summary_without_sqft_above = model2_without_sqft_above.summary()

model2_summary_without_sqft_above
This also does not have a major impact on our relationship (0.544 versus 0.537 R squared values), so we can omit. 
### Feature Ranking With Recursive Elimination to validate the selected features for improving our model
#from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
predictors_for_feature_ranking = data2[['bathrooms', 'sqft_living', 'grade', 'sqft_above', 'sqft_living15']]
selector = RFE(linreg, n_features_to_select=3)
selector = selector.fit(predictors_for_feature_ranking, data2['price'])

# Calling the .support_ attribute tells you which variables are selected

selector.support_ 

# Calling .ranking_ shows the ranking of the features, selected features are assigned rank 1

selector.ranking_
This validates our selection of bathrooms, sft_living, and grade as the most important features since they were ranked as best with the ranking of 1. sqft_above and sqft_living have rankings of 2 and 3 respectively. So, this validates the fact that we discarded them.
### Final Model- First Iteration

The selected features are: 
Bathrooms, sqft_living, and grade
data2 = data2.drop(columns = ['sqft_above', 'sqft_living15'], axis = 1)
data2.head()
Let's see whether the selected columns are categorical and deal with them
print("Number of unique values in bathrooms column:", data2['bathrooms'].nunique())
print("Number of unique values in sqft_living column:", data2['sqft_living'].nunique())
print("Number of unique values in grade column:", data2['grade'].nunique())

Let's change the column data dtypes to string, to prevent them from being deemed as non-categorical, simply because they have numerical data types
data['bathrooms'] = data['bathrooms'].astype(str)
data['grade'] = data['grade'].astype(str)
data['sqft_living15'] = data['sqft_living15'].astype(str)
print(data2['bathrooms']is pd.CategoricalDtype)
print(data2['sqft_living']is pd.CategoricalDtype)
print(data2['grade']is pd.CategoricalDtype)
# Get the list of all categorical columns
data2.select_dtypes(include=['object']).columns.tolist()
There are no categorical columns
data2.select_dtypes(exclude=['object']).columns.tolist()
All the columns are numerical
data2._get_numeric_data().head()
All the columns are numeric
Using all three methods above, we can see that none of the predictors are categorical, and we, therefore, do not have to deal with them 
Let's convert them back to their original data types
data['bathrooms'] = data['bathrooms'].astype(float)
data['grade'] = data['grade'].astype(int)
data['sqft_living15'] = data['sqft_living15'].astype('int')
x = data2[['bathrooms', 'sqft_living', 'grade']]
x.head()
y = data2['price']
y.head()
### Test Train Split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2,random_state=42)

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# Modeling
model = LinearRegression()
# Training the model
model.fit(x_train_scaled,y_train)
y_pred = model.predict(x_test_scaled)
data_after_test_training = pd.DataFrame({"true":y_test,"pred":y_pred})
data_after_test_training.head()
### Validation
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
r2  = r2_score(y_test,y_pred)

print("mse",mse)
print("mae",mae)
print("R2" ,r2)


y_hat_train = model.predict(x_train_scaled)
y_hat_test = model.predict(x_test_scaled)
train_residuals = y_hat_train - y_train
test_residuals = y_hat_test - y_test
train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
print('Train Mean Squared Error:', train_mse)
print('Test Mean Squared Error:', test_mse)
print('Difference between Train and Test MSE is:', (train_mse/test_mse-1).round(3)*100,'%)')

The R squared value from the first iteration of our model shows that our model is accurate 53.11% of the time
A mean absolute error of 161628 shows the predicted value is 161628 points away from the actual value. Our training and test MSE exhibit a very small difference (2.9%). This is a sign that the model generalizes well to future cases.

These are not ideal conditions for the model we want to eventually have, let's make some iterations.
### Final Model- Second Iteration
Let us change the ratio of our test-train data (from 80-20 to 70-30) and see whether this improves our model
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
#IMPORTANT NOTES!!!!!!!!!
# train data with fit transform 
# test data with test data
# We only scale our predictors, we don't scale the predicted value
x_test_scaled = scaler.transform(x_test)
# Modeling
model = LinearRegression()
# Training the model
model.fit(x_train_scaled,y_train)
y_pred = model.predict(x_test_scaled)
data_after_test_training = pd.DataFrame({"true":y_test,"pred":y_pred})
data_after_test_training.head()
### Validation
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
r2  = r2_score(y_test,y_pred)

print("mse",mse)
print("mae",mae)
print("R2" ,r2)

y_hat_train = model.predict(x_train_scaled)
y_hat_test = model.predict(x_test_scaled)

train_residuals = y_hat_train - y_train
test_residuals = y_hat_test - y_test

train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
print('Train Mean Squared Error:', train_mse)
print('Test Mean Squared Error:', test_mse)
print('Difference between Train and Test MSE is:', (train_mse/test_mse-1).round(3)*100,'%)')
The R squared value from the second iteration of our model shows that our model is accurate 53.23% of the time based on the test data.<br>
A mean absolute error of 161609 shows the predicted value is 161609 points away from the actual value, based on the test data.<br> However, the difference between the train and test mean squared error is now very little (1%). This indicates that we are not overfitting. This is a large improvement on our model and we will take it.
Now let's get our model's summary to get values like the p-values of the predictors, f-statistic and its probability, R squared, and coefficients..
### Model Summary
import statsmodels.api as sm

# Add a constant to the model (for the intercept)
x_train_sm = sm.add_constant(x_train)

# Fit the model using statsmodels
model_sm = sm.OLS(y_train, x_train_sm).fit()

# Print the model summary
print(model_sm.summary())
***Model Summary Interpretation***<br> R-squared: 0.539 This means that approximately 53.9% of the variability in the house prices can be explained by the model.<br>

Adjusted R-squared:  We will create another model less one identified predictor to see how this impacts the model and whether it performs better. If the model performs better, we need to drop this variable or decide that it is not one of the most important ones.<br>

F-statistic and its p-value: F-statistic: 5895, Prob (F-statistic): 0.00 The very low p-value suggests that the overall model is statistically significant, meaning that at least one of the predictors is significantly related to the dependent variable (house prices).<br>

Coefficients:<br>

Each row represents a predictor in the model with its coefficient, standard error, t-value, and p-value. For instance, the coefficient for sqft_living is 197.0231, meaning that for each additional square foot of living space, the house price increases by approximately $197.0231, holding other variables constant. The p-values for all predictors are less than 0.05, indicating that they are statistically significant.<br>
Let's use feature ranking with recursive elimination to select only two variables for our model.
#from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
predictors_for_feature_ranking2 = data2[['bathrooms', 'sqft_living', 'grade']]
selector2 = RFE(linreg, n_features_to_select=2)
selector2 = selector.fit(predictors_for_feature_ranking2, data2['price'])

# Calling the .support_ attribute tells you which variables are selected

selector.support_ 

# Calling .ranking_ shows the ranking of the features, selected features are assigned rank 1

selector.ranking_
All the predictors still have the same ranking of 1
import pandas as pd
import statsmodels.api as sm

def stepwise_selection(x, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out=0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        x - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(x.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
    return included

result = stepwise_selection(x, y, verbose=True);
print('resulting features:')
print(result)
All the selected predictors have been retained
# 8. Regression Results:
The features with strong relationships to sale prices are: <br>
-- Number of bathrooms (`bathrooms`)<br>
-- Sqft_living (`sqft_living`)<br>
-- Grade (`grade`)<br>

Their coeficients are:<br>
-- `bathrooms`: -33,304. For each additional bathroom, the house price decreases by approximately $33,304, holding other variables constant.<br>
-- `sqft_living`: $197.0231. For each additional square foot of living space, the house price increases by approximately $197.0231, holding other variables constant.<br>
-- `grade`: $107,900. For each increase in grade, the house price increases by approximately $107,900, holding other variables constant.<br>



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))

# Create the scatter plot
scatter = sns.scatterplot(x=y_train, y=y_hat_train, alpha=0.5, hue=y_hat_train, palette='viridis')

# Add the line for y=x
plt.plot([data2['price'].min(), data2['price'].max()],
         [data2['price'].min(), data2['price'].max()], 'k--', lw=2)

# Add labels and title
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price')

# Add legend
plt.legend(title='Predicted Price', loc='upper left')

# Show the plot
plt.show()


# 9. Conclusion
## Significant Predictors of House Prices:
### Objective 1: Explore the Relationship Between Number of bathrooms and its price

**Number of Bathrooms (bathrooms)**: Contrary to initial expectations, the coefficient suggests that each additional bathroom is associated with a decrease in house prices by $-33,300 , holding other variables constant. This unexpected finding may warrant further investigation into potential factors influencing this relationship. 
### Objective 2: Explore the relationship between the square footage of the home and its price

**Square Footage of Living Space (sqft_living)**: The coefficient suggests that for each additional square foot of living space, house prices are estimated to increase by $197.0231, holding other variables constant. This indicates that larger houses tend to command higher prices.
### Objective 3: Explore the impact of the King County housing grading system on the pricing of homes in the market.

**Grade of the House (grade)**: A higher grade is associated with higher house prices, with each unit increase in grade estimated to increase house prices by $107,900, holding other variables constant. Grades typically reflect the quality and features of the house, influencing buyer preferences and willingness to pay.

### Objective 4: Explore the impact of a home's neighborhood on its price
**sq_ft_living15(The square footage of interior housing living space for the nearest 15 neighbors)**: A higher sqft_living15 is associated with higher house prices. With a strong linear relationship of 0.59 with price, this feature can be important for the Real Estate Development Company when picking the best neighborhoods for their houses. It is, however, worth noting that it was not selected for the model since through feature ranking with recursive elimination, it received a ranking of 3.  

### Objective 5: Develop a Linear Regression Model to Predict Housing Prices
**Model Fit and Statistical Significance**:

The overall model has an R-squared value of 0.539, indicating that approximately 53.9% of the variance in house prices can be explained by the predictors (sqft_living, grade, bathrooms) included in the model. This suggests that while these predictors are significant, there are other factors not captured by the model that also influence house prices.

The F-statistic is very high (8359) with a low p-value (0.00), indicating that the model is statistically significant. This implies that at least one of the predictors is significantly related to house prices, reinforcing the reliability of the model's predictions.
# 10. Recommendations
Business Strategy Implications
Given these insights, the Real Estate Development Company should consider the following strategies:

### Target Larger Houses: 
* Since the model indicates that house price increases with the square footage of living space, the company should focus on acquiring or developing larger houses to maximize potential sales prices.

### Focus on High-Grade Houses: 
* Houses with higher grades are significantly associated with higher prices. The company should aim to develop or invest in properties with higher quality finishes, better construction, and more desirable features.

### Investigate Bathroom Anomaly: 
* The negative coefficient for bathrooms is unusual and suggests a deeper investigation is needed. The company should look into whether this result is due to multicollinearity or other factors not captured in the model. It may be beneficial to re-examine the quality or location of properties with many bathrooms.

### Consider Additional Predictors: 
* Since the R-squared value is 0.537, there are other factors influencing house prices that are not included in this model. The company should consider incorporating additional relevant variables such as location, lot size, view, and age of the house to improve the model's predictive power.

### Use Predictive Analytics for Targeted Marketing: 
* By using the model to predict high-priced houses, the company can create targeted marketing strategies to attract high-end buyers. This can involve highlighting the features that are most strongly associated with higher prices, such as larger living spaces and higher grades.

### Price Optimization: 
* The company can use the model's predictions to set competitive yet profitable prices for their properties, ensuring that they are priced appropriately based on their characteristics.