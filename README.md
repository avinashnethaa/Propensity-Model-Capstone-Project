# Propensity Model Capstone Project

## Problem Statement 
This project aims to build a Propensity Model to identify potential customers. The goal is to predict whether a customer is likely to respond positively to a campaign, helping optimize marketing efforts.

## Dataset Overview
The dataset contains information about customers, including their age, profession, marital status, education, and other features. The goal is to predict the 'responded' variable, indicating whether a customer responded positively to a campaign.

### Features
- `custAge`: Customer's age
- `profession`: Customer's profession
- `marital`: Marital status
- `schooling`: Customer's education level
- `default`: Default status
- `housing`: Housing status
- `loan`: Loan status
- `contact`: Contact method
- `month`: Month of contact
- `day_of_week`: Day of the week of contact
- `campaign`: Number of contacts during the campaign
- `pdays`: Number of days since the customer was last contacted
- `previous`: Number of contacts performed before this campaign
- `poutcome`: Outcome of the previous marketing campaign
- `emp.var.rate`: Employment variation rate
- `cons.price.idx`: Consumer price index
- `cons.conf.idx`: Consumer confidence index
- `euribor3m`: Euribor 3-month rate
- `nr.employed`: Number of employees
- `pmonths`: Pseudo-monthly indicator
- `pastEmail`: Number of emails sent in the past
- `responded`: Target variable indicating campaign response (yes/no)
  
##### The dataset has 8238 entries and 22 features. Some features have missing values that need to be addressed during preprocessing. The target variable, 'responded,' will be predicted using various machine learning models.

## Objective
This project is aimed at building a propensity model to identify potential customers for an insurance company to develop a tool to optimize their marketing efforts.

# Description
Propensity modeling is a method that aims to forecast the chance that individuals, leads, and customers will engage in specific actions. This method uses statistical analysis, which takes into account all the independent and confounding factors that impact customer behavior.

# Data
The insurance company has provided a historical dataset (train.csv). The company has also provided you with a list of potential customers to whom to market (test.csv). From this list of potential customers, we need to determine yes/no whether we wish to market to them.

#### Input(Independent) Variables:
`custAge`, `profession`, `marital`, `schooling`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `campaign`, `pdays`, `previous`, `poutcome`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`, `pmonths`, `pastEmail`

#### Output(Dependent/Target) Variables:
`responded`

# Steps involved in building the model and identifying target customers
### Started with importing necessary libraries
- Basic libraries like pandas, numpy, matplotlib, seaborn, and warnings
- Z-score for outlier treatment
- Preprocessing Libraries, Models, Evaluation Metrics.
- Hyperparameter Tuning Libraries and Cross-Validation Libraries
### Train Dataset Details
- Total number of rows in data: 8238
- Total number of columns: 24
### Deleting Unwanted rows and columns 
- Dropped id and profit columns as they are not necessary as of now.
### Exploratory Data Analysis
- Generated the pandas_profile report on the train dataset.
- Data info to know about data types and null values.
- Data describe to get an overview of Descriptive Statistics of numerical columns.
- ### Duplicate values
   - As the dataset is having 36 duplicated rows, dropped all duplicate rows.
- ### Null Values
   - To find the null values used isnull() and plotted bar chart.
   - To handle null values, filled all null values using Forward-fill as the dataset belongs to the time series, and added the chart is for reference.
- ### Outliers
   - Before finding outliers replaced '999' value in pdays and pmonths columns with -1 to identify never contacted, as '999' can be treated as an outlier and affect the analysis.
   - To view the outliers used a boxplot to visualize the outlier values.
   - Used Z-scores to find the rows with outlier values from custAge, campaign, pdays, and pastEmail columns.
   - Handled outliers by removing the outlier rows and saved all remaining rows as cleaned_data.
   - Generated Pandas_profiling again after performing outlier treatment.
- ### Data visualization
    - Before visualizing and analyzing the data separated the categorical and numerical columns.
    - Set up default plot size is (13,5).
    - Visualized the data based on the target column 'responded' in plots per below and updated observation for every plot:
    - Plot 1:#  Pairplot to understand the distribution between numerical columns.
    - Plot 2: Response based on Profession of the customer.
    - Plot 3: Response based on  Marital Status of the customer.
    - Plot 4: Histplot for Schooling with response.
    - Plot 5: Pairplot to understand the distribution between numerical columns.
    - Plot 6: Distribution of Customer Age with Housing using countplot.
    - Plot 7 : Distribution of  Customer Age with Loan using countplot.
    - Plot 7: Distribution of  Day of Week with Response using countplot.
    - Plot 8: Distribution of Previous Outcome with Response using countplot.
    - Plot 12: Distribution on all categorical variable by using pie plot.
    - Plot 13: Distribution on all numerical variable by using bar plot.
    -   
### Feature Engineering  
-  #### Encoding
  -  Encoded all categorical columns using Label Encoder.
-  #### Scaling
  -  Scaled all numerical columns using Standard Scaler
### Split Train and Test Data
  - Defined X with all columns except the target column and y with the target column
  - Did train and test split on data using train-test split library.  - 
- #### Sampling Training Dataset
    - As the dataset is highly imbalanced on the target column we need to perform any sampling technique.
    - Using RandomUnderSampler undersampled the training dataset.
    - Using SMOTE oversampled the undersampled training dataset.
### Model Selection
- As the target variable is a classification output we can perform multiple classification Models.
- Models are
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Support Vector Classifier
  - KNeighbors Classifier
  - XG Boosting Classifier
  - Neural Network Classifier
### Model Training and Testing
- Trained all models with the resampled training data.
- Made predictions with test data.
### Model Evaluation
- Evaluated all models' performance using evaluation metrics like:
  - Accuracy Score
  - ROC AUC Score
  - Classification Report.
  - Confusion Matrix
### Best Model Evaluation
- Using a for loop determined the best model with a minimum threshold of 0.80.
- Gradient boosting Classifier Model turned out to be the BEST MODEL with the highest Accuracy of 0.87.
### Hyperparameter Tuning
- Performed RandomSearch CV with Gradient boosting Classifier model as it was the best from all trained classification models.
- Trained the grid search with the original training dataset.
- With the best estimator from Grid Search CV predicted the test dataset achieved an accuracy increase to 0.87% for the best estimator of Gradient boosting Classifier.
- So, the model performance increased by hyperparameter tuning.
### Cross-Validation
- Cross-validated the model evaluation using 5 folds and achieved an accuracy of 0.90%
- The best estimator achieved an accuracy of approximately 72.03%, which is consistent with the cross-validation accuracy of 90.3%. This suggests that the chosen hyperparameters provide a stable and reliable model with minimal variance across different folds.
### Feature Importance
- Feature importances indicate the contribution of each feature to the model's predictions. A higher importance value suggests a stronger influence on the model's decision-making.
- Features with the highest importance:
   - 'nr.employed' (38.31%) and 'euribor3m (21.25%) significantly impact the model, indicating that economic factors play a crucial role in predicting the outcome.
### Predictions on Unseen Data
- As provided with test.csv file and asked to predict and add a column in that set to identify the potential customer to go with marketing followed below steps;
  - Read the test.csv file.
  - Handled null values.
  - Replaced 999 with -1 in pdays and pmonths columns.
  - Separated categorical columns and numerical columns.
  - Encoded categorical columns.
  - Scaled numerical columns.
  - And predicted Marketing Decision as 1(yes) or 0 (No) using the Best model from hyperparameter tuning.
  - Created a column Marketing Decision using predictions.

**_As the test data is encoded and scaled we cannot get an exact idea from these rows; to get an exact list of Customers that can be contacted for marketing, read the test.csv again into candidates and added the predictions into the Marketing Decision Column and displayed all potential customers' rows. Exported the potential customers list to a new excel file marketing_list._**

## Difficulties
- As the dataset is highly imbalanced finding a suitable sampling technique is a bit hard, found undersample could help better for high imbalance so used oversampling over undersampling. 
- To better understanding, performed all kinds of Distribution to understand the dataset.
- As the dataset is having very high null values, finding better imputing methods took time.

# Conclusion
- Hyperparameter tuning using RandomizedSearchCV significantly improved the performance of the Gradient Descent classifier, achieving an accuracy of approximately 86.91%.
- Cross-validation results confirmed the model's consistency and generalizability.
- Feature importance analysis highlighted the crucial role of economic factors, particularly 'nr.employed' and 'emp.var.rate.'
- In summary, the tuned Gradient Descent model demonstrates robust performance, but ongoing monitoring and potential adjustments are essential to adapt to evolving data patterns and enhance predictive capabilities. The Propensity Model project has been a rewarding journey, leveraging data for effective marketing. With advanced algorithms and insightful analysis, the developed tool identifies potential customers. The resulting customer list, combined with strategic insights, positions us for targeted and impactful marketing campaigns aligned with customer preferences and behaviors. Continuous improvement and adaptation will be key to maintaining the model's effectiveness in a dynamic business environment.

##                                                             _**Thank you!**_
