# Credit_Card_Analysis_Machine_Learning
# Overview 
## Purpose
Machine learning can be used to predict credit card risk and solve loan approval questions becasue credit risk is ultimatley an unbalanced classification problem. Previous data on loan approval can be used to train machine learning models to determine if someone if worthy of a loan, but good loans undeniably outnumber risky loans, and this class imbalance must be considered when a choosing model for prediciting credit risk. The purpose of this project was to use the Python's Sciki-learn (**sklearn**) and Imbalanced-learn (**imblearn**) libraries to build and evaluate six different classification models that use differnt resampling and ensemble learning algorithms. The goal was to determine which of the six models are best at predicting credit risk based evidence from confusion matrices and classification reports. 
## Analysis Roadmap
The [credit card dataset](LoanStats_2019Q1.csv.zip) used in this project to train machine learning models came from LendingClub, a peer-to-peer lending services company. This dataset was imported to a DataFrame, cleaned, and encoded using the **Pandas** library. 

This project was broken down into following three main parts found in the Analysis Section:
1. Using Oversampling and Undersampling Technqiues to Predict Credit Risk
   - Use the oversampling **RandomOverSampler** algorithm to resample the dataset and train a logisitc classifier.
   - Use the oversampling **SMOTE** algorithm to resample the dataset and train a logisitc classifier.
   - Use the undersampling **ClusterCentroids** algorithm to resample the dataset and train a logisitc classifier.
2. Using a Combination Sampling Technique to Predict Credit Risk
   - Use the combinatorial over- and under-sampling **SMOTEENN** algorithm to resample the dataset and train a logisitc classifier.
3. Using Ensemble Learning Techniques to Resample Data Predict Credit Risk
   - Use the ensemble **BalancedRandomForestClassifier** algorithm to resample the dataset and train an ensemble classifier.
   - Use the ensemble **EasyEnsembleClassifier** algorithm to resample the dataset and train an ensemble classifier. 

The evaluation of each of the six models can be found in the Results section with images of the confusion matrix and classification report generated for each model provided for support.

Finally, the Summary section will provide a summary of the results of the machine learning models and discuss which model would best predict credit risk. 

# Analysis
## Data Preprocessing
This analysis was spread across two differnt .ipynb files, but the data cleaning and preprocessing was the same in both files. I included my data cleaning process below with explanations in the comments. I first removed rows that did not seem useful based on visual examination of the data. Next, I used **NumPy** to change percentages to floating integer. Then, I used **Pandas** to remove null rows and columns and encode my features and target. 
```
### SETTING UP DATAFRAME
# Columns were first chosen through visual evaluation. 
# If a column contained all '0' values or all '1' values, it was not included
# If a column contained the same value for every row, it was not included 
columns = [
    "loan_amnt", "term", "int_rate", "installment", "grade",
    "home_ownership", "annual_inc", "verification_status", "issue_d","loan_status",
    "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "total_acc", "initial_list_status", "out_prncp", "out_prncp_inv", 
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int","last_pymnt_amnt", 
    "next_pymnt_d", "collections_12_mths_ex_med", "application_type", "tot_coll_amt", "tot_cur_bal",
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "mths_since_rcnt_il",
    "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl", "inq_last_12m", 
    "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths", 
    "delinq_amnt", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", 
    "mort_acc", "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl", 
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl", 
    "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats", "num_tl_90g_dpd_24m", "num_tl_op_past_12m", 
    "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies", "tot_hi_cred_lim", "total_bal_ex_mort", 
    "total_bc_limit", "total_il_high_credit_limit"
]

target = ["loan_status"]

### LOADING DATAFRAME AND CLEANING VALUES
# Load the data
file_path = 'LoanStats_2019Q1.csv'
df = pd.read_csv(file_path, skiprows=1)[:-2]
df = df.loc[:, columns].copy()

# Any coloumns where all values were null we dropped
df = df.dropna(axis='columns', how='all')

# Any rows that contained null values were dropped
df = df.dropna()

# The `Issued` loan status was removed from the loan_status column
issued_mask = df['loan_status'] != 'Issued'
df = df.loc[issued_mask]

# The interest rate was converted to numerical
df['int_rate'] = df['int_rate'].str.replace('%', '')
df['int_rate'] = df['int_rate'].astype('float') / 100

# The target column values were converted to low_risk and high_risk based on their values
x = {'Current': 'low_risk'}   
df = df.replace(x)

x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')    
df = df.replace(x)

df.reset_index(inplace=True, drop=True)

### ENCODING FEATURES AND TARGET
# Encode feature columns with string values 
df_binary_encoded = pd.get_dummies(df, columns = [
    "term",
    "grade",
    "home_ownership",
    "issue_d",
    "verification_status",
    "initial_list_status",
    "next_pymnt_d",
    "application_type"
])

# Encode target column
# low_risk: 0
# high_risk: 1
x = {'low_risk': 0}   
df_binary_encoded = df_binary_encoded.replace(x)
y = {'high_risk': 1}
df_binary_encoded = df_binary_encoded.replace(y)
```
## Creating Testing and Training Data
The cleaned and encoded DataFrame contained _94 columns_. There were _93 features_ (listed in the "columns" variable in the code above) and 1 target variable -- _"loan_status"_. The "loan_status" column has two classes: >"low_risk", labeled "0", and "high_risk", labeled "1". The data was severely unblanced as there were 70,699 samples for "low_risk" class and only 358 samples for the "high_risk" class.
```
# Creating the features variable
X = df_binary_encoded.drop(columns="loan_status")

# Creating the target variable
y = df_binary_encoded["loan_status"]

# Checking the balance of the target values
# low_risk: 0
# high_risk: 1
y.value_counts()
   #Output: 0    70669
            1      358
            Name: loan_status, dtype: int64
```
The data was split into training and testing sets using the **sklearn.model_selection** package. I used the defualt parameters of the **train_test_split** function so that 75% of the data was used for training and only 25% of the data was used for testing. _For consistency and reproducibility sake, I chose a **random_state** of 1 and used this in every model throughout the analysis_. To confirm that the dataset is in fact imbalanced, I created a **Counter** instance using the **collections** module to count the samples in each class. This instance reported a count of the classes in the "y_train" dataset of 52,996 samples for the 0 (low_risk) class and 274 samples for the 1 (high_risk) class. 
```
# Normal train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Confirming imbalance in the training set
Counter(y_train)
   # Output: Counter({0: 52996, 1: 274})
```
## Using Oversampling and Undersampling Techniques to Predict Credit Risk
The code referenced in this subsection can be found in the [credit_risk_resampling.ipynb file](credit_risk_resampling.ipynb). I imported the necessary dependencies for each section directly above the code for each section for clarity reasons. However, the resampled data in every section was fit to a logistic regression model which I imported the dependency to create these logistic regression models at the top of the code using the **LogisticRegression** class from the **sklearn.linear_model** module. 

### Oversampling Techniques 
Using oversampling techniques, the minority class was resampled to make it larger. Then, the resampled data was fit to logistic regression models.

#### Model 1: **RandomOverSampler** algorithm 
Using the naive random over-sampling technique algorithm, instances of the minority class were randomly selected, reused and added to the training set until the majority and minority classes were balanced. For this section I imported the **RandomOverSampler** class from the **imblearn.over_sampling** module. I first created an instance of the **RandomOverSampler** algorithm, then resampled the data using this algorithm, and, finally, fit the resampled data to a **LogisticRegression** model using the **lbfgs** solver. The computer generated predictions using the resampled data. The **RandomOverSampler** algorithm added 52,722 samples to the minority class. 
```
# Resampling the training data with the RandomOversampler
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Confirming enlargement of the minority class
Counter(y_resampled)
   # Output: Counter({0: 52996, 1: 52996})

# Training the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)

# Generating preciditions with trained model
y_pred = model.predict(X_test)

```
Below is a screencap of metrics generated for this model which are further recapped in the Results seciton. The process of gathering these metrics is also explained in the Results section.

![Random_OS_Metrics](https://user-images.githubusercontent.com/104794100/197619398-b82a1d9a-90ec-4e5d-894b-c9254b1d34af.png)

#### Model 2: **SMOTE** algorithm
Using the Synthetic Minority Over-Sampling Technqiue (SMOTE) algorithm, instances of the minority class were interpolated added to the training set until the majority and minority classes were balanced. Instances from the minority class were added by creating _synthetic instances_ based on the neighboring values of the exisiting instances. For this section I imported the **SMOTE** class from the **imblearn.over_sampling** module. I first created an instance of the **SMOTE** algorithm, then resampled the data using this algorithm, and, finally, fit the resampled data to a **LogisticRegression** model using the **lbfgs** solver. Then, the computer generated predictions using the resampled data. The **SMOTE** algorithm added 52,722 samples to the minority class.
```
# Resampling the training data with SMOTE
from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE(random_state=1, sampling_strategy='auto').fit_resample(
    X_train, y_train
)

# Confirming enlargement of the minority class
Counter(y_resampled)
   # Output: Counter({0: 52996, 1: 52996})

# Training the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)

# Generating preciditions with trained model
y_pred = model.predict(X_test)
```
Below is a screencap of metrics generated for this model which are further recapped in the Results seciton. The process of gathering these metrics is also explained in the Results section.

![SMOTE_OS_Metrics](https://user-images.githubusercontent.com/104794100/197619469-aa80142c-cc58-45cc-bb7a-6bfd914ae70d.png)

### Undersampling Technique
Using an under-sampling technique, the majority class was resampled to make it smaller. Then, the resampled data was fit to a logistic regression model.

#### Model 3: **ClusterCentroids** techinque
Using the cluster centroid technique algorithm, clusters of the majority class instances were identified and _synthetic instances_ (centroids) were generated. These instances were representative of the clusters and used as the data points for the majority class. For this section I imported the **ClusterCentroids** class from the **imblearn.under_sampling** module. I first created an instance of the **ClusterCentroids** algorithm, then resampled the data using this algorithm, and, finally, fit the resampled data to a **LogisticRegression** model using the **lbfgs** solver. The computer generated predictions using the resampled data. The **ClusterCentroids** algorithm reduced the majority class by 52,722 samples. 
```
# Resampling the data using the ClusterCentroids resampler
from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)

# Verifying reduction of the majority class
Counter(y_resampled)
   # Output: Counter({0: 274, 1: 274})

# Training the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=78)
model.fit(X_resampled, y_resampled)

# Generating preciditions with trained model
y_pred = model.predict(X_test)
```
Below is a screencap of metrics generated for this model which are further recapped in the Results seciton. The process of gathering these metrics is also explained in the Results section.

![ClusterCentroid_US_Metrics](https://user-images.githubusercontent.com/104794100/197619535-7455cd5b-9929-4f93-8827-3e91a582bd6f.png)

## Using a Combination Sampling Technique to Predict Credit Risk
The code referenced in this subsection can be found in the [credit_risk_resampling.ipynb file](credit_risk_resampling.ipynb). I imported the necessary dependencies for this section directly above the code for this section for clarity reasons. However, the resampled data in this section was fit to a logistic regression model which I imported the dependency to create this logistic regression model at the top of the code using the **LogisticRegression** class from the **sklearn.linear_model** module.

There are downsideds to over-sampling with the **SMOTE** algorithm because it relies on the immediate neighbors of a data point and fails to recognize the overall distribution of the data. This leads to noisy data because the newly generated data points can be heavily influenced by outliers. There are also downsides to any under-sampling techniques because it involves the overall loss of data. One way to deal with these challenges is to use a sampling technique that is a combination of oversampling and undersampling.

### Model 4: **SMOTEENN** algorithm
Using the Synthetic Minority Over-Sampling Technqiue and Edited Nearest Neighbors (SMOTEENN) algorithm, instances of the minoirty class were oversampled using the **SMOTE** algorithm, but then cleaned and under-sampled using the **ENN** algorithm. If the two nearest neighbors of a data point belonged to two different classes, that data point was dropped. For this section I imported the **SMOTEENN** class from the **imblearn.combine** module. I first created an instance of the **SMOTEENN** algorithm, then resampled the data using this algorithm, and, finally, fit the resampled data to a **LogisticRegression** model using the **lbfgs** solver. Then, the computer generated predictions using the resampled data. The **SMOTEENN** algorithm added 70,380 samples to the minority class, and it added 11,038 samples to the majority class. 
```
# Resampling the training data with SMOTEENN
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

# Confirming balance
Counter(y_resampled)
   # Output: Counter({0: 64034, 1: 70654})

# Training the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)

# Generating preciditions with trained model
y_pred = model.predict(X_test)
```
Below is a screencap of metrics generated for this model which are further recapped in the Results seciton. The process of gathering these metrics is also explained in the Results section.

![SMOTEENN_CS_Metrics](https://user-images.githubusercontent.com/104794100/197619686-73bb5b93-e4ac-4e0e-872c-14f2ec506f23.png)

## Using Ensemble Learning Technqiues to Resample Data Predict Credit Risk
The code referenced in this subsection can be found in the [credit_risk_ensemble.ipynb file](credit_risk_ensemble.ipynb). I imported the necessary dependencies for each section directly above the code for each section for clarity reasons. However, the resampled data in every section was fit to a logistic regression model which I imported the dependency to create these logistic regression models at the top of the code using the **LogisticRegression** class from the **sklearn.linear_model** module.

Ensemble learning is the process of combining multiple weaker models into one stronger model to help reduce bias.

### Model 5: **BalancedRandomForestClassifier** algorithm

### Model 6: **EasyEnsembleClassifier** algorithm

Results of the analysis: Explain the purpose of this analysis.
# Results: Using bulleted lists
- Accuracy scores
Models 1-6 produced the following accuracy scores:
- Precision scores
Models 1-6 produced the following precision scores:
- Recall scores
Models 1-6 produced the following accuracy scores:

# Summary
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.
