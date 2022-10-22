# Credit_Card_Analysis_Machine_Learning
# Overview 
## Purpose
Machine learning can be used to predict credit card risk and solve loan approval questions becasue credit risk is ultimatley an unbalanced classification problem. Previous data on loan approval can be used to train machine learning models to determine if someone if worthy of a loan, but good loans undeniably outnumber risky loans, and this class imbalance must be considered when choosing model for prediciting credit risk. The purpose of this project was to use the Python's Sciki-learn (**sklearn**) and Imbalanced-learn (**imblearn**) libraries to build and evaluate six differnt classification models that use differnt resampling and ensemble learning algorithms. The goal was to determine which of the six models are best at predicting credit risk based evidence from confusion matrices and classification reports. 
## Analysis Roadmap
The [credit card dataset](LoanStats_2019Q1.csv.zip) used in this project to train machine learning models came from LendingClub, a peer-to-peer lending services company. This dataset was imported to a DataFrame, cleaned, and encoded using the **Pandas** library. 

This project was broken down into following three main parts found in the Analysis Section:
1. Using Oversampling and Undersampling Algorithms to Predict Credit Risk
   - Use the oversampling **RandomOverSampler** algorithm to resample the dataset and train a logisitc classifier.
   - Use the oversampling **SMOTE** algorithm to resample the dataset and train a logisitc classifier.
   - Use the undersampling **ClusterCentroids** algorithm to resample the dataset and train a logisitc classifier.
2. Using a Combination Sampling Algorithm to Predict Credit Risk
   - Use the combinatorial over- and under-sampling **SMOTEENN** algorithm to resample the dataset and train a logisitc classifier.
3. Using Ensemble Classifiers to Resample Data Predict Credit Risk
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

###ENCODING FEATURES AND TARGET
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
The data was split into training and testing sets using the defualt parameters of the **train_test_split** module so that 75% of the data was used for training and only 25% of the data was used for testing. _For consistency and reproducibility sake, I chose a **random_state** of 1 and used this in every model throughout the analysis_.
```
# Normal train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Confirming imbalance in the training set
Counter(y_train)
   # Output: Counter({0: 52996, 1: 274})
```
## Using Oversampling and Undersampling Algorithms to Predict Credit Risk
The code referenced in this subsection can be in the [credit_risk_resampling.ipynb file](credit_risk_resampling.ipynb).

### Oversampling algorithms 

#### **RandomOverSampler** algorithm model

#### **SMOTE** algorithm model

### Undersampling algorithm

#### **ClusterCentroids** algorithm model

## Using a Combination Sampling Algorithm to Predict Credit Risk

### **SMOTEENN** algorithm model

## Using Ensemble Classifier Algorithms to Resample Data Predict Credit Risk

### **BalancedRandomForestClassifier** algorithm

### **EasyEnsembleClassifier** algorithm

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
