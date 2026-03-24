# EXNO:4-DS
# Name: Dinesh M
# Register:212222043002
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```

#Import necessary libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

#Load Dataset

df = pd.read_csv('Bmi.csv')  

print("Original Dataset:")
print(df.head())

```
```

#Handle Missing Values

df = df.dropna()

```
```

#Standardscaling

df_std = df.copy()
scaler_std = StandardScaler()
df_std[['Height', 'Weight']] = scaler_std.fit_transform(df_std[['Height', 'Weight']])

print("\nStandard Scaled Data:")
print(df_std.head())

```
```

#Minmaxscaling

df_minmax = df.copy()
scaler_minmax = MinMaxScaler()
df_minmax[['Height', 'Weight']] = scaler_minmax.fit_transform(df_minmax[['Height', 'Weight']])

print("\nMin-Max Scaled Data:")
print(df_minmax.head())

```
```

#MaxAbsscaling

df_maxabs = df.copy()
scaler_maxabs = MaxAbsScaler()
df_maxabs[['Height', 'Weight']] = scaler_maxabs.fit_transform(df_maxabs[['Height', 'Weight']])

print("\nMaxAbs Scaled Data:")
print(df_maxabs.head())

```
```
#Robustscaling

df_robust = df.copy()
scaler_robust = RobustScaler()
df_robust[['Height', 'Weight']] = scaler_robust.fit_transform(df_robust[['Height', 'Weight']])

print("\nRobust Scaled Data:")
print(df_robust.head())


```
```

#Save Scaled Datasets

#df_std.to_csv("BMI_StandardScaled.csv", index=False)
#df_minmax.to_csv("BMI_MinMaxScaled.csv", index=False)
#df_maxabs.to_csv("BMI_MaxAbsScaled.csv", index=False)
#df_robust.to_csv("BMI_RobustScaled.csv", index=False)

print("\nFeature Scaling Completed Successfully.")

```
```

#Import Required Libraries

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

```
```

#Read the CSV file

df = pd.read_csv("income.csv")

print("Dataset Preview:")
print(df.head())

```
```

#Encode Cateorigal Variables

categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation',
                       'relationship', 'race', 'gender', 'nativecountry']

df[categorical_columns] = df[categorical_columns].astype('category').apply(lambda x: x.cat.codes)

```
```

#Encode Target Variable

if df['SalStat'].dtype == 'object':
    df['SalStat'] = df['SalStat'].astype('category').cat.codes

```
```

#Separate features and Target

X = df.drop(columns=['SalStat'])
y = df['SalStat']

```
```

#Scale Data for Chi-Square (Non-negative required)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

```
```

#Filter Method: Chi-Square

selector_chi2 = SelectKBest(score_func=chi2, k=6)
selector_chi2.fit(X_scaled, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("\nChi-Square Selected:", list(selected_features_chi2))

```
```

#Filter Method: ANOVA

selector_anova = SelectKBest(score_func=f_classif, k=5)
selector_anova.fit(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nANOVA Selected:", list(selected_features_anova))

```
```

#Wrapper Method: RFE

logreg = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=logreg, n_features_to_select=6)
rfe.fit(X, y)
selected_features_rfe = X.columns[rfe.support_]
print("\nRFE Selected:", list(selected_features_rfe))

```
```

#Embedded Method: SelectFromModel

rf = RandomForestClassifier(n_estimators=100, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf.fit(X_train, y_train)

selector_embedded = SelectFromModel(rf, threshold="mean")
selector_embedded.fit(X_train, y_train)

selected_features_embedded = X.columns[selector_embedded.get_support()]
print("\nEmbedded Method Selected:", list(selected_features_embedded))

```
```

#Accuracy using Embedded Features

X_train_sel = selector_embedded.transform(X_train)
X_test_sel = selector_embedded.transform(X_test)

rf.fit(X_train_sel, y_train)
y_pred = rf.predict(X_test_sel)

print("\nModel Accuracy (Embedded Method):", accuracy_score(y_test, y_pred))

```

<img width="461" height="215" alt="image" src="https://github.com/user-attachments/assets/3aef0e1c-c3c9-40cb-abfb-b2f7298e91a6" />

<img width="545" height="246" alt="image" src="https://github.com/user-attachments/assets/4722d9fa-5aea-4084-8aff-58a3896a4d97" />

<img width="547" height="224" alt="image" src="https://github.com/user-attachments/assets/bc05a944-db56-4ec1-903f-e8d2985f5a25" />

<img width="482" height="235" alt="image" src="https://github.com/user-attachments/assets/bb50d94c-b84d-4f92-8936-ab4a5c7afea3" />

<img width="632" height="285" alt="image" src="https://github.com/user-attachments/assets/a77b1f5d-cb00-4a77-b37f-cb902bf64910" />

<img width="811" height="533" alt="image" src="https://github.com/user-attachments/assets/00ea1f38-f586-40f6-b928-376633a0aad0" />

<img width="1096" height="363" alt="image" src="https://github.com/user-attachments/assets/097f951a-4c76-41a4-9cfe-8337aa894a26" />

<img width="861" height="312" alt="image" src="https://github.com/user-attachments/assets/78a72046-c473-48a7-91af-9150d36cec28" />

<img width="861" height="312" alt="image" src="https://github.com/user-attachments/assets/23bf65ac-b504-447b-9777-d482b20b3a72" />

<img width="825" height="404" alt="image" src="https://github.com/user-attachments/assets/0614f3f9-19e7-45f3-b991-7c401fe31b35" />

# RESULT:
Thus the Feature Scaling and selection Executed successfully.
