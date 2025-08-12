#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 16:34:33 2025

@author: bradenmindrum
"""

#%%

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np

#%%

df = pd.read_csv("star_data.csv")

#%%

"""
Perform some preliminary work checking data types, missing values,
range of values, and etc.
"""

print("\n-----------------------------------------------\n")
print("Checking data types and missing values:")
print(df.dtypes)
print(f"Has null values: {df.isnull().values.any()}")
print(f"Has na values: {df.isna().values.any()}")
print("\n-----------------------------------------------\n")

# Change star type to object
df["Star type"] = df["Star type"].astype('object')

#%%

for col in df.columns:
    x = df[f"{col}"]
    if x.dtype == object:
        plt.bar(df[f"{col}"].value_counts().index,
                df[f"{col}"].value_counts().values,
                width=0.5)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()
    else:
        plt.hist(x)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()
    del x
    
#%%

"""
Important: the sklearn.tree.DecisionTreeClassifier does not support 
categorical variable. We will recreate the data frame with one hot
encodings of the categorical variables.
Also, the XXXXXX
"""

# Create one-hot encodings of categorical variables that are not the targets
for col in df.columns:
    if col == "Star type":
        continue
    x = df[f"{col}"]
    if x.dtype == object:
        for typ in x.value_counts().index:
            df.insert(len(df.columns), f"{col}: {typ}",
                      np.where(df[f"{col}"] == typ, 1, 0))
        # Delete original categorical variables
        df.drop(f"{col}", axis=1, inplace=True)
    del x         

df["Star type"] = df["Star type"].astype(str)    

#%%

"""
Split data into train, validation, and test sets. Due to small data size,
a 60/20/20 ratio is used.
"""
x_train, x_test, y_train, y_test = train_test_split(df.drop("Star type", axis=1), 
                                                    df["Star type"], 
                                                    test_size=0.4, 
                                                    random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                test_size=0.5, 
                                                random_state=42)

#%%

"""
Methodology: Find the full tree. Pick the pruned tree that has the 
best classification rate on the validation set.
"""
                                           
tree = sklearn.tree.DecisionTreeClassifier(random_state=42)
path = tree.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas = path.ccp_alphas

#%%

options = []
for alpha in ccp_alphas:
    tree_option = sklearn.tree.DecisionTreeClassifier(random_state=42, 
                                                      ccp_alpha=alpha)
    tree_option.fit(x_train, y_train)
    options.append(tree_option)
    del tree_option

#%%

train_scores = [option.score(x_train, y_train) for option in options]
val_scores = [option.score(x_val, y_val) for option in options]

plt.plot(ccp_alphas, train_scores, marker="o", label="Train", drawstyle="steps-post")
plt.xlabel("Alpha")
plt.ylabel("Training Accuracy")
plt.show()

plt.plot(ccp_alphas, val_scores, marker="o", label="Train", drawstyle="steps-post",
         color="orange")
plt.xlabel("Alpha")
plt.ylabel("Validation Accuracy")
plt.show()

"""
Apparently star classification is a perfect and non-messy problem...
Nevertheless, let us verify with the testing data.
"""

#%%

model = options[0]
print(f"Accuracy on testing data: {model.score(x_test, y_test)}")

#%%

"""
The training and validation prediction scores were perfect. The test score
did not get a perfect score. Further inspect the cause...
"""

predictions = model.predict(x_test) == np.array(y_test)
false_index = np.where(predictions == False)[0] # Only has one

for col in x_test.columns:
    print(f"{col}: {int(x_test[col].iloc[false_index].iloc[0])}")
print(f"Class: {int(y_test.iloc[false_index].iloc[0])}")

#%%

# Print tree
sklearn.tree.plot_tree(model, feature_names=x_train.columns)

#%%

























































