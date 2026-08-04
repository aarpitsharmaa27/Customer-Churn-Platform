import pandas as pd

df = pd.read_csv("data/customer_churn.csv")


print(df)
#df.shape

#print(df.head())
#df.shape

df.info()

print(df.describe())

print(df.isnull())

print(df.isnull().sum() )
df.info()

print(df["TotalCharges"].head(20))

print(df["TotalCharges"].unique())

print(df[df["TotalCharges"] == " "])

df = df[df["TotalCharges"] != " " ]


print(df)
print(df.info)

df ["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

print(df.info)

df.shape
print(df.describe())

df.info()

print(df["Churn"].value_counts())

print(df["Churn"].value_counts(normalize=True))

df.info()

import matplotlib.pyplot as plt    # Matploylib 

plt.figure(figsize=(10,6))

df["Churn"].value_counts().plot(kind="bar")

plt.title("Customer Churn Count")

plt.xlabel("Churn")

plt.ylabel("Number of Customers")

#plt.show()

'''print(pd.crosstab(df["SeniorCitizen"],df["Churn"]))

print(pd.crosstab(
    df["SeniorCitizen"],
    df["Churn"],
    normalize="index"
) * 100) '''

'''import seaborn as sns       # Seaborn

plt.figure(figsize=(10,6))

sns.countplot(x="SeniorCitizen", hue= "Churn", data = df)

plt.title("Senior Citizen Churn")

plt.xlabel("Senior Citizen")

plt.ylabel("Number of Customers")

plt.show()'''



'''import matplotlib.pyplot as plt  
import seaborn as sns

plt.figure(figsize=(10,6))

sns.countplot(
    x= "SeniorCitizen",
    hue = "Churn",
    data = df
)

plt.title("Senior Citizen Churn")
plt.xlabel("Senior Citizen")
plt.ylabel("Number of Customers")

plt.show() '''


print(pd.crosstab(df["SeniorCitizen"], 
            df["Churn"],
            normalize ="index"
            ) * 100)

print(df.describe())

print(df.head())

print(df["Contract"].value_counts())

'''pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100'''


'''import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

sns.countplot(
    x = "Contract",
    hue = "Churn",
    data = df
)

plt.title("Contract Type Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")   

plt.show() '''

print(df["PaymentMethod"].value_counts())

print(pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize = "index"
) * 100  )

print(df["InternetService"].value_counts())

print(pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) *100)

print(df["MonthlyCharges"].describe())


'''sns.boxplot(         # BOXPLOT
    x = "Churn",
    y = "MonthlyCharges",
    data = df
)

plt.show()


print(df["tenure"].describe())'''


# plt.figure(figsize=(10,6))

'''sns.boxplot(
    x = "Churn",
    y = "tenure",

    data = df
)

plt.title("Tenure vs churn")
plt.xlabel("Churn")
plt.ylabel("tenure")

plt.show() '''
df["Churn"] = df["Churn"].replace({
    "Yes": 1,
    "No": 0
})

print(df["Churn"].value_counts())
           

x = df.drop("Churn", axis = 1)
y = df["Churn"]


print(x.head())
print(y.head())


print(df["Churn"])
print(df[["Churn"]])



from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(
    x,
    y,
    test_size= 0.2
)


print(X_train.shape)
print(y_train.shape)

print(df.shape)
print(X_test.shape)
print(y_test.shape)



X_train = pd.get_dummies(
    X_train,
    columns=[
        "gender",
        "Partner",
        "Dependents",
        "InternetService",
        "PaymentMethod"
    ],
    drop_first = True
)
print(X_train)

print(X_train.columns)

print(X_train.shape)

print(X_train.head())   

X_test = pd.get_dummies(
    X_test,
    columns=[
        "gender",
        "Partner",
        "Dependents",
        "InternetService",
        "PaymentMethod"
    ],
    drop_first = True
)


X_train["Contract"] = X_train["Contract"].replace({
    "Month-to-month" : 0,
    "One year" : 1,
    "Two year": 2
})

print(X_train["Contract"].head())



print(X_train["Contract"].tail())

X_test["Contract"] = X_test["Contract"].replace({
    "Month-to-month" : 0 , 
    "One year" : 1 ,
    "Two year" : 2

})

print(X_test["Contract"].head())
print(X_test["Contract"].tail())

print(X_train["Contract"].dtype)
print(X_test["Contract"].dtype)

print(X_train["Contract"].unique())
print(X_test["Contract"].unique())  

X_train["Contract"] = X_train["Contract"].astype(int)
X_test["Contract"] = X_test["Contract"].astype(int)

print(X_train["Contract"].dtype)










X_train = X_train.drop("customerID", axis = 1)
X_test = X_test.drop("customerID", axis = 1)

print(X_train.columns)

X_train = pd.get_dummies(
    X_train,
    columns = [
        "PhoneService",
        "MultipleLines",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
        "OnlineSecurity",
        "OnlineBackup",


    ],
    drop_first = True
)

print(X_train.dtypes)


X_test = pd.get_dummies(
    X_test,
    columns = [
        "PhoneService",
        "MultipleLines",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
        "OnlineSecurity",
        "OnlineBackup",


    ],
    drop_first = True
)

print(X_test.dtypes)

print(y_train.shape)
print(y_train.describe())

print(X_train.columns)
print(X_train)
print(X_train.dtypes)


#print(y_train.unique())



print(X_train.dtypes)

print("--------------------------------")

print(X_train.select_dtypes(include=["object", "string"]).columns)  

print(y_train.dtypes)
print(type(y_train))
print(type(y_train.iloc[0]))
print(y_train.unique())

y_train = y_train.astype(int)

print(y_train.dtypes)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter= 10000)

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

print(y_pred)

print(y_test.unique())

print(type(y_test))
print(type(y_pred))


print(y_test.dtype)
print(y_pred.dtype)

print(y_test.shape)
print(y_pred.shape)


y_test = y_test.astype(int)
print(y_test.dtype)

from sklearn.metrics import accuracy_score

accuracy  = accuracy_score(y_test, y_pred)

print(accuracy)

print(f"Accuracy: {accuracy:.4f}")


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

y_prob = model.predict_proba(X_test)
print(y_prob[:10])


from sklearn.metrics import roc_auc_score

roc = roc_auc_score(y_test, y_prob[:,1])

print(f"ROC-AUC Score: {roc:.4f}")