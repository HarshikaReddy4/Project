

#k means clustering
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans

df = pd.read_csv("org_mooc.csv")

cat_features = ["City_Tier", "Occupation"]
df = pd.get_dummies(df, columns=cat_features, drop_first=True)

df["Savings_Rate"] = df["Desired_Savings"] / df["Income"]
df["Debt_Rate"] = df["Loan_Repayment"] / df["Income"]
df["Expense_to_Income"] = (df["Rent"] + df["Groceries"] + df["Transport"] + df["Eating_Out"] +
                           df["Entertainment"] + df["Utilities"] + df["Healthcare"] + df["Education"] + df["Miscellaneous"]) / df["Income"]
df["Liquid_Term"] = df["Desired_Savings"] / (df["Income"] - df["Desired_Savings"])

features_for_clustering = df[["Savings_Rate", "Debt_Rate", "Expense_to_Income", "Liquid_Term"]].fillna(0)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Stability"] = kmeans.fit_predict(features_for_clustering)

X = df.drop(columns=["Stability", "Income", "Desired_Savings", "Desired_Savings_Percentage"])
y = df["Stability"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)



from sklearn.metrics import f1_score, calinski_harabasz_score


ch_score = calinski_harabasz_score(features_for_clustering, df["Stability"])
print(f"Calinski-Harabasz Score: {ch_score:.3f}")


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features_for_clustering, df["Stability"], test_size=0.2, random_state=42, stratify=df["Stability"])

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

f1 = f1_score(y_test, y_pred, average="weighted")
print(f"F1 Score: {f1:.3f}")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler




features = ["Savings_Rate", "Debt_Rate", "Expense_to_Income", "Liquid_Term"]
features_for_clustering = df[features]


scaler = StandardScaler()
features_for_clustering = scaler.fit_transform(features_for_clustering)


inertia = []
K_range = range(1, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features_for_clustering)
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='dashed')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(x=df["Stability"], palette="Set2")
plt.xlabel("Cluster Label")
plt.ylabel("Count")
plt.title("Cluster Distribution")
plt.show()

df_sampled = df.groupby("Stability").sample(n=100, replace=True, random_state=42)


custom_palette = {0: "red", 1: "blue", 2: "green", 3: "purple", 4: "orange"}


sns.pairplot(df_sampled, hue="Stability", vars=["Savings_Rate", "Debt_Rate", "Expense_to_Income", "Liquid_Term"], palette=custom_palette)
plt.show()
