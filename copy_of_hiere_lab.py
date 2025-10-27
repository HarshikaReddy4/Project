

#hierarchial
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df = pd.read_csv("org_mooc.csv")

df = pd.get_dummies(df, columns=["City_Tier", "Occupation"], drop_first=True)

df["Savings_Rate"] = df["Desired_Savings"] / df["Income"]
df["Debt_Rate"] = df["Loan_Repayment"] / df["Income"]
df["Expense_to_Income"] = (df["Rent"] + df["Groceries"] + df["Transport"] + df["Eating_Out"] +
                           df["Entertainment"] + df["Utilities"] + df["Healthcare"] + df["Education"] + df["Miscellaneous"]) / df["Income"]
df["Liquid_Term"] = df["Desired_Savings"] / (df["Income"] - df["Desired_Savings"])

features_for_clustering = df[["Savings_Rate", "Debt_Rate", "Expense_to_Income", "Liquid_Term"]].fillna(0)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_for_clustering)

linkage_matrix = linkage(features_scaled, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.xlabel("Data Points")
plt.ylabel("Cluster Distance")
plt.title("Dendrogram for Hierarchical Clustering")
plt.show()

df["Stability"] = fcluster(linkage_matrix, t=3, criterion='maxclust') - 1


plt.figure(figsize=(6, 4))
sns.countplot(x=df["Stability"], palette="coolwarm")
plt.xlabel("Stability Clusters")
plt.ylabel("Number of People")
plt.title("Distribution of Stability Clusters")
plt.show()


sns.pairplot(df, hue="Stability", vars=["Savings_Rate", "Debt_Rate", "Expense_to_Income", "Liquid_Term"], palette="coolwarm")
plt.show()

calinski_harabasz = calinski_harabasz_score(features_scaled, df["Stability"])



print(f"Calinski-Harabasz Score (CHS): {calinski_harabasz:.4f}")
