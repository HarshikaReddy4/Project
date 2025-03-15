import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Financial Stability Clustering",
    page_icon="💰",
    layout="wide"
)

# Add title and description
st.title("💰 Financial Stability Clustering Analysis")
st.markdown("""
This application analyzes financial patterns using DBSCAN clustering to identify different financial stability profiles.
Upload your CSV file containing financial data to begin the analysis.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your financial data CSV", type=["csv"])

# Sidebar for parameters
st.sidebar.header("Clustering Parameters")
eps = st.sidebar.slider("DBSCAN Epsilon (Neighborhood Size)", 0.1, 2.0, 0.5, 0.1)
min_samples = st.sidebar.slider("DBSCAN Min Samples", 3, 15, 5, 1)
use_pca = st.sidebar.checkbox("Use PCA for Dimensionality Reduction", True)
pca_components = st.sidebar.slider("PCA Components", 2, 4, 2, 1) if use_pca else None

def run_clustering_analysis(df):
    # Feature Engineering
    with st.expander("Feature Engineering Details", expanded=False):
        st.markdown("""
        The following features are calculated:
        - **Savings Rate**: Desired savings divided by income
        - **Debt Rate**: Loan repayment divided by income
        - **Expense to Income Ratio**: Total expenses divided by income
        - **Liquid Term**: Desired savings divided by (income minus desired savings)
        """)
    
    df["Savings_Rate"] = df["Desired_Savings"] / df["Income"]
    df["Debt_Rate"] = df["Loan_Repayment"] / df["Income"]
    df["Expense_to_Income"] = (df["Rent"] + df["Groceries"] + df["Transport"] + df["Eating_Out"] +
                           df["Entertainment"] + df["Utilities"] + df["Healthcare"] + df["Education"] + df["Miscellaneous"]) / df["Income"]
    df["Liquid_Term"] = df["Desired_Savings"] / (df["Income"] - df["Desired_Savings"])
    
    # Prepare Features for Clustering
    features_for_clustering = df[["Savings_Rate", "Debt_Rate", "Expense_to_Income", "Liquid_Term"]].fillna(0)
    
    # Display feature statistics
    st.subheader("Feature Statistics")
    st.dataframe(features_for_clustering.describe())
    
    # Scaling
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features_for_clustering)
    
    # Dimensionality Reduction if selected
    if use_pca:
        pca = PCA(n_components=pca_components)
        features_scaled = pca.fit_transform(features_scaled)
        
        # Display PCA explained variance
        explained_variance = pca.explained_variance_ratio_
        st.subheader("PCA Explained Variance")
        fig = px.bar(
            x=[f"Component {i+1}" for i in range(pca_components)],
            y=explained_variance,
            labels={"x": "Principal Component", "y": "Explained Variance Ratio"}
        )
        st.plotly_chart(fig)
    
    # DBSCAN Clustering
    with st.spinner("Running DBSCAN clustering..."):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features_scaled)
        
        # Check if meaningful clusters were found
        if len(set(labels)) < 2 or -1 in labels and np.sum(labels == -1) / len(labels) > 0.5:
            st.warning("⚠️ The current parameters result in poor clustering. Try adjusting epsilon and min_samples.")
            return
        
        df["Stability_Cluster"] = labels
    
    # Display clustering metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_points = np.sum(labels == -1)
    
    st.subheader("Clustering Results")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Number of Clusters", n_clusters)
    with col2:
        st.metric("Noise Points", noise_points)
    with col3:
        if n_clusters > 1:
            silhouette = silhouette_score(features_scaled, labels)
            st.metric("Silhouette Score", f"{silhouette:.4f}")
    with col4:
        if n_clusters > 1:
            davies_bouldin = davies_bouldin_score(features_scaled, labels)
            st.metric("Davies-Bouldin Score", f"{davies_bouldin:.4f}")
    
    # Cluster distribution
    st.subheader("Cluster Distribution")
    cluster_counts = df["Stability_Cluster"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]
    
    fig = px.pie(
        cluster_counts, 
        values="Count", 
        names="Cluster",
        title="Distribution of Clusters",
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig)
    
    # Cluster profiles
    st.subheader("Cluster Profiles")
    
    cluster_profiles = df.groupby("Stability_Cluster")[["Savings_Rate", "Debt_Rate", "Expense_to_Income", "Liquid_Term"]].mean().reset_index()
    
    fig = px.parallel_coordinates(
        cluster_profiles,
        color="Stability_Cluster",
        labels={"Stability_Cluster": "Cluster", "Savings_Rate": "Savings Rate", 
                "Debt_Rate": "Debt Rate", "Expense_to_Income": "Expense to Income", 
                "Liquid_Term": "Liquid Term"},
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Parallel Coordinates Plot of Cluster Profiles"
    )
    st.plotly_chart(fig)
    
    # Interactive scatter plot matrix
    st.subheader("Interactive Feature Relationships by Cluster")
    
    scatter_df = df[["Savings_Rate", "Debt_Rate", "Expense_to_Income", "Liquid_Term", "Stability_Cluster"]].copy()
    scatter_df["Stability_Cluster"] = scatter_df["Stability_Cluster"].astype(str)
    
    fig = px.scatter_matrix(
        scatter_df,
        dimensions=["Savings_Rate", "Debt_Rate", "Expense_to_Income", "Liquid_Term"],
        color="Stability_Cluster",
        opacity=0.7,
        title="Scatter Plot Matrix of Financial Features"
    )
    st.plotly_chart(fig)
    
    # 3D visualization if PCA was used with 3 components
    if use_pca and pca_components >= 3:
        st.subheader("3D Cluster Visualization")
        
        pca_df = pd.DataFrame(
            features_scaled, 
            columns=[f"PC{i+1}" for i in range(pca_components)]
        )
        pca_df["Cluster"] = labels
        
        fig = px.scatter_3d(
            pca_df, x="PC1", y="PC2", z="PC3",
            color=pca_df["Cluster"].astype(str),
            opacity=0.7,
            title="3D Visualization of Clusters (PCA)",
            labels={"color": "Cluster"}
        )
        st.plotly_chart(fig)
    
    # Table with detailed cluster information
    st.subheader("Detailed Cluster Information")
    cluster_details = df.groupby("Stability_Cluster").agg({
        "Savings_Rate": ["mean", "median", "std"],
        "Debt_Rate": ["mean", "median", "std"],
        "Expense_to_Income": ["mean", "median", "std"],
        "Liquid_Term": ["mean", "median", "std"],
        "Income": ["mean", "count"]
    }).reset_index()
    
    cluster_details.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in cluster_details.columns]
    cluster_details = cluster_details.rename(columns={"Income_count": "Count"})
    
    st.dataframe(cluster_details)
    
    # Download results button
    st.download_button(
        label="Download Results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="financial_stability_results.csv",
        mime="text/csv"
    )
    
    return df

# Run analysis if file is uploaded
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if required columns exist
        required_columns = ["Income", "Desired_Savings", "Loan_Repayment", "Rent", "Groceries", 
                           "Transport", "Eating_Out", "Entertainment", "Utilities", 
                           "Healthcare", "Education", "Miscellaneous"]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Error: Missing required columns: {', '.join(missing_columns)}")
        else:
            # Show raw data preview
            with st.expander("Raw Data Preview"):
                st.dataframe(df.head())
            
            # Run the analysis
            results_df = run_clustering_analysis(df)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Please upload a CSV file to begin the analysis.")
    
    # Sample data description
    st.markdown("""
    ### Expected Data Format
    
    Your CSV should contain the following columns:
    - `Income`: Monthly income
    - `Desired_Savings`: Target monthly savings
    - `Loan_Repayment`: Monthly loan payments
    - Expense categories:
        - `Rent`
        - `Groceries`
        - `Transport`
        - `Eating_Out`
        - `Entertainment`
        - `Utilities`
        - `Healthcare`
        - `Education`
        - `Miscellaneous`
    """)

# Footer
st.markdown("""
---
### About Financial Stability Clustering

This app uses DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to identify 
patterns in financial behaviors. The algorithm groups similar financial profiles together, 
helping to identify different financial stability segments.

**Key Metrics:**
- **Savings Rate**: Indicates what portion of income is saved
- **Debt Rate**: Shows debt burden relative to income
- **Expense to Income Ratio**: Measures overall spending relative to income
- **Liquid Term**: Represents financial resilience (how long savings would last)
""")
