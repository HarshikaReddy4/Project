import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Set page config
st.set_page_config(
    page_title="Fiscallytic",
    page_icon="💰",
    layout="wide"
)

# Add title and description
st.title("💰 Fiscallytic: Financial Stability Analysis")
st.markdown("""
This app uses machine learning (DBSCAN clustering) to analyze your financial stability based on your income, expenses, savings, and debt information.
Enter your financial details below to receive your personalized stability assessment.
""")

# Create a sidebar for model parameters
with st.sidebar:
    st.header("ML Model Parameters")
    st.markdown("Adjust DBSCAN parameters for classification")
    eps = st.slider("Epsilon (Neighborhood Size)", 0.1, 1.0, 0.5, 0.1)
    min_samples = st.slider("Min Samples", 2, 10, 5, 1)
    
    st.markdown("---")
    
    # Add information about DBSCAN
    st.markdown("""
    ### About DBSCAN
    
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions.
    
    - **Epsilon**: The maximum distance between two samples for one to be considered as in the neighborhood of the other
    - **Min Samples**: The number of samples in a neighborhood for a point to be considered as a core point
    """)

# Define the form for user input
st.subheader("Enter Your Financial Information")
with st.form("financial_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input("Monthly Income (₹)", min_value=0.0, value=0.0, step=1000.0)
        desired_savings = st.number_input("Monthly Savings Target (₹)", min_value=0.0, value=0.0, step=500.0)
        loan_repayment = st.number_input("Monthly Loan Repayments (₹)", min_value=0.0, value=0.0, step=500.0)
        rent = st.number_input("Monthly Rent/Mortgage (₹)", min_value=0.0, value=0.0, step=500.0)
        groceries = st.number_input("Monthly Groceries (₹)", min_value=0.0, value=0.0, step=500.0)
    
    with col2:
        transport = st.number_input("Monthly Transport (₹)", min_value=0.0, value=0.0, step=500.0)
        eating_out = st.number_input("Monthly Eating Out (₹)", min_value=0.0, value=0.0, step=500.0)
        entertainment = st.number_input("Monthly Entertainment (₹)", min_value=0.0, value=0.0, step=500.0)
        utilities = st.number_input("Monthly Utilities (₹)", min_value=0.0, value=0.0, step=500.0)
        healthcare = st.number_input("Monthly Healthcare (₹)", min_value=0.0, value=0.0, step=500.0)
        education = st.number_input("Monthly Education (₹)", min_value=0.0, value=0.0, step=500.0)
        miscellaneous = st.number_input("Monthly Miscellaneous (₹)", min_value=0.0, value=0.0, step=500.0)
    
    submit_button = st.form_submit_button("Analyze My Financial Stability")

# Function to generate synthetic financial data for clustering
def generate_synthetic_data(user_data_row):
    # Create a range of financial profiles for clustering
    np.random.seed(42)  # For reproducibility
    
    # Generate 100 synthetic data points around different financial profiles
    n_samples = 300
    
    # Low stability profiles
    low_stability = pd.DataFrame({
        "Savings_Rate": np.random.uniform(0, 0.1, n_samples // 3),
        "Debt_Rate": np.random.uniform(0.3, 0.8, n_samples // 3),
        "Expense_to_Income": np.random.uniform(0.7, 1.2, n_samples // 3),
        "Liquid_Term": np.random.uniform(0, 0.1, n_samples // 3)
    })
    
    # Moderate stability profiles
    moderate_stability = pd.DataFrame({
        "Savings_Rate": np.random.uniform(0.1, 0.2, n_samples // 3),
        "Debt_Rate": np.random.uniform(0.15, 0.3, n_samples // 3),
        "Expense_to_Income": np.random.uniform(0.5, 0.7, n_samples // 3),
        "Liquid_Term": np.random.uniform(0.1, 0.3, n_samples // 3)
    })
    
    # High stability profiles
    high_stability = pd.DataFrame({
        "Savings_Rate": np.random.uniform(0.2, 0.5, n_samples // 3),
        "Debt_Rate": np.random.uniform(0, 0.15, n_samples // 3),
        "Expense_to_Income": np.random.uniform(0.3, 0.5, n_samples // 3),
        "Liquid_Term": np.random.uniform(0.3, 1.0, n_samples // 3)
    })
    
    # Combine all profiles
    synthetic_data = pd.concat([low_stability, moderate_stability, high_stability])
    
    # Add user data to the synthetic dataset
    synthetic_data = pd.concat([synthetic_data, user_data_row[["Savings_Rate", "Debt_Rate", "Expense_to_Income", "Liquid_Term"]]])
    
    return synthetic_data

# Process when form is submitted
if submit_button:
    # Create a dataframe with the user input
    user_data = pd.DataFrame({
        "Income": [income],
        "Desired_Savings": [desired_savings],
        "Loan_Repayment": [loan_repayment],
        "Rent": [rent],
        "Groceries": [groceries],
        "Transport": [transport],
        "Eating_Out": [eating_out],
        "Entertainment": [entertainment],
        "Utilities": [utilities],
        "Healthcare": [healthcare],
        "Education": [education],
        "Miscellaneous": [miscellaneous]
    })
    
    # Feature Engineering
    user_data["Savings_Rate"] = user_data["Desired_Savings"] / user_data["Income"] if user_data["Income"].values[0] > 0 else 0
    user_data["Debt_Rate"] = user_data["Loan_Repayment"] / user_data["Income"] if user_data["Income"].values[0] > 0 else 0
    user_data["Expense_to_Income"] = (user_data["Rent"] + user_data["Groceries"] + 
                                    user_data["Transport"] + user_data["Eating_Out"] +
                                    user_data["Entertainment"] + user_data["Utilities"] + 
                                    user_data["Healthcare"] + user_data["Education"] + 
                                    user_data["Miscellaneous"]) / user_data["Income"] if user_data["Income"].values[0] > 0 else 0
    
    # Handle division by zero for Liquid_Term
    if income - desired_savings <= 0:
        user_data["Liquid_Term"] = 100  # High value to indicate infinite term
    else:
        user_data["Liquid_Term"] = user_data["Desired_Savings"] / (user_data["Income"] - user_data["Desired_Savings"])
    
    # Display calculated metrics
    st.subheader("Your Financial Metrics")
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("Savings Rate", f"{user_data['Savings_Rate'].values[0]:.1%}")
    with metrics_col2:
        st.metric("Debt Rate", f"{user_data['Debt_Rate'].values[0]:.1%}")
    with metrics_col3:
        st.metric("Expense to Income", f"{user_data['Expense_to_Income'].values[0]:.1%}")
    with metrics_col4:
        liquid_term = user_data['Liquid_Term'].values[0]
        if liquid_term >= 100:
            st.metric("Liquid Term", "∞")
        else:
            st.metric("Liquid Term", f"{liquid_term:.2f}")
    
    # Method 1: Rule-based classification for comparison
    savings_rate = user_data['Savings_Rate'].values[0]
    debt_rate = user_data['Debt_Rate'].values[0]
    expense_ratio = user_data['Expense_to_Income'].values[0]
    
    # Calculate rule-based score
    rule_score = 0
    
    # Savings Rate scoring
    if savings_rate >= 0.20:
        rule_score += 3  # High savings rate
    elif savings_rate >= 0.10:
        rule_score += 2  # Moderate savings rate
    elif savings_rate > 0:
        rule_score += 1  # Low savings rate
    
    # Debt Rate scoring (lower is better)
    if debt_rate <= 0.15:
        rule_score += 3  # Low debt rate
    elif debt_rate <= 0.30:
        rule_score += 2  # Moderate debt rate
    elif debt_rate <= 0.40:
        rule_score += 1  # High debt rate
    
    # Expense to Income scoring (lower is better)
    if expense_ratio <= 0.50:
        rule_score += 3  # Low expense ratio
    elif expense_ratio <= 0.70:
        rule_score += 2  # Moderate expense ratio
    elif expense_ratio <= 0.85:
        rule_score += 1  # High expense ratio
    
    # Determine rule-based stability category
    if rule_score >= 7:
        rule_stability = "High Stability"
        rule_color = "#28a745"  # Green
    elif rule_score >= 4:
        rule_stability = "Moderate Stability"
        rule_color = "#ffc107"  # Yellow
    else:
        rule_stability = "Low Stability"
        rule_color = "#dc3545"  # Red
    
    # Method 2: DBSCAN Clustering for ML-based classification
    # Generate synthetic data
    st.subheader("Machine Learning Classification")
    with st.expander("DBSCAN Clustering Process", expanded=True):
        st.markdown("""
        The following steps are performed to classify your financial stability using DBSCAN clustering:
        
        1. Generate synthetic financial profiles representing different stability levels
        2. Scale the data using RobustScaler to normalize the features
        3. Apply DBSCAN clustering to identify dense regions of similar financial profiles
        4. Determine which cluster your financial profile belongs to
        5. Label the clusters based on their financial characteristics
        """)
        
        # Generate synthetic data for clustering including user data
        combined_data = generate_synthetic_data(user_data)
        st.write(f"Generated {len(combined_data)} financial profiles for clustering")
        
        # Scale the data
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(combined_data)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_data)
        
        # Add cluster labels to the data
        combined_data['Cluster'] = clusters
        
        # Determine user's cluster
        user_cluster = clusters[-1]
        
        # Visualize the clusters with PCA
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)
        
        # Create a DataFrame for visualization
        viz_df = pd.DataFrame({
            'PCA1': reduced_data[:, 0],
            'PCA2': reduced_data[:, 1],
            'Cluster': clusters
        })
        
        # Highlight user data point
        viz_df['IsUser'] = [i == len(viz_df) - 1 for i in range(len(viz_df))]
        
        # Plot clusters
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each cluster
        for cluster in sorted(viz_df['Cluster'].unique()):
            if cluster == -1:
                # Noise points
                cluster_points = viz_df[viz_df['Cluster'] == cluster]
                plt.scatter(cluster_points['PCA1'], cluster_points['PCA2'], 
                           label=f'Noise', color='gray', alpha=0.5, s=30)
            else:
                # Cluster points
                cluster_points = viz_df[(viz_df['Cluster'] == cluster) & (~viz_df['IsUser'])]
                plt.scatter(cluster_points['PCA1'], cluster_points['PCA2'], 
                           label=f'Cluster {cluster}', alpha=0.7, s=30)
        
        # Highlight user data point
        user_point = viz_df[viz_df['IsUser']]
        plt.scatter(user_point['PCA1'], user_point['PCA2'], 
                   color='red', marker='*', s=300, label='Your Profile')
        
        plt.title('Financial Profile Clustering (PCA Visualization)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        
        # Display the plot
        st.pyplot(fig)
        
        # Calculate cluster means
        cluster_means = combined_data.groupby('Cluster').mean()
        
        # Display cluster characteristics
        st.subheader("Cluster Characteristics")
        st.write(cluster_means)
        
        # Determine the stability level of each cluster
        # Higher savings rate, lower debt rate, and lower expense ratio = better stability
        if user_cluster == -1:
            # Noise point
            ml_stability = rule_stability  # Fall back to rule-based
            ml_color = rule_color
            ml_description = "Your financial profile is unique and doesn't fit standard patterns. Using rule-based assessment instead."
        else:
            # Get characteristics of user's cluster
            cluster_sr = cluster_means.loc[user_cluster, 'Savings_Rate']
            cluster_dr = cluster_means.loc[user_cluster, 'Debt_Rate']
            cluster_er = cluster_means.loc[user_cluster, 'Expense_to_Income']
            
            # Score the cluster
            cluster_score = 0
            
            # Savings Rate scoring
            if cluster_sr >= 0.20:
                cluster_score += 3  # High savings rate
            elif cluster_sr >= 0.10:
                cluster_score += 2  # Moderate savings rate
            elif cluster_sr > 0:
                cluster_score += 1  # Low savings rate
            
            # Debt Rate scoring (lower is better)
            if cluster_dr <= 0.15:
                cluster_score += 3  # Low debt rate
            elif cluster_dr <= 0.30:
                cluster_score += 2  # Moderate debt rate
            elif cluster_dr <= 0.40:
                cluster_score += 1  # High debt rate
            
            # Expense to Income scoring (lower is better)
            if cluster_er <= 0.50:
                cluster_score += 3  # Low expense ratio
            elif cluster_er <= 0.70:
                cluster_score += 2  # Moderate expense ratio
            elif cluster_er <= 0.85:
                cluster_score += 1  # High expense ratio
            
            # Determine ML-based stability category
            if cluster_score >= 7:
                ml_stability = "High Stability"
                ml_color = "#28a745"  # Green
                ml_description = "The DBSCAN algorithm has identified your financial profile as part of a cluster with excellent financial stability characteristics."
            elif cluster_score >= 4:
                ml_stability = "Moderate Stability"
                ml_color = "#ffc107"  # Yellow
                ml_description = "The DBSCAN algorithm has identified your financial profile as part of a cluster with moderate financial stability characteristics."
            else:
                ml_stability = "Low Stability"
                ml_color = "#dc3545"  # Red
                ml_description = "The DBSCAN algorithm has identified your financial profile as part of a cluster with concerning financial stability characteristics."
    
    # Compare rule-based and ML-based classifications
    st.subheader("Classification Results")
    
    comparison_col1, comparison_col2 = st.columns(2)
    
    with comparison_col1:
        st.markdown("### Rule-Based Classification")
        st.markdown(f"""
        <div style="background-color: {rule_color}; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
            <h2 style="color: white; margin: 0;">{rule_stability}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with comparison_col2:
        st.markdown("### Machine Learning Classification")
        st.markdown(f"""
        <div style="background-color: {ml_color}; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
            <h2 style="color: white; margin: 0;">{ml_stability}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Display assessment details for ML-based classification
    st.subheader("ML Assessment Details")
    st.write(ml_description)
    
    if user_cluster == -1:
        st.warning("Your financial profile was classified as an outlier by DBSCAN. This suggests that your financial situation is unique and doesn't fit into the common patterns. This is not necessarily bad - it just means your profile is different from the typical cases in our data.")
    
    # Detailed breakdown
    st.subheader("Breakdown of Your Financial Health")
    
    breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
    
    with breakdown_col1:
        st.markdown("### Savings")
        if savings_rate >= 0.20:
            st.markdown("✅ **Excellent**: Your savings rate is above 20%")
        elif savings_rate >= 0.10:
            st.markdown("✓ **Good**: Your savings rate is between 10-20%")
        elif savings_rate > 0:
            st.markdown("⚠️ **Needs Improvement**: Your savings rate is below 10%")
        else:
            st.markdown("❌ **Critical**: You are not saving any money")
    
    with breakdown_col2:
        st.markdown("### Debt")
        if debt_rate <= 0.15:
            st.markdown("✅ **Excellent**: Your debt burden is low")
        elif debt_rate <= 0.30:
            st.markdown("✓ **Good**: Your debt is at a manageable level")
        elif debt_rate <= 0.40:
            st.markdown("⚠️ **Caution**: Your debt burden is high")
        else:
            st.markdown("❌ **Critical**: Your debt burden is very high")
    
    with breakdown_col3:
        st.markdown("### Expenses")
        if expense_ratio <= 0.50:
            st.markdown("✅ **Excellent**: Your expenses are well controlled")
        elif expense_ratio <= 0.70:
            st.markdown("✓ **Good**: Your expenses are reasonable")
        elif expense_ratio <= 0.85:
            st.markdown("⚠️ **Caution**: Your expenses are high relative to income")
        else:
            st.markdown("❌ **Critical**: Your expenses are too high")
    
    # Use the ML stability for recommendations
    stability = ml_stability
    
    # Recommendations
    st.subheader("Recommendations")
    
    # Different recommendations based on stability category
    if stability == "High Stability":
        st.markdown("""
        ### 🌟 Congratulations on your excellent financial management! 🌟
        
        - 🚀 **Keep up the great work!** You're on the path to long-term financial freedom
        - 💎 **Consider investing** for even greater long-term growth
        - 🏆 **Share your knowledge** with others who might benefit from your financial discipline
        - 🎯 **Set new financial goals** to stay motivated and continue your success
        - 🛡️ **Review your insurance coverage** to ensure your financial security is protected
        """)
    else:
        if savings_rate < 0.10:
            st.markdown("- 💡 **Increase savings**: Try to save at least 10% of your income")
        
        if debt_rate > 0.30:
            st.markdown("- 💡 **Reduce debt**: Focus on paying down high-interest debt")
        
        if expense_ratio > 0.70:
            st.markdown("- 💡 **Review expenses**: Look for areas where you can reduce spending")
            
            # Identify highest expense categories
            expenses = {
                "Rent/Mortgage": rent,
                "Groceries": groceries,
                "Transport": transport,
                "Eating Out": eating_out,
                "Entertainment": entertainment,
                "Utilities": utilities,
                "Healthcare": healthcare,
                "Education": education,
                "Miscellaneous": miscellaneous
            }
            
            top_expenses = sorted(expenses.items(), key=lambda x: x[1], reverse=True)[:3]
            st.markdown("- 💡 **Highest expense categories**:")
            for category, amount in top_expenses:
                if amount > 0:  # Only show categories with expenses
                    if income > 0:
                        percentage = amount / income * 100
                        st.markdown(f"  - {category}: ₹{amount:.2f} ({percentage:.1f}% of income)")
                    else:
                        st.markdown(f"  - {category}: ₹{amount:.2f}")

# Information section at the bottom
st.markdown("""
---
### About Fiscallytic - Machine Learning Financial Classification

This app uses the DBSCAN clustering algorithm to classify financial stability based on key financial ratios:

1. **Savings Rate** = Monthly Savings ÷ Monthly Income
2. **Debt Rate** = Monthly Loan Repayments ÷ Monthly Income
3. **Expense to Income Ratio** = Total Monthly Expenses ÷ Monthly Income
4. **Liquid Term** = Monthly Savings ÷ (Monthly Income - Monthly Savings)

The DBSCAN algorithm identifies clusters of similar financial profiles and determines which cluster your financial profile belongs to, providing a data-driven assessment of your financial stability.

For a more personalized financial strategy, consider consulting with a financial advisor.
""")
