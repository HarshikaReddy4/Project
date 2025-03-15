import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler

# Set page config
st.set_page_config(
    page_title="Fiscallytic ML",
    page_icon="💰",
    layout="wide"
)

# Add title and description
st.title("💰 Fiscallytic: ML-Based Financial Stability Analysis")
st.markdown("""
This app uses machine learning (DBSCAN clustering) to analyze your financial stability based on your income, 
expenses, savings, and debt information.
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
    
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm 
    that groups together points that are closely packed together, marking as outliers points 
    that lie alone in low-density regions.
    
    - **Epsilon**: The maximum distance between two samples for one to be considered as in the 
      neighborhood of the other
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
    
    # Generate synthetic data points around different financial profiles
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
    
    # DBSCAN Clustering for ML-based classification
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
        
        # Calculate cluster means
        cluster_means = combined_data.groupby('Cluster').mean()
        
        # Display cluster characteristics
        st.subheader("Cluster Characteristics")
        st.write(cluster_means)
        
        # Define function to classify clusters
        def classify_cluster(sr, dr, ei):
            """Classify a cluster based on its mean values"""
            # Calculate a score based on financial metrics
            score = 0
            
            # Savings Rate scoring
            if sr >= 0.20:
                score += 3  # High savings rate
            elif sr >= 0.10:
                score += 2  # Moderate savings rate
            elif sr > 0:
                score += 1  # Low savings rate
            
            # Debt Rate scoring (lower is better)
            if dr <= 0.15:
                score += 3  # Low debt rate
            elif dr <= 0.30:
                score += 2  # Moderate debt rate
            elif dr <= 0.40:
                score += 1  # High debt rate
            
            # Expense to Income scoring (lower is better)
            if ei <= 0.50:
                score += 3  # Low expense ratio
            elif ei <= 0.70:
                score += 2  # Moderate expense ratio
            elif ei <= 0.85:
                score += 1  # High expense ratio
            
            # Classify based on score
            if score >= 7:
                return "High Stability", "#28a745"  # Green
            elif score >= 4:
                return "Moderate Stability", "#ffc107"  # Yellow
            else:
                return "Low Stability", "#dc3545"  # Red
        
        # Create cluster classifications for all clusters
        cluster_classifications = {}
        for cluster in cluster_means.index:
            if cluster != -1:  # Skip noise cluster
                sr = cluster_means.loc[cluster, 'Savings_Rate']
                dr = cluster_means.loc[cluster, 'Debt_Rate']
                ei = cluster_means.loc[cluster, 'Expense_to_Income']
                stability, color = classify_cluster(sr, dr, ei)
                cluster_classifications[cluster] = (stability, color)
        
        # Display cluster classifications
        st.subheader("Cluster Stability Classifications")
        for cluster, (stability, _) in cluster_classifications.items():
            st.write(f"Cluster {cluster}: {stability}")
        
        # Determine user's financial stability
        if user_cluster == -1:
            # User is an outlier - analyze individually
            savings_rate = user_data['Savings_Rate'].values[0]
            debt_rate = user_data['Debt_Rate'].values[0]
            expense_ratio = user_data['Expense_to_Income'].values[0]
            stability, color = classify_cluster(savings_rate, debt_rate, expense_ratio)
            st.warning("Your profile was classified as an outlier. Individual assessment applied.")
        else:
            # User belongs to a cluster
            stability, color = cluster_classifications.get(user_cluster, ("Unknown", "#6c757d"))
        
        # Description map
        stability_descriptions = {
            "High Stability": "Your financial profile shows excellent financial management with strong savings, low debt, and controlled expenses.",
            "Moderate Stability": "Your financial profile shows reasonable financial management with some areas for improvement.",
            "Low Stability": "Your financial profile indicates concerns in multiple areas that need attention.",
            "Unknown": "Unable to determine stability level."
        }
        
        # Get description
        description = stability_descriptions.get(stability, "")
    
    # Display the results
    st.subheader("Financial Stability Assessment")
    st.markdown(f"""
    <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
        <h2 style="color: white; margin: 0;">{stability}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display assessment details
    st.subheader("Assessment Details")
    st.write(description)
    
    # Detailed breakdown
    st.subheader("Breakdown of Your Financial Health")
    
    breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
    
    with breakdown_col1:
        st.markdown("### Savings")
        savings_rate = user_data['Savings_Rate'].values[0]
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
        debt_rate = user_data['Debt_Rate'].values[0]
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
        expense_ratio = user_data['Expense_to_Income'].values[0]
        if expense_ratio <= 0.50:
            st.markdown("✅ **Excellent**: Your expenses are well controlled")
        elif expense_ratio <= 0.70:
            st.markdown("✓ **Good**: Your expenses are reasonable")
        elif expense_ratio <= 0.85:
            st.markdown("⚠️ **Caution**: Your expenses are high relative to income")
        else:
            st.markdown("❌ **Critical**: Your expenses are too high")
    
    # Recommendations based on stability category
    st.subheader("Recommendations")
    
    if stability == "High Stability":
        st.markdown("""
        ### 🌟 Congratulations on your excellent financial management! 🌟
        
        - 🚀 **Keep up the great work!** You're on the path to long-term financial freedom
        - 💎 **Consider investing** for even greater long-term growth
        - 🏆 **Share your knowledge** with others who might benefit from your financial discipline
        - 🎯 **Set new financial goals** to stay motivated and continue your success
        - 🛡️ **Review your insurance coverage** to ensure your financial security is protected
        """)
    elif stability == "Moderate Stability":
        st.markdown("""
        ### 🔍 You have a solid foundation, but there's room for improvement
        
        - 📊 **Track your spending** more closely to identify potential savings
        - 💰 **Increase your savings rate** by 3-5% for greater long-term security
        - 💳 **Review your debt strategy** to potentially accelerate payoff of high-interest debt
        - 📝 **Create a monthly budget** to better manage your cash flow
        - 🛒 **Look for small optimizations** in your recurring expenses
        """)
    else:
        st.markdown("""
        ### 🚨 Your financial situation needs immediate attention
        
        - 📉 **Create an emergency budget** to reduce expenses immediately
        - 🛑 **Pause non-essential spending** until your situation stabilizes
        - 💸 **Look for additional income sources** to improve your cash flow
        - 🔄 **Consolidate high-interest debt** if possible to reduce monthly payments
        - 📞 **Consider financial counseling** for personalized guidance
        """)
        
    # Specific targeted recommendations based on metrics
    st.markdown("### Targeted Recommendations")
    
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

The DBSCAN algorithm identifies clusters of similar financial profiles and determines which cluster your 
financial profile belongs to, providing a data-driven assessment of your financial stability.

For a more personalized financial strategy, consider consulting with a financial advisor.
""")
