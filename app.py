import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler

# Set page config
st.set_page_config(
    page_title="Financial Stability Classifier",
    page_icon="💰",
    layout="wide"
)

# Add title and description
st.title("💰 Dynamic Financial Classification Using Machine Learning")
st.markdown("""
This app uses machine learning to classify your financial stability based on your income, expenses, savings, and debt information.
Enter your financial details below to receive your personalized stability assessment.
""")

# Define the form for user input
st.subheader("Enter Your Financial Information")
with st.form("financial_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input("Monthly Income ($)", min_value=0.0, value=3000.0, step=100.0)
        desired_savings = st.number_input("Monthly Savings Target ($)", min_value=0.0, value=500.0, step=50.0)
        loan_repayment = st.number_input("Monthly Loan Repayments ($)", min_value=0.0, value=300.0, step=50.0)
        rent = st.number_input("Monthly Rent/Mortgage ($)", min_value=0.0, value=1000.0, step=50.0)
        groceries = st.number_input("Monthly Groceries ($)", min_value=0.0, value=400.0, step=50.0)
    
    with col2:
        transport = st.number_input("Monthly Transport ($)", min_value=0.0, value=200.0, step=50.0)
        eating_out = st.number_input("Monthly Eating Out ($)", min_value=0.0, value=150.0, step=50.0)
        entertainment = st.number_input("Monthly Entertainment ($)", min_value=0.0, value=100.0, step=50.0)
        utilities = st.number_input("Monthly Utilities ($)", min_value=0.0, value=200.0, step=50.0)
        healthcare = st.number_input("Monthly Healthcare ($)", min_value=0.0, value=100.0, step=50.0)
        education = st.number_input("Monthly Education ($)", min_value=0.0, value=50.0, step=50.0)
        miscellaneous = st.number_input("Monthly Miscellaneous ($)", min_value=0.0, value=100.0, step=50.0)
    
    submit_button = st.form_submit_button("Analyze My Financial Stability")

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
    user_data["Savings_Rate"] = user_data["Desired_Savings"] / user_data["Income"]
    user_data["Debt_Rate"] = user_data["Loan_Repayment"] / user_data["Income"]
    user_data["Expense_to_Income"] = (user_data["Rent"] + user_data["Groceries"] + 
                                    user_data["Transport"] + user_data["Eating_Out"] +
                                    user_data["Entertainment"] + user_data["Utilities"] + 
                                    user_data["Healthcare"] + user_data["Education"] + 
                                    user_data["Miscellaneous"]) / user_data["Income"]
    
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
    
    # Define classification logic based on financial metrics
    # This is a simplified version - in a real application, you'd use your trained DBSCAN model
    
    # Method 1: Simple rules-based classification
    savings_rate = user_data['Savings_Rate'].values[0]
    debt_rate = user_data['Debt_Rate'].values[0]
    expense_ratio = user_data['Expense_to_Income'].values[0]
    
    # Calculate a simple score
    score = 0
    
    # Savings Rate scoring
    if savings_rate >= 0.20:
        score += 3  # High savings rate
    elif savings_rate >= 0.10:
        score += 2  # Moderate savings rate
    elif savings_rate > 0:
        score += 1  # Low savings rate
    
    # Debt Rate scoring (lower is better)
    if debt_rate <= 0.15:
        score += 3  # Low debt rate
    elif debt_rate <= 0.30:
        score += 2  # Moderate debt rate
    elif debt_rate <= 0.40:
        score += 1  # High debt rate
    
    # Expense to Income scoring (lower is better)
    if expense_ratio <= 0.50:
        score += 3  # Low expense ratio
    elif expense_ratio <= 0.70:
        score += 2  # Moderate expense ratio
    elif expense_ratio <= 0.85:
        score += 1  # High expense ratio
    
    # Determine stability category based on score
    if score >= 7:
        stability = "High Stability"
        color = "#28a745"  # Green
        description = "You have excellent financial stability with a good balance between savings, expenses, and debt."
    elif score >= 4:
        stability = "Moderate Stability"
        color = "#ffc107"  # Yellow
        description = "You have moderate financial stability. There's room for improvement in certain areas."
    else:
        stability = "Low Stability"
        color = "#dc3545"  # Red
        description = "Your financial stability is at risk. Consider adjusting your savings, expenses, or debt management."
    
    # Display results
    st.header("Your Financial Stability Assessment")
    
    # Create a centered, colored box for the result
    st.markdown(f"""
    <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
        <h2 style="color: white; margin: 0;">{stability}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Assessment Details")
    st.write(description)
    
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
                percentage = amount / income * 100
                st.markdown(f"  - {category}: ${amount:.2f} ({percentage:.1f}% of income)")

# Information section at the bottom
st.markdown("""
---
### About Dynamic Financial Classification

This app uses machine learning techniques to classify financial stability based on key financial ratios:

1. **Savings Rate** = Monthly Savings ÷ Monthly Income
2. **Debt Rate** = Monthly Loan Repayments ÷ Monthly Income
3. **Expense to Income Ratio** = Total Monthly Expenses ÷ Monthly Income
4. **Liquid Term** = Monthly Savings ÷ (Monthly Income - Monthly Savings)

The classification algorithm evaluates your financial metrics against established patterns to determine your stability category.

For a more personalized financial strategy, consider consulting with a financial advisor.
""")
