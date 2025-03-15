# 💰 Fiscallytic

## 🌟 Overview
This application uses machine learning to classify financial stability based on income, expenses, savings, and debt information. It provides users with a personalized stability assessment and recommendations for improving their financial health.

✨ **Try it now:** [Fiscallytic App](https://project-w23cojwxbalxahfwafplcw.streamlit.app/)

## ✅ Features
- **🖥️ Interactive Dashboard**: Easy-to-use interface for entering financial data
- **⚡ Real-time Analysis**: Instant feedback on financial stability
- **🎯 Personalized Recommendations**: Tailored suggestions based on individual financial situations
- **🎨 Visual Indicators**: Color-coded stability ratings and detailed breakdowns
- **📊 Key Financial Metrics**: Calculation and display of important financial ratios

## 🔧 Technical Details
The application uses the following technologies:
- **Streamlit**: For the interactive web interface
- **Pandas**: For data manipulation
- **NumPy**: For numerical operations
- **Scikit-learn**: For the machine learning components
  - DBSCAN algorithm for clustering financial profiles
  - RobustScaler for feature normalization

## 📈 Financial Metrics
The application calculates the following key financial metrics:
1. **Savings Rate** = Monthly Savings ÷ Monthly Income
2. **Debt Rate** = Monthly Loan Repayments ÷ Monthly Income
3. **Expense to Income Ratio** = Total Monthly Expenses ÷ Monthly Income
4. **Liquid Term** = Monthly Savings ÷ (Monthly Income - Monthly Savings)

## 🧮 Classification Method
The application uses a dual approach to financial stability classification:
1. **Rules-based scoring system**: Evaluates savings rate, debt rate, and expense ratio against predefined thresholds
2. **Machine learning clustering** (DBSCAN): Identifies patterns in financial data to group similar financial profiles

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/fiscallytic.git
cd fiscallytic
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🔍 Usage
1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and go to:
```
http://localhost:8501
```

3. Enter your financial information in the form and click "Analyze My Financial Stability"

## ⚙️ Customization
The classification thresholds can be adjusted in the code to match specific financial contexts or regional economic conditions.

## ⚠️ Disclaimer
This application provides general financial assessments and should not be considered as professional financial advice. For personalized financial guidance, please consult with a qualified financial advisor.
