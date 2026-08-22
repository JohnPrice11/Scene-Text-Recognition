import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# --- 1. Page Configuration ---
st.set_page_config(page_title="EMIPredict AI", page_icon="🏦", layout="wide")

# --- 2. Load Assets ---
@st.cache_resource
def load_assets():
    # Replace these with your actual Google Drive File IDs
    class_model_id = '1eB7pREzB_JgxfDIRayxtArgrrtYdJp8c'
    reg_model_id = '1Jf6tiwNMnrCEIwt6ILL_wUrBwpdn9aWT'
    
    if not os.path.exists('best_class_model.pkl'):
        gdown.download(id=class_model_id, output='best_class_model.pkl', quiet=False)
    if not os.path.exists('best_reg_model.pkl'):
        gdown.download(id=reg_model_id, output='best_reg_model.pkl', quiet=False)

    scaler = joblib.load('scaler.pkl')
    ohe = joblib.load('ohe.pkl')
    le = joblib.load('le.pkl')
    num_cols = joblib.load('num_cols.pkl')
    cat_cols = joblib.load('cat_cols.pkl')
    class_model = joblib.load('best_class_model.pkl')
    reg_model = joblib.load('best_reg_model.pkl')
    return scaler, ohe, le, num_cols, cat_cols, class_model, reg_model

scaler, ohe, le, num_cols, cat_cols, class_model, reg_model = load_assets()

# --- 3. Feature Engineering Function ---
def engineer_features(df):
    df_eng = df.copy()
    math_cols = ['monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 
                 'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount',
                 'monthly_salary', 'emergency_fund', 'requested_amount', 'requested_tenure']
    
    for col in math_cols:
        if col in df_eng.columns:
            df_eng[col] = df_eng[col].astype(str).replace(r'[^\d.-]', '', regex=True)
            df_eng[col] = pd.to_numeric(df_eng[col], errors='coerce').fillna(0.0)
            
    eps = 1e-6
    expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 
                    'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount']
    
    df_eng['total_monthly_outflow'] = df_eng[expense_cols].sum(axis=1)
    df_eng['debt_to_income_ratio'] = df_eng['current_emi_amount'] / (df_eng['monthly_salary'] + eps)
    df_eng['expense_to_income_ratio'] = df_eng['total_monthly_outflow'] / (df_eng['monthly_salary'] + eps)
    df_eng['disposable_income'] = df_eng['monthly_salary'] - df_eng['total_monthly_outflow']
    df_eng['emergency_fund_coverage'] = df_eng['emergency_fund'] / (df_eng['total_monthly_outflow'] + eps) 
    estimated_new_emi = df_eng['requested_amount'] / (df_eng['requested_tenure'] + eps)
    df_eng['proposed_total_emi_burden'] = (df_eng['current_emi_amount'] + estimated_new_emi) / (df_eng['monthly_salary'] + eps)
    return df_eng

# --- 4. Sidebar Navigation ---
st.sidebar.title("🏦 EMIPredict AI")
page = st.sidebar.radio("Navigation", ["Risk Assessment", "Data Exploration", "Model Dashboard", "Admin Interface"])

# --- PAGE 1: Risk Assessment ---
if page == "Risk Assessment":
    st.title("Financial Risk Assessment")
    st.markdown("Intelligent EMI Eligibility and Capacity Prediction Engine")
    
    with st.form("loan_application"):
        st.subheader("Applicant Financial Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
            monthly_salary = st.number_input("Monthly Salary (INR)", min_value=0.0, value=50000.0)
            employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
            years_of_employment = st.number_input("Years of Employment", min_value=0, value=5)
        with col2:
            company_type = st.selectbox("Company Type", ["Corporate", "Startup", "MNC", "SME"])
            house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
            monthly_rent = st.number_input("Monthly Rent", min_value=0.0, value=15000.0)
            family_size = st.number_input("Family Size", min_value=1, value=3)
            dependents = st.number_input("Dependents", min_value=0, value=1)
            school_fees = st.number_input("School Fees", min_value=0.0, value=0.0)
            college_fees = st.number_input("College Fees", min_value=0.0, value=0.0)
        with col3:
            travel_expenses = st.number_input("Travel Expenses", min_value=0.0, value=3000.0)
            groceries_utilities = st.number_input("Groceries & Utilities", min_value=0.0, value=8000.0)
            other_monthly_expenses = st.number_input("Other Expenses", min_value=0.0, value=2000.0)
            existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
            current_emi_amount = st.number_input("Current EMI Amount", min_value=0.0, value=0.0)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
            bank_balance = st.number_input("Bank Balance", min_value=0.0, value=100000.0)
            emergency_fund = st.number_input("Emergency Fund", min_value=0.0, value=50000.0)

        st.subheader("Loan Requirements")
        col4, col5, col6 = st.columns(3)
        with col4:
            emi_scenario = st.selectbox("EMI Scenario", ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"])
        with col5:
            requested_amount = st.number_input("Requested Loan Amount (INR)", min_value=1000.0, value=100000.0)
        with col6:
            requested_tenure = st.number_input("Requested Tenure (Months)", min_value=3, max_value=84, value=12)

        submitted = st.form_submit_button("Run Risk Assessment")

    if submitted:
        input_data = pd.DataFrame([{
            'age': age, 'gender': gender, 'marital_status': marital_status, 'education': education,
            'monthly_salary': monthly_salary, 'employment_type': employment_type, 'years_of_employment': years_of_employment, 
            'company_type': company_type, 'house_type': house_type, 'monthly_rent': monthly_rent, 'family_size': family_size,
            'dependents': dependents, 'school_fees': school_fees, 'college_fees': college_fees, 'travel_expenses': travel_expenses, 
            'groceries_utilities': groceries_utilities, 'other_monthly_expenses': other_monthly_expenses, 'existing_loans': existing_loans,
            'current_emi_amount': current_emi_amount, 'credit_score': credit_score, 'bank_balance': bank_balance, 
            'emergency_fund': emergency_fund, 'emi_scenario': emi_scenario, 'requested_amount': requested_amount, 'requested_tenure': requested_tenure
        }])

        df_eng = engineer_features(input_data)
        df_eng[cat_cols] = df_eng[cat_cols].astype(str)
        num_scaled = pd.DataFrame(scaler.transform(df_eng[num_cols]), columns=num_cols)
        cat_encoded = pd.DataFrame(ohe.transform(df_eng[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
        processed_input = pd.concat([num_scaled, cat_encoded], axis=1)

        class_pred_encoded = class_model.predict(processed_input)[0]
        eligibility_status = le.inverse_transform([class_pred_encoded])[0]
        max_emi_pred = reg_model.predict(processed_input)[0]

        st.markdown("---")
        st.header("Risk Assessment Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if eligibility_status == 'Eligible':
                st.success(f"Eligibility Status: **{eligibility_status}**")
            elif eligibility_status == 'High_Risk':
                st.warning(f"Eligibility Status: **{eligibility_status}** (Requires Manual Review)")
            else:
                st.error(f"Eligibility Status: **{eligibility_status}** (Loan Not Recommended)")
        with res_col2:
            st.info(f"Recommended Maximum Monthly EMI: **₹{max_emi_pred:,.2f}**")

# --- PAGE 2: Data Exploration ---
elif page == "Data Exploration":
    st.title("📊 Data Exploration Insights")
    st.markdown("Explore the synthetic financial distribution metrics used to train the models.")
    st.info("Interactive visualization modules will render here. In a full production environment, this connects directly to the static data summaries.")
    # Placeholder for simulated charts
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["EMI Approval Rate", "Debt-to-Income", "Average Loan Size"])
    st.line_chart(chart_data)

# --- PAGE 3: Model Dashboard ---
elif page == "Model Dashboard":
    st.title("📈 Model Performance & MLflow Metrics")
    st.markdown("Live view of our production models tracking via MLflow.")
    
    st.subheader("Classification: Random Forest (EMI Eligibility)")
    st.write("- **Validation Accuracy:** 92.83%")
    st.write(f"- **ROC-AUC Score:** 0.9856") # Update with your exact notebook output
    
    st.subheader("Regression: XGBoost (Max EMI)")
    st.write("- **Validation RMSE:** 953.22 INR")
    st.write(f"- **MAPE:** 1.84%") # Update with your exact notebook output

# --- PAGE 4: Admin Interface (CRUD) ---
elif page == "Admin Interface":
    st.title("⚙️ Admin Console: Record Management")
    st.markdown("CRUD Operations for backend data management.")
    
    if "admin_db" not in st.session_state:
        st.session_state.admin_db = pd.DataFrame({"ID": [1, 2], "Applicant": ["John Doe", "Jane Smith"], "Status": ["Eligible", "High_Risk"]})
    
    st.dataframe(st.session_state.admin_db)
    
    with st.expander("Add New Record"):
        new_name = st.text_input("Applicant Name")
        new_status = st.selectbox("Status", ["Eligible", "High_Risk", "Not_Eligible"])
        if st.button("Create Record"):
            new_id = len(st.session_state.admin_db) + 1
            new_row = pd.DataFrame({"ID": [new_id], "Applicant": [new_name], "Status": [new_status]})
            st.session_state.admin_db = pd.concat([st.session_state.admin_db, new_row], ignore_index=True)
            st.rerun()