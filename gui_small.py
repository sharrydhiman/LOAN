import streamlit as st
import joblib
from PIL import Image
import numpy as np  # Add this import for NumPy


# Load the loan prediction model
model = joblib.load("loan_prediction_model.joblib")

def prediction(Current_Loan_Amount, Term, Credit_Score, Annual_Income, Years_in_current_job, Home_Ownership, Purpose, Monthly_Debt, Years_of_Credit_History, Months_since_last_delinquent, Number_of_Open_Accounts, Number_of_Credit_Problems, Current_Credit_Balance, Maximum_Open_Credit, Bankruptcies, Tax_Liens):
    try:
        input_data = np.array([[Current_Loan_Amount, Term, Credit_Score, Annual_Income, Years_in_current_job, Home_Ownership, Purpose, Monthly_Debt, Years_of_Credit_History, Months_since_last_delinquent, Number_of_Open_Accounts, Number_of_Credit_Problems, Current_Credit_Balance, Maximum_Open_Credit, Bankruptcies, Tax_Liens]])
        prediction = model.predict(input_data)
        print(prediction)
        if prediction[0] == '1':
            pred = 'Approve'
        else:
            pred = 'Reject'
        return pred
    except Exception as e:
        return f"Prediction Error: {str(e)}"
mapping = {
    '10+ years': 10,
    '9 years': 9,
    '8 years': 8,
    '7 years': 7,
    '6 years': 6,
    '5 years': 5,
    '4 years': 4,
    '3 years': 3,
    '2 years': 2,
    '1 years': 1,
    '< 1 years': 0
    }
mapping4 = {
    'Own Home': 1,
    'Home Mortgage': 0,
    'Rent':'2',
    'HaveMortgage':3}
mapping5 = {
    'Debt Consolidation': 1,
    'Buy House': 2,
    'other': 3,
    'Take a Trip': 4,
    'Home Improvements':5,
    'Other':3,
    'Buy a Car':6,
    'Medical Bills':7,
    'wedding':8,
    'Business Loan':9,
    'small_business':10,
    'major_purchase':11,
    'vacation':4,
    'Educational Expenses':12,
    'moving':13,
    'renewable_energy':14}
def main():
    img1 = Image.open('background.jpg')
    img1 = img1.resize((2500,2500))
    st.image(img1,use_column_width=True)
    st.title("Bank Loan Prediction using Machine Learning")
    st.header('Enter Customer Information:')
    Current_Loan_Amount= st.number_input('Current Loan Amount', min_value=0)
    Term = st.selectbox('Term', ['Short', 'Long'])
    if Term == 'Short':
        Term = 1
    else:
        Term = 0

    Credit_Score= st.number_input('Credit Score', min_value=0)
    Annual_Income = st.number_input('Annual Income', min_value=0)
    Years_in_current_job = st.selectbox('Years in Current Job', ['< 1 years', '1 years', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'])
    Years_in_current_job=mapping[Years_in_current_job]
    Home_Ownership = st.selectbox('Home Ownership', ['Own Home', 'Rent', 'Mortgage'])
    Home_Ownership=mapping4[Home_Ownership]
    Purpose = st.selectbox('Purpose', ['Debt Consolidation', 'Buy House', 'Home Improvements', 'Other', 'Buy a Car', 'Medical Bills', 'wedding', 'Business Loan', 'small_business', 'major_purchase', 'vacation', 'Educational Expenses', 'moving', 'renewable_energy'])
    Purpose=mapping5[Purpose]
    Monthly_Debt = st.number_input('Monthly Debt', min_value=0)
    Years_of_Credit_History = st.number_input('Years of Credit History', min_value=0)
    Months_since_last_delinquent = st.number_input('Months Since Last Delinquent', min_value=0)
    Number_of_Open_Accounts = st.number_input('Number of Open Accounts', min_value=0)
    Number_of_Credit_Problems = st.number_input('Number of Credit Problems', min_value=0)
    Current_Credit_Balance = st.number_input('Current Credit Balance', min_value=0)
    Maximum_Open_Credit = st.number_input('Maximum Open Credit', min_value=0)
    Bankruptcies = st.number_input('Bankruptcies', min_value=0)
    Tax_Liens = st.number_input('Tax Liens', min_value=0)

    if st.button("Predict"):
        result = prediction(Current_Loan_Amount, Term, Credit_Score, Annual_Income, Years_in_current_job, Home_Ownership, Purpose, Monthly_Debt, Years_of_Credit_History, Months_since_last_delinquent, Number_of_Open_Accounts, Number_of_Credit_Problems, Current_Credit_Balance, Maximum_Open_Credit, Bankruptcies, Tax_Liens)
        st.success('Loan Status: {}'.format(result))

if __name__ == '__main__':
    main()
