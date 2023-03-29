import streamlit as st
import numpy as np
import pandas as pd
from standardize import standard
import joblib
model = joblib.load("model.sav")
def main():
    st.title("Prediction of customer churn 💳")
    st.markdown(
        """
    This is a streamlit app that was constructed to predict churn for credit card customers.  
    """
    )
    st.text("")
    st.text("")
    st.text("")
    st.sidebar.title("Type of Prediction")
    st.sidebar.markdown(
        """
    Please kindly indicate if you would like to predict customer churn for single or multiple customers.
    """
    )
    st.sidebar.text("")
    st.sidebar.text("")
    add_selectbox = st.sidebar.selectbox("Single Customer or Multiple Customers?", ("Single", "Multiple"))
    if add_selectbox == "Single":
        total_trans_amt = st.number_input("Total Transaction Amount(Last 12 months):", min_value = 0)
        total_trans_ct = st.number_input("Total Transaction Count(Last 12 months):", min_value = 0)
        total_amt_chng_q4_q1 = st.number_input("Change in Transaction Amount (Q4 over Q1):", min_value = 0.0, step=0.001, format="%0.3f")
        total_revolving_bal = st.number_input("Total Revolving Balance on the Credit Card:", min_value = 0)
        total_ct_chng_q4_q1 = st.number_input("Change in Transaction Count (Q4 over Q1):", min_value = 0.0, step=0.001, format="%0.3f")
        months_inactive_12_mon = st.number_input("No. of months inactive in the last 12 months:", min_value = 0)
        contacts_count_12_mon = st.number_input("No. of Contacts in the last 12 months:", min_value = 0)
        total_relationship_count = st.number_input("Total number of products held by the customer:", min_value = 0)
        credit_limit = st.number_input("Credit Limit on the Credit Card:", min_value = 0)
        avg_utilization_ratio = st.number_input("Average Card Utilization Ratio:", min_value = 0.0, step=0.001, format="%0.3f")
        input_data=[[total_trans_amt, total_trans_ct, total_amt_chng_q4_q1, total_revolving_bal, total_ct_chng_q4_q1, 
                     months_inactive_12_mon, contacts_count_12_mon, total_relationship_count, credit_limit, 
                     avg_utilization_ratio]]
        std_data = standard(input_data)
 
        # make prediction
        prediction_single = model.predict(std_data)
        
        if st.button('Predict'):
            if (prediction_single[0] == 1):
                st.warning('The customer is going to churn! 😡')
            else:
                st.success('The customer is not going to churn. 😊')
    
    if add_selectbox == "Multiple":
        file_upload = st.file_uploader('Please kindly upload csv file for predictions', type = ['csv'])
        if file_upload is not None:
            input_data_multiple = pd.read_csv(file_upload)
            df = pd.DataFrame(input_data_multiple)
            df.columns = map(str.lower, df.columns)
            df = df[['total_trans_amt', 'total_trans_ct', 'total_amt_chng_q4_q1', 'total_revolving_bal', 
                     'total_ct_chng_q4_q1', 'months_inactive_12_mon', 'contacts_count_12_mon', 'total_relationship_count',
                     'credit_limit', 'avg_utilization_ratio']]
            std_data_multiple = standard(df)
            
            if st.button('Predict'):
                prediction = model.predict(std_data_multiple)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({0:'The customer is not going to churn.',
                                                    1:'The customer is going to churn!'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)
            
if __name__ == '__main__':
        main()
