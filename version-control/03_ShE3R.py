import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import plotly  
from plotly import graph_objs as go 
import numpy as np
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error as mae
st.set_option('deprecation.showPyplotGlobalUse', False)

#adding Title
st.title("SHE3R")
st.subheader("TimeSeries Forecast")

#importing data
uploaded_file = st.file_uploader("File_Upload")
data = pd.read_csv(uploaded_file)


# adding a navigation bar
nav= st.sidebar.radio("Navigation",["HOME","PREDICTION","METRICS"])
# Action for each navigation tasks
## Home Page Action

if nav == "HOME":
    st.write("HOME")
    st.image("stratlytics.jpeg",width=300)
    #st.write("Select your Product")
    l=list(data.columns)
    del l[0]
    d=st.selectbox("Select your Product",l)
    
    if st.checkbox("Show Recent"):
        st.table(data[["Time",d]].tail())
    if st.checkbox("Show Beginning"):
        st.table(data[["Time",d]].head())
    # Data Viz
    st.write("Product Performance")
    st.line_chart(data,x='Time',y=d)

    #st.line_chart(data,x='Time',y=d)

    st.write("Relative Product Performance")
    col2=st.multiselect("Select Products for Comparision",l)
    #plt.plot(data2['num'],data2[col2])
    st.line_chart(data,x='Time',y=col2)
    #st.pyplot()
    
    

if nav == "PREDICTION":
    Model = st.radio("Model",["Model1","Model2","Model3"])
    st.header("Your Product Prediction")
    l=list(data.columns)
    del l[0]
    d=st.selectbox("Select your Product",l)
    period= st.slider("Weeks of prediction:",1,12,4)
    if Model == "Model1":
        df_train = data[['Time',d]]
        df_train=df_train.rename(columns= {"Time": "ds",d:"y"})
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period,freq="W")
        forecast = m.predict(future)
        output=forecast[["ds","yhat_lower","yhat_upper","yhat"]]
        output.rename(columns={"ds":"Period","yhat_lower":"Lower Limit","yhat_upper":"Upper Limit","yhat":"Prediction"},inplace=True)
        if st.button("Predict"):
            st.write(output.tail(period))
            y_actual=df_train["y"].tail(5)
            y_pred=(output.iloc[:-period,:]).tail(5)["Prediction"]
            def rmse(predictions, targets):
                return np.sqrt(((predictions - targets) ** 2).mean())
            r = round(rmse(y_pred,y_actual),2)
            st.write("RMSE")
            st.write(r)
            error = round(mae(y_actual, y_pred),2)
            st.write("MAE")
            st.write(error)
            def MAPE(Y_actual,Y_Predicted):
                mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
                return mape
            error = round(MAPE(y_actual, y_pred),2)
            st.write("MAPE")
            st.write(error)

            st.subheader('Forecast data plot')
            fig1 = plot_plotly(m,forecast)
            st.plotly_chart(fig1)
            st.download_button("Download Output",forecast.to_csv(),file_name = "Your_Output.csv",mime='text/csv')
            

    if Model == "Model2":
        st.write("Raj Naam to suna hi Hoga")
        st.image("Raj.jpg",width=300)
    if Model == "Model3":
        st.write("Raj Naam to suna hi Hoga")
        st.image("Raj2.jpg",width=300)

    #st.write(output.tail(period))
    

if nav  == "METRICS":
    st.header("Your Product Metrics")
    l=list(data.columns)
    del l[0]
    d=st.selectbox("Select your Product",l)
    #period= st.slider("Weeks of prediction:",1,12,4)
    df_train = data[['Time',d]]
    df_train=df_train.rename(columns= {"Time": "ds",d:"y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=4,freq="W")
    forecast = m.predict(future)
    if st.button("Metrics"):
        st.subheader('Forecast component plot')
        fig2 = m.plot_components(forecast)
        st.write(fig2)

