import streamlit as st 
from PIL import Image
import pandas as pd 
import matplotlib.pyplot as plt 
import plotly  
from plotly import graph_objs as go 
import numpy as np
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import datetime
import pmdarima as pm
from xgboost import XGBRegressor
st.set_option('deprecation.showPyplotGlobalUse', False)


img = Image.open('stratlytics.jpeg')
st.set_page_config(page_title = "SHE3R",
    page_icon=img)
#adding Title
st.title("SHE3R")
st.image("stratlytics.jpeg",width=300)
st.subheader("TimeSeries Forecast")

#importing data
uploaded_file = st.file_uploader("File_Upload")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("Weekly_Sales.csv")

# adding a navigation bar
nav= st.sidebar.radio("Navigation",["HOME","PREDICTION"])
# Action for each navigation tasks
## Home Page Action

if nav == "HOME":
    st.write("HOME")
    
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
    col2=st.multiselect("Select Products for Comparision",l,[l[0],l[2]])
    #plt.plot(data2['num'],data2[col2])
    st.line_chart(data,x='Time',y=col2)
    #st.pyplot()
    
    

if nav == "PREDICTION":
    #Model = st.radio("Model",["Model1","Model2","Model3"])
    st.header("Your Product Prediction")
    l=list(data.columns)
    del l[0]
    d=st.selectbox("Select your Product",l)
    period= st.slider("Weeks of prediction:",4,12,6)
    data = data[['Time',d]]
    p=period
    
    
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    def MAPE(Y_actual,Y_Predicted):
        mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
        return mape

    ### 12 Period Moving Average

    def ma12_metric_return(sales):
    #Preparing training and testing data (Prepared 48 weeks of training and 4 weeks of testing)
        sales = sales[[d]]
        training_end=sales.index[-5]
        testing_end=sales.index[-1]
        training_data=sales[:training_end+1]
        testing_data=sales[training_end+1:]
        #Fitting Moving Average
        #create the model
        model = ARIMA(training_data, order=(0,0,11))
        #fit the model
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=4)
        #Metric calculation
        metric_rmse = round(rmse(predictions,testing_data[d]),2)
        metric_mae = round(mae(testing_data[d], predictions),2)
        metric_mape= round(MAPE(testing_data[d], predictions),2)
        return predictions,metric_rmse, metric_mae, metric_mape

    ### Auto Arima

    def ar_metric_return(sales):
        #Preparing training and testing data (Prepared 48 weeks of training and 4 weeks of testing)
        training_end=sales.index[-5]
        testing_end=sales.index[-1]
        training_data=sales[:training_end+1]
        testing_data=sales[training_end+1:]
        #Fitting Auto Arima
        model_auto = pm.auto_arima(training_data[d],
                            m = 0, seasonal = False,
                            start_p = 0,start_q=0,max_order=2,test='adf',error_action = 'ignore',
                            suppress_warnings =True,
                            stepwise =True,trace = True)
        model_auto_fit = model_auto.fit(training_data['P1'])
        forecast_ar = model_auto_fit.predict(n_periods=4)
        #Metric calculation
        metric_rmse = round(rmse(forecast_ar,testing_data[d]),2)
        metric_mae = round(mae(testing_data[d], forecast_ar),2)
        metric_mape= round(MAPE(testing_data[d], forecast_ar),2)
        return forecast_ar,metric_rmse, metric_mae, metric_mape

    ## Random Forest

    def rf_metric_return(sales):
        sales['Lag_p']=sales['P1'].shift(p)
        sales['Lag_p1']=sales['P1'].shift(p+1)
        sales=sales.dropna()
        model=RandomForestRegressor(n_estimators=100,max_features=2, random_state=1)
        #Creating values for model metrics calculation
        x1_v,x2_v,y=sales['Lag_p'],sales['Lag_p1'],sales['P1']
        x1_v,x2_v,y=np.array(x1_v),np.array(x2_v),np.array(y)
        x1_v,x2_v,y=x1_v.reshape(-1,1),x2_v.reshape(-1,1),y.reshape(-1,1)
        final_x_v=np.concatenate((x1_v,x2_v),axis=1)
        #print(final_x_v)
        X_train,X_test,y_train,y_test=final_x_v[:-4],final_x_v[-4:],y[:-4],y[-4:]
        model.fit(X_train,y_train)
        pred=model.predict(X_test)
        metric_rmse = round(rmse(pred,y_test),2)
        metric_mae = round(mae(y_test, pred),2)
        metric_mape= round(MAPE(y_test, pred),2)
        return pred,metric_rmse,metric_mae,metric_mape
    
    ## XGboost model

    def xgb_metric_return(sales):
        sales['Lag_p']=sales[d].shift(p)
        sales['Lag_p1']=sales[d].shift(p+1)
        sales=sales.dropna()
        model = XGBRegressor()
        #Creating values for model metrics calculation
        x1_v,x2_v,y=sales['Lag_p'],sales['Lag_p1'],sales['P1']
        x1_v,x2_v,y=np.array(x1_v),np.array(x2_v),np.array(y)
        x1_v,x2_v,y=x1_v.reshape(-1,1),x2_v.reshape(-1,1),y.reshape(-1,1)
        final_x_v=np.concatenate((x1_v,x2_v),axis=1)
        #print(final_x_v)
        X_train,X_test,y_train,y_test=final_x_v[:-4],final_x_v[-4:],y[:-4],y[-4:]
        model.fit(X_train,y_train)
        pred=model.predict(X_test)
        metric_rmse = round(rmse(pred,y_test),2)
        metric_mae = round(mae(y_test, pred),2)
        metric_mape= round(MAPE(y_test, pred),2)
        return pred,metric_rmse,metric_mae,metric_mape

    ma_test,ma_rmse,ma_mae,ma_mape = ma12_metric_return(data)
    ma_metrics=[ma_rmse,ma_mae,ma_mape]
    ar_test,ar_rmse,ar_mae,ar_mape = ar_metric_return(data)
    ar_metrics=[ar_rmse,ar_mae,ar_mape]
    rf_test,rf_rmse,rf_mae,rf_mape = rf_metric_return(data)
    rf_metrics=[rf_rmse,rf_mae,rf_mape]
    xgb_test,xgb_rmse,xgb_mae,xgb_mape = xgb_metric_return(data)
    xgb_metrics=[xgb_rmse,xgb_mae,xgb_mape]

    avg_pred = pd.DataFrame(list(zip(ar_test, rf_test,ma_test,xgb_test)),columns={'ARIMA','Random Forest','MA','XGB'})
    data_test_pred = data[[d]].tail(4)
    data_test_pred['ARIMA']=ar_test
    data_test_pred['RANDOM FOREST']=rf_test
    data_test_pred['MA']=ma_test
    data_test_pred['XGB']=xgb_test
    data_test_pred['Avg_AR_RF_MA']=data_test_pred[['ARIMA','RANDOM FOREST','MA','XGB']].mean(axis=1)

    def avg_return_metrics(sales):
        metric_rmse = round(rmse(sales['Avg_AR_RF_MA'],sales[d]),2)
        metric_mae = round(mae(sales[d], sales['Avg_AR_RF_MA']),2)
        metric_mape= round(MAPE(sales[d], sales['Avg_AR_RF_MA']),2)
        return metric_rmse,metric_mae,metric_mape

    avg_rmse,avg_mae,avg_mape = avg_return_metrics(data_test_pred)
    avg_metrics=[avg_rmse,avg_mae,avg_mape]

    #Preparing metrics table for all the models(Mapping not done properly)
    metrics_table = pd.DataFrame(list(zip(ar_metrics, avg_metrics, rf_metrics,ma_metrics,xgb_metrics)),columns={'ARIMA','Average_AR_RF','Random Forest','MA','XGB'})
    list_metric = ['RMSE','MAE','MAPE']
    metrics_table['Metric']=list_metric
    metrics_table = metrics_table.set_index('Metric')

    st.dataframe(metrics_table)
    mapping_dict = {"RMSE":0,"MAE":1,"MAPE":2}
    user_input_metric=st.radio("Navigation",list_metric)
    st.write(user_input_metric)
    
    def winner_model(ar,ma,rf,xgb,avg):
        l=[ar[mapping_dict[user_input_metric]],ma[mapping_dict[user_input_metric]],rf[mapping_dict[user_input_metric]],xgb[mapping_dict[user_input_metric]],avg[mapping_dict[user_input_metric]]]
        min_element = l[0]
        min_index=0
        for i in range(len(l)):
            if l[i]<min_element:
                min_element=l[i]
                min_index=i
        return(min_index)

    winner_metric = winner_model(ar_metrics, ma_metrics, rf_metrics,xgb_metrics,avg_metrics)
    st.write(winner_metric)

    def ar_forecast_return(data):
        model_auto = pm.auto_arima(data[d],
                            m = 52, seasonal = False,
                            start_p = 0,start_q=0,max_order=2,test='adf',error_action = 'ignore',
                            suppress_warnings =True,
                            stepwise =True,trace = True)
        model_auto_fit = model_auto.fit(data[d])
        forecast_ar = model_auto_fit.predict(n_periods=p)
        prediction = pd.DataFrame(forecast_ar, columns=[d])
        return prediction 

    st.write(ar_forecast_return(data))





   