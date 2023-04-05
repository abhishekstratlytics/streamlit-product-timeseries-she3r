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

    ### AR Model
    def ar_metric_return(sales,d):
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
        model_auto_fit = model_auto.fit(training_data[d])
        forecast_ar = model_auto_fit.predict(n_periods=4)
        #Metric calculation
        metric_rmse = round(rmse(forecast_ar,testing_data[d]),2)
        metric_mae = round(mae(testing_data[d], forecast_ar),2)
        metric_mape= round(MAPE(testing_data[d], forecast_ar),2)
        return forecast_ar,metric_rmse, metric_mae, metric_mape
    
    ar_test,ar_rmse,ar_mae,ar_mape = ar_metric_return(data,d)
    ar_metrics=[ar_rmse,ar_mae,ar_mape]
    #st.write(ar_test)
    ### RF Model
    def rf_metric_return(sales,d,p):
        sales['Lag_p']=sales[d].shift(p)
        sales['Lag_p1']=sales[d].shift(p+1)
        sales=sales.dropna()
        model=RandomForestRegressor(n_estimators=100,max_features=2, random_state=1)
        #Creating values for model metrics calculation
        x1_v,x2_v,y=sales['Lag_p'],sales['Lag_p1'],sales[d]
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
    rf_test,rf_rmse,rf_mae,rf_mape = rf_metric_return(data,d,p)
    rf_metrics=[rf_rmse,rf_mae,rf_mape]
    #st.write(rf_test)

    #Third Prediction
    #avg_pred = pd.DataFrame(list(zip(ar_test, rf_test)),columns={'ARIMA','Random Forest'})
    #data_test_pred = data[d].tail(4)
    data_test_pred=pd.DataFrame()
    data_test_pred = data[["Time",d]].tail(4)
    #st.write(data_test_pred)
    data_test_pred['ARIMA']=ar_test
    data_test_pred['RANDOM FOREST']=rf_test
    data_test_pred['Avg_AR_RF']=data_test_pred[['ARIMA','RANDOM FOREST']].mean(axis=1)
    st.write(data_test_pred)

    
    def avg_return_metrics(sales,d):
        metric_rmse = round(rmse(sales['Avg_AR_RF'],sales[d]),2)
        metric_mae = round(mae(sales[d], sales['Avg_AR_RF']),2)
        metric_mape= round(MAPE(sales[d], sales['Avg_AR_RF']),2)
        return metric_rmse,metric_mae,metric_mape

    avg_rmse,avg_mae,avg_mape = avg_return_metrics(data_test_pred,d)
    avg_metrics=[avg_rmse,avg_mae,avg_mape]

    #Preparing metrics table for all the models
    metrics_table = pd.DataFrame(list(zip(ar_metrics, avg_metrics, rf_metrics)),columns={'ARIMA','Average_AR_RF','Random Forest'})
    list_metric = ['RMSE','MAE','MAPE']
    metrics_table['Metric']=list_metric
    metrics_table = metrics_table.set_index('Metric')
    metrics_table=metrics_table.reset_index()
    metrics_table=metrics_table[["Metric","ARIMA","Random Forest","Average_AR_RF"]]
    st.write(metrics_table)

    user_input_metric=st.radio("Navigation",list_metric)
    st.write(user_input_metric)
    mapping_dict = {"RMSE":0,"MAE":1,"MAPE":2}
    
    def winner(ar,rf,avg,user_input_metric):
        if ar[mapping_dict[user_input_metric]]<rf[mapping_dict[user_input_metric]] and ar[mapping_dict[user_input_metric]]<avg[mapping_dict[user_input_metric]]:
            return('The winner is ARIMA')
            return 0
        elif rf[mapping_dict[user_input_metric]]<ar[mapping_dict[user_input_metric]] and rf[mapping_dict[user_input_metric]]<avg[mapping_dict[user_input_metric]]:
            return('The winner is Random Forest')
            return 1
        else:
            return('The winner is average model')
            return 2

    winner_metric = winner(ar_metrics,rf_metrics,avg_metrics,user_input_metric)
    st.write(winner_metric)
    

    








    # if Model == "Model1":
    #     df_train = data[['Time',d]]
    #     df_train=df_train.rename(columns= {"Time": "ds",d:"y"})
    #     m = Prophet()
    #     m.fit(df_train)
    #     future = m.make_future_dataframe(periods=period,freq="W")
    #     forecast = m.predict(future)
    #     output=forecast[["ds","yhat_lower","yhat_upper","yhat"]]
    #     output.rename(columns={"ds":"Period","yhat_lower":"Lower Limit","yhat_upper":"Upper Limit","yhat":"Prediction"},inplace=True)
    #     if st.button("Predict"):
    #         st.write(output.tail(period))
    #         y_actual=df_train["y"].tail(5)
    #         y_pred=(output.iloc[:-period,:]).tail(5)["Prediction"]
    #         def rmse(predictions, targets):
    #             return np.sqrt(((predictions - targets) ** 2).mean())
    #         r = round(rmse(y_pred,y_actual),2)
    #         st.write("RMSE")
    #         st.write(r)
    #         error = round(mae(y_actual, y_pred),2)
    #         st.write("MAE")
    #         st.write(error)
    #         def MAPE(Y_actual,Y_Predicted):
    #             mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    #             return mape
    #         error = round(MAPE(y_actual, y_pred),2)
    #         st.write("MAPE")
    #         st.write(error)

    #         st.subheader('Forecast data plot')
    #         fig1 = plot_plotly(m,forecast)
    #         st.plotly_chart(fig1)
    #         st.download_button("Download Output",forecast.to_csv(),file_name = "Your_Output.csv",mime='text/csv')
            

    # if Model == "Model2":
    #     st.write("Raj Naam to suna hi Hoga")
    #     st.image("Raj.jpg",width=300)
    # if Model == "Model3":
    #     st.write("Raj Naam to suna hi Hoga")
    #     st.image("Raj2.jpg",width=300)

    #st.write(output.tail(period))
    

# if nav  == "METRICS":
#     st.header("Your Product Metrics")
#     l=list(data.columns)
#     del l[0]
#     d=st.selectbox("Select your Product",l)
#     #period= st.slider("Weeks of prediction:",1,12,4)
#     df_train = data[['Time',d]]
#     df_train=df_train.rename(columns= {"Time": "ds",d:"y"})
#     m = Prophet()
#     m.fit(df_train)
#     future = m.make_future_dataframe(periods=4,freq="W")
#     forecast = m.predict(future)
#     if st.button("Metrics"):
#         st.subheader('Forecast component plot')
#         fig2 = m.plot_components(forecast)
#         st.write(fig2)

