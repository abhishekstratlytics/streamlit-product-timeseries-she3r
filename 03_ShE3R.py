import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly  
#from plotly import graph_objs as go 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import datetime
import pmdarima as pm
from sklearn.metrics import mean_absolute_error as mae
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf,pacf
import itertools 


### Naming the app with an icon
img = Image.open('stratlytics.jpeg')
st.set_page_config(page_title = "SHE3R",
    page_icon=img)

## Functions for creating the app

### Checks for stationarity and returns the order of differencing required
def return_stationary(sales_orignal):
    ctr=0
    if adfuller(sales_orignal)[1]>0.05:
        while(ctr<=3):
            ctr+=1
            sales_orignal = sales_orignal - sales_orignal.shift(1)
            sales_orignal.dropna(inplace=True)
            if adfuller(sales_orignal)[1]<=0.05:
                break
            else:
                continue
    return ctr
# Generating training and testing Datasets required for time series modelling
def train_test_data_prep_ts(data,training_periods):
    train_data = data[:training_periods]
    test_data = data[training_periods:]
    return train_data,test_data

# Generated the input data required for forcasting values of test period using regression models
def train_test_data_prep_reg(data,periods_of_forecast,training_periods):
    test_periods=len(data)-training_periods
    sales = data.copy(deep=True)
    d = sales.columns[0]
    sales['Lag_p']=sales[d].shift(periods_of_forecast) # Lag for p periods
    sales['Lag_p1']=sales[d].shift(periods_of_forecast+1) # Lag for p+1 periods
    sales=sales.dropna()
    #Creating values for model metrics calculation
    x1_v,x2_v,y=sales['Lag_p'],sales['Lag_p1'],sales[d] #Storing values of lags(independent features) and actual sales(dependent feature) in series
    x1_v,x2_v,y=np.array(x1_v),np.array(x2_v),np.array(y) #Converting the series in array
    x1_v,x2_v,y=x1_v.reshape(-1,1),x2_v.reshape(-1,1),y.reshape(-1,1)
    final_x_v=np.concatenate((x1_v,x2_v),axis=1) # Series of independent features
    #print(final_x_v)
    X_train,X_test,y_train,y_test=final_x_v[:-test_periods],final_x_v[-test_periods:],y[:-test_periods],y[-test_periods:] #Splitting data in train and test
    return X_train,X_test,y_train,y_test

# Generated the input data required for forcasting future values using regression models
def reg_model_data_generator(data,periods):
    sales = data.copy(deep=True)
    d = sales.columns[0]
    sales['Lag_p']=sales[d].shift(periods) # Lag for p periods
    sales['Lag_p1']=sales[d].shift(periods+1) # Lag for p+1 periods
    sales=sales.dropna()
    x1,x2,y = np.array(sales['Lag_p']),np.array(sales['Lag_p1']),np.array(sales[d])
    x1,x2=x1.reshape(-1,1),x2.reshape(-1,1)
    x_model,y_model = np.concatenate((x1,x2),axis=1),y
    x1_pred,x2_pred = np.array(sales[d][-periods:]),np.array(sales[d][-(periods+1):-1])
    x1_pred,x2_pred=x1_pred.reshape(-1,1),x2_pred.reshape(-1,1)
    x_pred = np.concatenate((x1_pred,x2_pred), axis=1)
    return x_model,y_model,x_pred
# Returns the metrics (RMSE, MAPE, MAE) of the prediction models
def metrics(predictions,targets):
    return round(np.sqrt(((predictions - targets) ** 2).mean()),2),round((np.mean(np.abs((targets - predictions)/targets))*100),2), round(mae(targets, predictions),2)


# Functions for the model

# 9 week Moving Average
def ma12_prediction(sales,periods_of_forecast):
    #sales = sales[[d]]
    model = ARIMA(sales, order=(0,0,9))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=periods_of_forecast)
    return predictions

#Auto ARIMA
def ar_prediction(sales,d,periods_of_forecast):
    model_auto = pm.auto_arima(sales[d], start_p=0, d=return_stationary(sales),start_q=0,
                          max_p = 5,max_d=2,max_q=5, m = 0, seasonal = False,
                          max_order=2,test='adf',error_action = 'ignore',
                          suppress_warnings =True,
                          stepwise =True,trace = True, n_fits=50)
    model_auto_fit = model_auto.fit(sales[d])
    predictions = model_auto_fit.predict(n_periods=periods_of_forecast)
    return predictions

# SARIMA
def sarima_prediction(sales,d,periods_of_forecast):
    p = q = range(0, 2)
    diff= range(0, 2)
    pdq = list(itertools.product(p, diff, q))
    ARIMA_AIC = pd.DataFrame(columns=['param', 'AIC'])
    for param in pdq:
        ARIMA_model = ARIMA(sales[d],order=param).fit()
        #print('ARIMA{} - AIC:{}'.format(param,ARIMA_model.aic))
        ARIMA_AIC = ARIMA_AIC.append({'param':param, 'AIC': ARIMA_model.aic}, ignore_index=True)
    ARIMA_AIC = ARIMA_AIC.sort_values(by='AIC',ascending=True)
    model=sm.tsa.statespace.SARIMAX(sales[d],order=(ARIMA_AIC['param'][ARIMA_AIC['param'].index[0]][0], ARIMA_AIC['param'][ARIMA_AIC['param'].index[0]][1], ARIMA_AIC['param'][ARIMA_AIC['param'].index[0]][2]),seasonal_order=(ARIMA_AIC['param'][ARIMA_AIC['param'].index[0]][0],ARIMA_AIC['param'][ARIMA_AIC['param'].index[0]][1],ARIMA_AIC['param'][ARIMA_AIC['param'].index[0]][2],12))
    results=model.fit()
    predictions=results.predict(start=sales.count()[0],end=sales.count()[0]+periods_of_forecast-1,dynamic=True)
    return predictions

# Random Forest
def rf_prediction(X_training,y_training,X_prediction):
    model=RandomForestRegressor(n_estimators=100,max_features=2, random_state=1)
    #Creating values for model metrics calculation
    model.fit(X_training,y_training) # Fitting the model
    predictions=model.predict(X_prediction) # Getting predictions for test phase
    return predictions

# XG Boost
def xgb_prediction(X_train,y_train,X_test):
    model = XGBRegressor()
    model.fit(X_train,y_train) # Fitting the model
    predictions=model.predict(X_test) # Getting predictions for test phase
    return predictions


### Functions for Analysis
# Returns the index of winner model
def win(df,models,metric,d):
    l=[]
    for i in range(len(models)):
        l.append(metrics(df[d],df[models[i]])[metric])
    min_index=0
    for i in range(len(l)):
        if l[i]<l[min_index]:
            min_index=i
    return min_index

# Display which model is winning
def display_winning_model(models,index):
    print(f'The winner model is {models[index]} model')

# Returns the Dataframe of models and metrics
def metrics_table(data,data_test_pred,actual_data,list_metric,models,d):
    metrics_table=pd.DataFrame(columns=models)
    for i in models:
        metrics_table[i] = list(metrics(data_test_pred[i],actual_data))
    metrics_table['Metric']=list_metric
    metrics_table = metrics_table.set_index('Metric')
    return metrics_table




## The main function

def main():
    st.title("SHE3R")
    st.image("stratlytics.jpeg",width=300)
    st.subheader("TimeSeries Forecast")
    nav= st.sidebar.radio("Navigation",["HOME","PREDICTION","OVERALL"])

    ## Loading the data
    uploaded_file = st.file_uploader("File_Upload")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("Weekly_Sales.csv")


    if nav == "HOME":
        st.write("HOME PAGE")
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

    elif nav == "PREDICTION":
        st.header("Your Product Prediction")
        l=list(data.columns)
        del l[0]
        ## Input Data Selection
        d=st.selectbox("Select your Product",l)
        data = data[['Time',d]]
        data=data.set_index("Time")
        ## Testing period Selection       
        testing_periods = st.slider("Weeks of testing/validating:",4,12,6)
        ## Output Period Selection
        periods_of_forecast= st.slider("Weeks of prediction:",4,12,6)
        forecast_periods = periods_of_forecast
        ### Testing the second Function
        training_periods = len(data)-testing_periods
        a,b = train_test_data_prep_ts(data,training_periods)
        col1,col2 = st.columns(2)
        with col1:
            st.write("Data used for building the model")
            st.write(a)
        with col2:
            st.write("Data used for testings the model")
            st.write(b)
        ### Testing the third function-- Working
        ### Testing the fourth function-- Working
        ### Testing the fifth function --- need to be re looked
        ### Testing the sixth function
        # z= rf_prediction(train_test_data_prep_reg(data,periods_of_forecast,training_periods)[0],
        #                     train_test_data_prep_reg(data,periods_of_forecast,training_periods)[2],
        #                     train_test_data_prep_reg(data,periods_of_forecast,training_periods)[1])
        # z = xgb_prediction(train_test_data_prep_reg(data,periods_of_forecast,training_periods)[0],
        #                     train_test_data_prep_reg(data,periods_of_forecast,training_periods)[2],
        #                     train_test_data_prep_reg(data,periods_of_forecast,training_periods)[1])
        #z = ar_prediction(data,d,forecast_periods)
        #st.write(z)
        models = ['ARIMA','SARIMA','Moving Average','Random Forest','XGBoost','Average'] #Change here when add/delete model
        models_without_average = ['ARIMA','SARIMA','Moving Average','Random Forest','XGBoost'] #Change here when add/delete model
        list_of_metrics=['RMSE','MAPE','MAE'] #Change here when add/delete metric
        test_values = {models[0]:ar_prediction(train_test_data_prep_ts(data,training_periods)[0],d,len(train_test_data_prep_ts(data,training_periods)[1])),models[1]:sarima_prediction(train_test_data_prep_ts(data,training_periods)[0],d,len(train_test_data_prep_ts(data,training_periods)[1])),models[2]:ma12_prediction(train_test_data_prep_ts(data,training_periods)[0],len(train_test_data_prep_ts(data,training_periods)[1])),models[3]: rf_prediction(train_test_data_prep_reg(data,forecast_periods,training_periods)[0],train_test_data_prep_reg(data,forecast_periods,training_periods)[2],train_test_data_prep_reg(data,forecast_periods,training_periods)[1]),models[4]:xgb_prediction(train_test_data_prep_reg(data,forecast_periods,training_periods)[0],train_test_data_prep_reg(data,forecast_periods,training_periods)[2],train_test_data_prep_reg(data,forecast_periods,training_periods)[1])}
        test_values = pd.DataFrame(test_values)
        
        test_values = test_values.reset_index(drop=True)
        #st.write(test_values)
        test_df = train_test_data_prep_ts(data,training_periods)[1]
        t2 = train_test_data_prep_ts(data,training_periods)[1]
        test_df = test_df.reset_index(drop = True)
        #st.write(test_values)
        for i in models_without_average:
            test_df[i]=test_values[i]
        test_df.index=t2.index 
        #test_df.index= pd.DataFrame(index=t2.index)
        #st.write(test_df)
        # test_df = pd.concat([test_df, test_values], axis=1)

        
        #st.write(test_df["ARIMA"])
        test_df['Average']=test_df[models_without_average].mean(axis=1)
        #st.write(test_df)
        # mapping of models with their predictions in the future
        prediction_values = {models[0]:ar_prediction(data,d,forecast_periods),
                             models[1]:sarima_prediction(data,d,forecast_periods),
                             models[2]:ma12_prediction(data,forecast_periods),
                             models[3]:rf_prediction(reg_model_data_generator(data,forecast_periods)[0],reg_model_data_generator(data,forecast_periods)[1],reg_model_data_generator(data,forecast_periods)[2]),
                             models[4]:xgb_prediction(reg_model_data_generator(data,forecast_periods)[0],reg_model_data_generator(data,forecast_periods)[1],reg_model_data_generator(data,forecast_periods)[2])}
        #st.write(test_values)
        # Dictionary is created that maps the metric to a numeric value
        mapping_dict = {list_of_metrics[0]:0,list_of_metrics[1]:1,list_of_metrics[2]:2}
        #st.write(mapping_dict)
        user_input_metric=st.radio("Navigation",list_of_metrics)
        metric_index = mapping_dict[user_input_metric] #storing the index of metrics
        #st.write(test_df,models,metric_index,d)
        index_of_winner_model = win(test_df,models,metric_index,d) #storing the index of winner model
        display_winning_model(models,index_of_winner_model) #Display the name of winner model
        #Creating a Data Frame of predictions of all the models including the average of all models
        predict_df = pd.DataFrame(prediction_values[models[0]],columns=[models[0]])
        for i in range(1,len(models_without_average)):
            predict_df[models_without_average[i]]=prediction_values[models_without_average[i]]
        predict_df['Average']=predict_df[models_without_average].mean(axis=1)
        predict_df = predict_df.round(2)
        metrics_df = metrics_table(data,test_df,train_test_data_prep_ts(data,training_periods)[1][d],list_of_metrics,models,d)
        st.write(metrics_df)
        fig1 = plt.figure(figsize=(15,6))
        plt.plot(train_test_data_prep_ts(data,training_periods)[1])
        plt.plot(test_df[models[index_of_winner_model]])
        plt.legend(["Actual","Prediction"], loc ="lower right")
        st.pyplot(fig1)
        fig2 = plt.figure(figsize=(20,6))
        plt.plot(data[d])
        plt.plot(predict_df[models[index_of_winner_model]])
        plt.legend(["Actual","Prediction"], loc ="upper right")
        st.pyplot(fig2)
        prediciton = predict_df[models[index_of_winner_model]]
        # prediciton.rename(columns = {models[index_of_winner_model]:d}, inplace = True)
        st.download_button("Download Output",prediciton.to_csv(),file_name = "Your_Output.csv",mime='text/csv')


    else:
        st.write("All SKU Prediction")
        sales_multi =data.copy()
        l=list(data.columns)
        del l[0]
        ## Input Data Selection
        d=st.selectbox("Select your Product",l)
        data = data[['Time',d]]
        data=data.set_index("Time")
        ## Testing period Selection       
        testing_periods = st.slider("Weeks of testing/validating:",4,12,6)
        training_periods = len(sales_multi)-testing_periods
        ## Output Period Selection
        periods_of_forecast= st.slider("Weeks of prediction:",4,12,6)
        forecast_periods = periods_of_forecast
        list_of_metrics=['RMSE','MAPE','MAE']
        mapping_dict = {list_of_metrics[0]:0,list_of_metrics[1]:1,list_of_metrics[2]:2}
        pred_multi = pd.DataFrame(columns=[sales_multi.columns[1:]])
        #st.write(mapping_dict)
        
        user_input_metric=st.radio("Navigation",list_of_metrics)
        models = ['ARIMA','SARIMA','Moving Average','Random Forest','XGBoost','Average'] #Change here when add/delete model
        models_without_average = ['ARIMA','SARIMA','Moving Average','Random Forest','XGBoost'] #Change here when add/delete model
        for d in sales_multi.columns[1:]:

        # mapping of models with their predictions in the test period
            data = sales_multi[[d]]
            test_values = {models[0]:ar_prediction(train_test_data_prep_ts(data,training_periods)[0],d,len(train_test_data_prep_ts(data,training_periods)[1])),\
                        models[1]:sarima_prediction(train_test_data_prep_ts(data,training_periods)[0],d,len(train_test_data_prep_ts(data,training_periods)[1])),\
                        models[2]:ma12_prediction(train_test_data_prep_ts(data,training_periods)[0],len(train_test_data_prep_ts(data,training_periods)[1])),\
                        models[3]: rf_prediction(train_test_data_prep_reg(data,forecast_periods,training_periods)[0],train_test_data_prep_reg(data,forecast_periods,training_periods)[2],train_test_data_prep_reg(data,forecast_periods,training_periods)[1]),\
                        models[4]:xgb_prediction(train_test_data_prep_reg(data,forecast_periods,training_periods)[0],train_test_data_prep_reg(data,forecast_periods,training_periods)[2],train_test_data_prep_reg(data,forecast_periods,training_periods)[1])}
            # creating a data frame of predictions of all models including the average model
            test_df = train_test_data_prep_ts(data,training_periods)[1]
            for i in models_without_average:
                test_df[i]=test_values[i]
            test_df['Average']=test_df[models_without_average].mean(axis=1)
            # mapping of models with their predictions in the future
            prediction_values = {models[0]:ar_prediction(data,d,forecast_periods),\
                                models[1]:sarima_prediction(data,d,forecast_periods),\
                                models[2]:ma12_prediction(data,forecast_periods),\
                                models[3]: rf_prediction(reg_model_data_generator(data,forecast_periods)[0],reg_model_data_generator(data,forecast_periods)[1],reg_model_data_generator(data,forecast_periods)[2]),\
                                models[4]:xgb_prediction(reg_model_data_generator(data,forecast_periods)[0],reg_model_data_generator(data,forecast_periods)[1],reg_model_data_generator(data,forecast_periods)[2])}
            # Dictionary is created that maps the metric to a numeric value
            metric_index = mapping_dict[user_input_metric]
            index_of_winner_model = win(test_df,models,metric_index,d) #storing the index of winner model
            display_winning_model(models,index_of_winner_model) #Display the name of winner model
            #Creating a Data Frame of predictions of all the models including the average of all models
            predict_df = pd.DataFrame(prediction_values[models[0]],columns=[models[0]])
            for i in range(1,len(models_without_average)):
                predict_df[models_without_average[i]]=prediction_values[models_without_average[i]]
            predict_df['Average']=predict_df[models_without_average].mean(axis=1)
            # Display predictions and plots
            pred_multi[d] = predict_df[models[index_of_winner_model]]
        st.write(pred_multi)
        st.download_button("Download All Output",pred_multi.to_csv(),file_name = "All_SKU_Output.csv",mime='text/csv')



if __name__ == '__main__':
	main()
