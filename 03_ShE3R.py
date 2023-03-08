import streamlit as st
from PIL import Image
import pandas as pd

### Naming the app with an icon
img = Image.open('stratlytics.jpeg')
st.set_page_config(page_title = "SHE3R",
    page_icon=img)

## Functions for creating the app

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

    elif nav == "PREDICTION":
        st.write("SKU Prediction")
    else:
        st.write("All SKU Prediction")
    



if __name__ == '__main__':
	main()
