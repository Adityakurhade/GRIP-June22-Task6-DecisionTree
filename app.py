# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:48:12 2022

@author: arkur
"""
import numpy as np
import streamlit as st
from pickle import load

dt_clf = load(open("dt_model.pkl","rb"))

st.title("Detect specis of Iris flower")

st.text("""Here you can simply guess which iris flower species you have in your hand.         
For that, simply fill the dimensions of flower below and rest will be on us.        
Don't forget to press predict button ;) """)

sepal_length = st.number_input("Sepal Length",
                               help="Enter sepal length in centimeters")

sepal_width = st.number_input("Sepal Width",
                              help="Enter Sepal Width in centimeters")

petal_length = st.number_input("Petal Length",
                               help="Enter Petal length in centimeters")

petal_width = st.number_input("Petal width",
                              help="Enter Petal Length in centimeters")

input_arr = np.array([sepal_length,sepal_width,
                      petal_length,petal_width])

result = dt_clf.predict(input_arr.reshape(1,-1))

species = result[0]

if st.button("Predict"):
    st.write("The species of Iris flower in your hand is ")
    st.subheader(species)
else:
    st.write("Your species name will appear here")