import os
import pandas as pd
import streamlit as st

from PIL import Image

# dataset
path = os.path.dirname(__file__)
df = pd.read_csv(path + '/owid-co2-data.csv')

# page title
icon = Image.open(path + '/co2.png')

st.set_page_config(
    page_title='CO2 Emissions',
    page_icon=icon,
    layout='wide'
)