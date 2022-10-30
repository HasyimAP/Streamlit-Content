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

st.title('CO2 and Greenhouse Gas Emissions')

# =========================================================
st.header('The Raw Dataset')
'''
The raw dataset consist of 25,204 rows and 58 columns.
There are **760,695 missing values** in total from 1,461,832 (25,204 \* 58) data.
That means we have around **52.03% missing values** from the entire dataset.
The only column that don't have any missing values are the country and year columns.

You can see the raw dataset below with the column descriptions.
'''
st.dataframe(df)

with st.expander('Column details'):
    column_desc = pd.read_csv(path + '/owid-co2-codebook.csv')
    column_desc = column_desc.drop('source', axis=1)
    
    for col in column_desc['column']:
        if col not in df.columns.tolist():
            column_desc = column_desc.drop(column_desc[column_desc['column'] == col].index, axis=0)

    column_desc.reset_index(drop=True, inplace=True)
    for i in range(column_desc.shape[0]):
        st.write(f'{column_desc.iloc[i, 0]}: {column_desc.iloc[i, 1]}')

'''
Below here we can see how many missing values on each variables, on each country, and on each continent (excl. iso_code).
'''

md_1, md_2, md_3 = st.columns(3)

