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
Below here we can see how many missing values on each variables, on each country, and on each continent.
'''

md_1, md_2, md_3 = st.columns(3)

with md_1:
    null_data = pd.DataFrame(df.isna().sum(), columns=['total missing values'])

    for curr_var in (null_data.index):
        total_miss_data = null_data.loc[curr_var, 'total missing values']
        pct = (total_miss_data/df.shape[0])*100
        null_data.at[curr_var, 'miss data pct.'] = pct

    st.dataframe(null_data)

with md_2:
    country_miss_data = pd.DataFrame(
        columns=[
            'country',
            'total missing data',
            'total data',
            'miss data pct.'
        ]
    )

    country_miss_data['country'] = df['country'].unique()

    for i in range(country_miss_data.shape[0]):
        curr_country = country_miss_data['country'][i]

        total_miss_data = df[df['country'] == curr_country].isna().sum().sum()
        country_miss_data.at[i, 'total missing data'] = total_miss_data

        total_data = df[df['country'] == curr_country].shape[0] * df[df['country'] == curr_country].shape[1]
        country_miss_data.at[i, 'total data'] = total_data

        pct = (total_miss_data/total_data)*100
        country_miss_data.at[i, 'miss data pct.'] = pct
    
    continent = df[df['iso_code'].isna() == True]['country'].unique().tolist()
    continent.extend(['Antarctica', 'World'])
    cont_miss_data = country_miss_data.loc[country_miss_data['country'].isin(continent)]
    cont_miss_data = cont_miss_data.rename(columns={'country': 'continent'})
    cont_miss_data.reset_index(inplace=True, drop=True)

    i_drop = country_miss_data.index[country_miss_data['country'].isin(continent)].tolist()
    country_miss_data = country_miss_data.drop(i_drop, axis=0)
    country_miss_data.reset_index(drop=True, inplace=True)

    st.dataframe(country_miss_data)

with md_3:
    st.dataframe(cont_miss_data)

