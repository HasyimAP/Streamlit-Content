import os
import pandas as pd
import streamlit as st
import plotly.express as px

# dataset
path = os.path.dirname(__file__)
data_file = path + '/data wayang.xlsx'

df_region = pd.read_excel(data_file, sheet_name='Tipe daerah')
df_gender = pd.read_excel(data_file, sheet_name='Jenis kelamin')
df_age = pd.read_excel(data_file, sheet_name='Umur')

# page settings
st.set_page_config(
    page_title='Peminat Wayang',
    layout='wide'
)

# ===================================================================
fig = px.line(
    df_region[df_region['Tipe Daerah'] == 'Indonesia'],
    x='Tahun',
    y='Jumlah Peminat',
    markers=True
)

fig.update_layout(
    width=1280,
    height=480,
    title='Peminat wayang di Indonesia',
    yaxis_title='Jumlah peminat'
)

st.plotly_chart(fig)

# ===================================================================
fig = px.line(
    df_region[df_region['Tipe Daerah'] != 'Indonesia'],
    x='Tahun', 
    y='Presentase Peminat', 
    color='Tipe Daerah',
    markers=True
)

fig.update_layout(
    width=1280,
    height=480,
    title='Presentase peminat wayang berdasarkan daerah',
    yaxis_title='Presentase peminat (%)'
)

st.plotly_chart(fig)

# ===================================================================
fig = px.bar(
    df_gender,
    x='Tahun',
    y='Presentase Peminat',
    color='Jenis Kelamin',
    text_auto=True
)

fig.update_layout(
    width=1280,
    height=480,
    title='Presentase peminat wayang berdasarkan jenis kelamin',
    yaxis_title='Presentase peminat (%)',
    barmode='group'
)

st.plotly_chart(fig)

# ===================================================================
selected_year = st.selectbox(
    'Pilih tahun:',
    df_age['Tahun'].unique().tolist()
)

fig = px.bar(
    df_age[df_age['Tahun'] == selected_year],
    x='Kelompok Umur',
    y='Presentase Peminat',
    text_auto=True
)

st.plotly_chart(fig)

# st.dataframe(df_region)
# st.dataframe(df_gender)
# st.dataframe(df_age)
