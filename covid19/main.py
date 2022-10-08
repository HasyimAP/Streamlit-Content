import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

# data
df = pd.read_csv('/covid_19_indonesia_time_series_all.csv')
# delete unused/unrelevant columns
df = df.dropna(axis=1, how='all')
df = df.drop(labels=['Location ISO Code', 'Special Status', 'Country','Continent', 'Time Zone', 'Location Level'],axis=1)

df_ind = df[df['Location'] == 'Indonesia']

# webapps design
st.set_page_config(
    page_title='Dashboard',
    page_icon=':bar_chart:',
    layout='wide'
)

st.header('Indonesia Covid-19 Update')

# Total active cases 
# ===============================
st.subheader('Total active cases from all over Indonesia')
st.write('Total active cases of covid-19 in Indonesia since March 2nd 2020 to September 16th 2022. Active cases refer to the number of infected people.')

fig_indo = px.line(df_ind, x='Date', y='Total Active Cases', width=1280, height=480)
st.plotly_chart(fig_indo)

# Active cases on each province
# ===============================
st.subheader('Total active cases on each province')

df_prov = df.copy()
i = df_prov[df_prov['Location'] == 'Indonesia'].index
df_prov = df_prov.drop(i)

province = st.multiselect(
    'Select the province(s):',
    options=df_prov['Location'].unique(),
    default=df_prov['Location'].unique()
)

df_prov = df_prov.query(
    'Location == @province'
)

fig_prov = px.line(df_prov, x='Date', y='Total Active Cases', color='Location', width=1280, height=480)
st.plotly_chart(fig_prov)

# Total cases grouped by province
# ===============================
st.subheader('Total cases on each province')

df_prov2 = pd.DataFrame(columns = ['Location', 'Total Cases'])
df_prov2['Location'] = df['Location'].unique()
i = df_prov2[df_prov2['Location'] == 'Indonesia'].index
df_prov2 = df_prov2.drop(i)

for land in df_prov2['Location']:
    i = df_prov2[df_prov2['Location']==land].index.values
    df_prov2['Total Cases'][i] = df[df['Location']==land]['Total Cases'].max()

df_prov2 = df_prov2.dropna()
df_prov2 = df_prov2.sort_values('Total Cases', ascending=False)
df_prov2 = df_prov2.reset_index(drop=True)

fig_prov2 = px.bar(df_prov2, y='Total Cases', x='Location', title='Total cases on each province', text_auto='.2s', width=1280, height=480)
fig_prov2.update_layout(title='', xaxis_title='Province')

st.plotly_chart(fig_prov2)

# Total cases grouped by island
# ===============================
st.subheader('Total cases on each island')

st.write('More than half of the Covid-19 cases occurred on the island of Java. This is because the Java is the most populous island in Indonesia.')

df_land = df_prov2.copy()
df_land['Island'] = ''

for prov in df_land['Location']:
    i = df_land[df_land['Location']==prov].index.values
    df_land['Island'][i] = df[df['Location']==prov]['Island'].unique()
    
df_island = pd.DataFrame(columns=['Island', 'Total Cases'])
df_island['Island'] = df_land['Island'].unique()

for land in df_island['Island']:
    i = df_island[df_island['Island']==land].index.values
    df_island['Total Cases'][i] = df_land[df_land['Island']==land]['Total Cases'].sum()

    
fig_island = px.pie(df_island, values='Total Cases', names='Island')
st.plotly_chart(fig_island)

# Total cases & population density
# ===============================
st.subheader('Total Cases and Population Density')

st.write('Areas with denser populations tend to have higher COVID-19 cases than other regions. This is because the denser an area is, the easier it is for the Covid-19 virus to spread.')

df_pd = pd.DataFrame(columns = ['Location', 'Population Density', 'Total Cases'])
df_pd['Population Density'] = df['Population Density'].unique()

for pop_den in df_pd['Population Density']:
    i = df_pd[df_pd['Population Density'] == pop_den].index.values
    df_pd['Total Cases'][i] = df[df['Population Density'] == pop_den]['Total Cases'].max()
    df_pd['Location'][i] = df[df['Population Density'] == pop_den]['Location'].unique()
    
# drop Indonesia's Population Density
for pop_den in df_pd['Population Density']:
    i = df_pd[df_pd['Population Density'] == pop_den].index.values
    for loc in df[df['Population Density'] == pop_den]['Location']:
        if loc == 'Indonesia':
            df_pd = df_pd.drop(labels=i, axis=0)
            break

df_pd = df_pd.sort_values('Population Density')
df_pd = df_pd.reset_index(drop=True)

fig_popden = px.line(df_pd, x='Population Density', y='Total Cases', text='Location', width=1280, height=480)
st.plotly_chart(fig_popden)
