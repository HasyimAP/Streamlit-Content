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
'''
# Wayang Indonesia
by Bangkit Flex Team
'''

# ===================================================================
col1_cont1, col2_cont1 = st.columns([1, 2])

with col1_cont1:
    st.subheader('Peminat Wayang di Indonesia')
    '''
    Pada grafik disamping dapat dilihat bagaimana perkembangan jumlah peminat wayang di Indonesia.
    Data yang dipilih merupakan jumlah penduduk yang menonton pertunjukan/pameran seni pewayangan secara langsung maupun tidak langsung dari seluruh Indonesia.
    Jumlah tertinggi terjadi pada tahun 2015 dengan jumlah penduduk yang menonton pentas seni pewayangan lebih dari **28 juta jiwa**.
    Namun jumlah ini menurun drastis setelah tahun 2015 dan hanya berjumlah **9 juta jiwa** di tahun 2021.
    '''

with col2_cont1:
    fig = px.line(
        df_region[df_region['Tipe Daerah'] == 'Indonesia'],
        x='Tahun',
        y='Jumlah Peminat',
        markers=True
    )

    fig.update_layout(
        width=960,
        yaxis_title='Jumlah peminat'
    )

    st.plotly_chart(fig)

# ===================================================================
col1_cont2, col2_cont2 = st.columns([2, 1])

with col1_cont2:
    fig = px.line(
        df_region,
        x='Tahun', 
        y='Presentase Peminat', 
        color='Tipe Daerah',
        markers=True
    )

    fig.update_layout(
        width=890,
        height=480,
        yaxis_title='Presentase peminat (%)'
    )

    st.plotly_chart(fig)

with col2_cont2:
    st.subheader('Berdasarkan Jenis Daerah')
    '''
    Pada grafik di samping terdapat 3 garis mewakili 3 daerah, daerah perkotaan, pedesaan, dan seluruh Indonesia.
    Presentase pada grafik mewakili **presentase dari seluruh penduduk yang terdapat pada daerah tersebut**.
    Dapat dilihat bahwa presentase penduduk **pedesaan lebih tinggi dari perkotaan**.
    Presentase penduduk perkotaan yang memiliki minat menonton seni pewayangan tidak sebanyak di pedesaan kemungkinan disebabkan oleh **modernisasi akibat teknologi**.
    Ditambah dengan **perkembangan internet** yang lebih pesat di daerah kota membuat turunnya minat masyarakat terhadap seni pewayangan.
    Di sisi lain, daerah pedesaan **masih kental dengan budaya dan tradisi** yang membuat presentase peminat seni pewayangannya lebih banyak dibanding daerah perkotaan.
    '''

# ===================================================================
col1_cont3, col2_cont3 = st.columns([1, 2])

with col1_cont3:
    st.subheader('Berdasarkan Jenis Kelamin')
    '''
    Meskipun minat masyarakat terhadap seni pewayangan menurun dari tahun ke tahun, presentase peminat laki-laki selalu lebih banyak dibandingkan perempuan.
    Melihat total jumlah penduduk laki-laki dan perempuan di Indonesia tidak berbeda jauh, maka dapat dikatakan secara absolut jumlah peminat laki-laki selalu lebih banyak dibandingkan perempuan terhadap seni pewayangan.
    '''

with col2_cont3:
    fig = px.bar(
        df_gender,
        x='Tahun',
        y='Presentase Peminat',
        color='Jenis Kelamin',
        text_auto=True
    )

    fig.update_layout(
        width=960,
        height=480,
        yaxis_title='Presentase peminat (%)',
        barmode='group'
    )

    st.plotly_chart(fig)

# ===================================================================
col1_cont4, col2_cont4 = st.columns([2, 1])

with col1_cont4:
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

    fig.update_layout(
        width=900,
        height=480,
        yaxis_title='Presentase peminat (%)'
    )

    st.plotly_chart(fig)

with col2_cont4:
    st.subheader('Distribusi Umur')
    '''
    Distribusi umur dari peminat seni pewayangan selalu sama dari tahun ke tahun dan didominasi oleh penduduk dengan umur **diatas 40 tahun**.
    Lansia memiliki presentase terbanyak setiap tahunnya sebagai peminat seni pewayangan.
    Hal ini kemungkinan terjadi karena generasi yang lebih tua sulit beradaptasi dengan perkembangan teknologi yang menyebabkan mereka lebih memilih untuk menikmati hiburan tradisional seperti wayang.
    Jika dibandingkan dari tahun ke tahun pula, jumlah peminat seni pewayangan terus menurun di berbagai kalangan usia.
    Kecuali di tahun 2015 terjadi sedikit peningkatan pada peminat seni pewayangan di semua kelompok umur dibandingkan tahun 2012. 
    '''

# ===================================================================
st.subheader('Sumber Referensi')
'''
https://www.bps.go.id
'''
