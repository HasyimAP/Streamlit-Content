import os
import pandas as pd
import streamlit as st
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt

from PIL import Image

# dataset
path = os.path.dirname(__file__)
data_file = path + '/data wayang.xlsx'

df_region = pd.read_excel(data_file, sheet_name='Tipe daerah')
df_gender = pd.read_excel(data_file, sheet_name='Jenis kelamin')
df_age = pd.read_excel(data_file, sheet_name='Umur')
df_prov = pd.read_excel(data_file, sheet_name='Provinsi')

# page settings
st.set_page_config(
    page_title='Peminat Wayang',
    layout='wide'
)

# ===================================================================
'''
# Wayang Indonesia
###### by Bangkit Flex Team

Data yang digunakan berasal dari situs web Badan Pusat Statistika (BPS) Indonesia.
Data bersumber dari buku katalog Statistik Sosial Budaya yang diterbitkan oleh BPS setiap 3 tahun.
Sehingga data yang tercantum pada *dashboard* analisis memiliki interval setiap 3 tahun.
Berdasarkan katalog, seni pewayangan termasuk ke dalam seni drama/teater/pedalangan.
Jadi pada analisi berikut ini mengikuti parameter yang tercantum pada katalog sumber.
Setiap katalog memiliki nama parameter yang berbeda, tapi memiliki maksud yang sama.  
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
st.subheader('Berdasarkan Provinsi')

df_geo = gpd.read_file(path + '/gadm41_IDN_1.json')

df_join = df_geo.merge(df_prov, how='inner', left_on='NAME_1', right_on='Provinsi')
df_join = df_join[[
    'Provinsi', 
    2009,
    2012,
    2015,
    2018,
    2021, 
    'geometry'
]]

year = st.selectbox(
    'Pilih tahun: ',
    options=df_age['Tahun'].unique().tolist()
)

fig, ax = plt.subplots(1, figsize=(21,7), constrained_layout=True)

ax.axis('off')

sm = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=0, vmax=df_join[year].max()*1.2))

cbar = fig.colorbar(sm, ax=ax)

df_join.plot(
    column=year,
    cmap='OrRd',
    linewidth=1,
    edgecolor='0.8',
    norm=plt.Normalize(vmin=0, vmax=df_join[year].max()*1.2),
    ax=ax,
    missing_kwds={
        'color': 'lightgrey',
        'edgecolor': 'red',
        'hatch': '///',
    }
)

df_join['coords'] = df_join['geometry'].apply(lambda x: x.representative_point().coords[:])
df_join['coords'] = [coords[0] for coords in df_join['coords']]
for idx, row in df_join.iterrows():
    plt.annotate(row['Provinsi'], xy=row['coords'], horizontalalignment='center')

st.pyplot(fig)

'''
Pada grafik dapat terlihat bahwa **semakin gelap warna merah** pada peta menandakan bahwa **semakin tinggi konsentrasi peminat seni pewayangan** pada daerah tersebut.
Melalui skala warna di kanan peta dapat dilihat nilai presentase (%) dari jumlah penduduk yang menonton seni pewayangan pada tahun tersebut.
Dalam setiap interval 3 tahun terlihat konsentrasi peminat seni pewayangan mayoritas berada di Pulau Jawa dan Bali, terutama di Jawa Timur, Yogyakarta, dan Bali.
Selain itu jika diperhatikan lebih detail lagi, ada beberapa daerah yang presentase penduduknya yang menonton seni pewayangan **di bawah 1%**.
Presentase penduduk yang menonton seni pewayangan dari tahun ke tahun juga mengalami **penurunan** yang cukup terlihat jelas.

\*Di tahun 2009 dan 2012 Provinsi Kalimantan Utara diberi garis merah karena pada tahun tersebut **belum** terjadi pemekaran Provinsi Kalimantan Utara.
'''

# ===================================================================
st.subheader('Apa Yang Mungkin Akan Terjadi?')

col1_cont5, col2_cont5 = st.columns([2,1])

with col1_cont5:
    '''
    Masih sulit untuk memprediksi apa yang akan terjadi dalam beberapa tahun ke depan akibat minimnya data yang tersedia.
    Namun, dengan melihat data yang kita miliki saat ini terdapat kemungkinan jumlah peminat seni pewayangan ke depannya akan **semakin berkurang**.
    Meskipun distribusi dari segi umur, jenis kelamin, dan daerah terlihat sama dari tahun ke tahunnya, tapi tidak dapat dipungkiri bahwa **jumlah keseluruhan dari peminat wayang terus menurun**.
    Terutama peminat dari kalangan muda yang berusia di bawah 30 tahun.
    Provinsi Bali masih memiliki jumlah peminat yang lebih banyak dibandingkan daerah lainnya kemungkinan dikarenakan faktor pariwisata yang membuat pendatang tidak hanya tertarik dengan budaya Provinsi Bali, tapi juga budaya daerah lainnya.
    Lalu di Yogyakarta juga memiliki peminat yang cukup tinggi dibandingkan daerah lainnya karena di sana terdapat banyak sekolah serta institusi yang membantu dalam melestarikan budaya Indonesia seperti wayang.
    Namun tidak semua provinsi di Indonesia memiliki keadaan seperti 2 provinsi tersebut, ditambah lagi jumlah serta kepadatan penduduk yang lebih kecil di luar Pulau Jawa.
    '''

with col2_cont5:
    image = Image.open(path + '/wayang_1.jpg')
    st.image(image, width=240)

# ===================================================================
st.subheader('Apa Yang Dapat Kita Lakukan?')

col1_cont6, col2_cont6 = st.columns([1,2])

with col1_cont6:
    image = Image.open(path +'/wayang_2.jpg')
    st.image(image, width=420)

with col2_cont6:
    '''
    Meningkatkan minat masyarakat terhadap seni pewayangan bukanlah hal yang mudah.
    Ada berbagai cara yang dapat dilakukan untuk meningkatkan minat masyarakat terhadap seni pewayangan, tapi tidak semuanya memberikan hasil yang instan dan beberapa membutuhkan waktu.
    Salah satu dari berbagai cara tersebut dengan menggunakan teknologi.
    Dengan memanfaatkan perkembangan teknologi kita dapat membuat masyarakat menjadi lebih dekat dengan seni pewayangan.
    Dan teknologi yang dapat kita gunakan tersebut salah satunya adalah perkembangan machine learning.

    Agar masyarakat mulai mengenal seni pewayangan lebih dekat kita memiliki beberapa opsi pendekatan, seperti jenis wayang, cerita-cerita populer dalam seni pewayangan, tokoh-tokoh dalam seni pewayangan, dan banyak lainnya.
    Kami memilih pendekatan melalui tokoh-tokoh pewayangan.
    Dengan memanfaatkan machine learning, kami mengembangkan aplikasi berbasis web yang dapat membantu mengenali karakter/tokoh wayang.
    Dengan adanya aplikasi ini diharapkan dapat meningkatkan ketertarikan masyarakat terhadap seni pewayangan.
    '''

# ===================================================================
st.subheader('Sumber Referensi')
'''
https://www.bps.go.id
'''
