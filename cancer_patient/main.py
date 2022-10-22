import os
import math
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from func import *
from PIL import Image
from scipy import stats

# dataset
path = os.path.dirname(__file__)
df = pd.read_excel(path + '/Group E. cancer patient data sets.xlsx')

# page title
icon = Image.open(path + '/vaccine.png')

st.set_page_config(
    page_title='Cancer Patient',
    page_icon=icon,
    layout='wide'
)

# content
# =========================================================
st.title('Cancer Patients Data')
st.header('Table of Contents')
'''
- [About The Dataset](#about-the-dataset)
- [Data Cleaning](#data-cleaning)
    - Finding correlation between variable
    - Handling outliers
    - Handling missing data
- [Data Visualization](#data-visualization)
- [Conclusion](#conclusion)
'''

# show raw dataset
st.header('About The Dataset')
'''
This dataset consist of 1000 rows data with 25 columns. 3 columns consist of the patient identity (Patient Id, Age, Gender), 21 columns is the patient's symptoms, and 1 column is the patient's cancer level.
'''

st.dataframe(df)
with st.expander('Column details'):
    '''
    - Patient Id: the id of each patient. This value is unique to each other
    - Age: the age of the patient
    - Gender: the gender of the patient. 1 for male, 2 for female
    - Air Pollution: level of patients exposed to air pollution
    - Alcohol use: alcohol consumption level of the patient
    - Dust Allergy: the patient's level of allergy to dust
    - OccuPational Hazards: the patient's occupational hazard level
    - Genetic Risk: the patient's level of genetic risk
    - chronic Lung Disease: level of patient's chronic lung disorders
    - Balanced Diet: patient's balanced diet
    - Obesity: whether or not the patient is obese
    - Smoking: level of patient's smoking habit
    - Passive Smoker: level of patients as passive smokers
    - Chest Pain: level patient's chest pain
    - Coughing of Blood: the severity of the patient's coughing up blood
    - Fatigue: how severe is the patient's level of fatigue
    - Weight Loss: how significant is the patient's weight loss
    - Shortness of Breath: level of experience of the patient with shortness of breath
    - Wheezing: how severe is the patient's wheezing
    - Swallowing Difficulty: the patient's level of difficulty swallowing
    - Clubbing of Finger Nails: clubbing patient's finger level
    - Frequent Cold: how often the patient catches a cold.
    - Dry Cough: patient's dry cough level
    - Snoring: the severity of the patient's snoring habit.
    - Level: patient's cancer level
    The higher the level, the more severe the symptoms
    '''

# data cleaning
# =========================================================
st.header('Data Cleaning')
'''
Data cleaning refer to removing/imputing incorrect data from the dataset. Incorrect data can lead to unreliable outcomes and algorithms, even if they seem correct. In the process of the data cleaning there are a few steps that needs to do.
- Finding correlation between columns/variables. By finding the the correlation coefficient we will know which column/variable that is important for our data analysis. As for the less important variable we can choose to us it or not.
- Handling outliers. Outliers can have a big impact for our data statistical analysis by increase the variability of our data, which decreases statistical power.
- Handling missing data. Similar as ouliers, missing data can reduce the statistical power of our data and can produce biased estimates, leading to invalid conclusion.
'''

# =========================================================
st.subheader('Correlation Between Variable Before Data Cleaning')
'''
Finding correlation between column to determine how each column affect to the other columns. By knowing the correlation coefficients we can assess the strength and direction of linear relationships between pairs of columns. There 3 methods are used to find the coefficient correlation and that are Pearson, Kendall, and Spearman.
'''

# make a copy of the dataframe
df_corr = df.copy()

# delete Patient Id column
df_corr = df_corr.drop('Patient Id', axis=1)

# Level column encoding
df_corr['Level'] = df_corr['Level'].replace({'Low': 1, 'Medium': 2, 'High': 3})

with st.expander('Correlation\'s heatmap'):
    tab_1, tab_2, tab_3 = st.tabs(['Pearson', 'Kendall', 'Spearman'])

    with tab_1:
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(df_corr.corr(method='pearson'), linewidths=0.1, center=0)
        st.pyplot(fig)

    with tab_2:
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(df_corr.corr(method='kendall'), linewidths=0.1, center=0)
        st.pyplot(fig)

    with tab_3:
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(df_corr.corr(method='spearman'), linewidths=0.1, center=0)
        st.pyplot(fig)

# =========================================================
st.subheader('Handling Outliers & Missing Data')
'''
Steps taken in this process are:
1. Replace outliers that are an entry mistake. Below here is the outliers that we believe is an entry mistake:
    - 66 -> 6
    - 700 -> 7
    - 222 -> 2
    - 20 -> 2
    - 11 -> 1
2. Replace the outliers that we can't determine as an entry mistake as missing values.
3. Imputation using linear regression on the missing value.
'''

df_1 = df.copy()

# Level column encoding
df_1['Level'] = df_1['Level'].replace({'Low': 1, 'Medium': 2, 'High': 3})

boxplot_1, boxplot_2, boxplot_3 = st.tabs(['Raw dataset', 'Without Outliers', 'Without outliers & after imputation'])

with boxplot_1:
    fig = boxplot_fig(df_1, 'Boxplot using original dataframe')
    st.plotly_chart(fig)

# replace outliers
df_1 = df_1.replace({66: 6,
                     700: 7,
                     222: 2,
                     20: 2,
                     11: 1
                    })

# change nan value to 0 temporary
df_1 = df_1.fillna(0)

# replace other outliers as 0
df_1.loc[:, 'Gender':] = np.where(np.abs(stats.zscore(df_1.loc[:, 'Gender':])) > 3, 0, df_1.loc[:, 'Gender':])

# return 0 value to nan again
df_1 = df_1.replace(0, np.nan)

with boxplot_2:
    fig = boxplot_fig(df_1, 'Boxplot without outliers')
    st.plotly_chart(fig)

df_2 = df_1.copy()

# replace missing value in column Age & Gender with their median
df_2['Age'] = df_2['Age'].fillna(df_2['Age'].median())
df_2['Gender'] = df_2['Gender'].fillna(df_2['Gender'].median())

# replace the other missing value using linear regression
df_2 = imp_miss_data(df_2, ['Patient Id'])

# there are some decimals data, let's change it to integer
df_2[df_2.select_dtypes(exclude=['object']).columns.tolist()] = df_2[df_2.select_dtypes(exclude=['object']).columns.tolist()].apply(np.ceil)

with boxplot_3:
    fig = boxplot_fig(df_2, 'Boxplot without outliers & after imputation')
    st.plotly_chart(fig)


# =========================================================
st.subheader('Correlation Between Variable After Data Cleaning')
'''
Below here is the new coefficient correlation between variables after handling the outliers and missing data. We can try to compare the correlation before and after data cleaning. 
'''

with st.expander('Correlation\'s heatmap'):
    tab_1, tab_2, tab_3, tab_4 = st.tabs(['Pearson', 'Kendall', 'Spearman', 'New dataframe'])

    with tab_1:
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(df_2.corr(method='pearson'), linewidths=0.1, center=0)
        st.pyplot(fig)

    with tab_2:
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(df_2.corr(method='kendall'), linewidths=0.1, center=0)
        st.pyplot(fig)

    with tab_3:
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(df_2.corr(method='spearman'), linewidths=0.1, center=0)
        st.pyplot(fig)

    with tab_4:
        '''
        On the Level column we've done an encoding that makes the value from the raw dataset as follows:
        - Low -> 1
        - Medium -> 2
        - High -> 3
        '''
        st.dataframe(df_2)

# =========================================================
st.subheader('Data Visualization')
'''
Let's do data visualization to simplify the data analysis process.
'''

fix_df = df_2.copy()
fix_df = fix_df.drop('Patient Id', axis=1)

dv_1, dv_2, dv_3, dv_4, dv_5 = st.tabs(['Dataframe', 'Statistics', 'Histogram', 'Bar chart', 'Boxplot'])

with dv_1:
    st.write('Here\'s dataframe that will be used in analyzing process. Please take a note that there\'s no Patient Id column, because that column is irrelevant in this analysis process.')
    st.dataframe(fix_df)

with dv_2:
    stat_desc = fix_df.describe()
    st.dataframe(stat_desc)

    skew_kurt = pd.DataFrame(columns=['Variance', 'Skewness', 'SE of Skewness', 'Kurtosis', 'SE of Kurtosis'])
    
    skew_kurt['Variance'] = fix_df.var()
    
    skew_kurt['Skewness'] = fix_df.skew()

    n = fix_df.shape[0]
    SE_skew = math.sqrt((6*n*(n-1))/((n-2)*(n+1)*(n+3)))
    skew_kurt['SE of Skewness'] = skew_kurt['SE of Skewness'].fillna(SE_skew)

    skew_kurt['Kurtosis'] = fix_df.kurt()
    
    SE_kurt = 2 * SE_skew * math.sqrt((n**2-1)/((n-3)*(n+5)))
    skew_kurt['SE of Kurtosis'] = skew_kurt['SE of Kurtosis'].fillna(SE_kurt)

    st.dataframe(skew_kurt.T)

with dv_3:
    hist_column = st.selectbox(
        'Select column',
        (fix_df.columns.tolist())
    )

    fig = the_histogram(fix_df, hist_column)
    st.plotly_chart(fig)

with dv_4:
    bar_column = st.selectbox(
        'Select column ',
        (fix_df.columns.tolist())
    )

    fig = px.bar(
        fix_df,
        x=fix_df[bar_column].unique(),
        y=fix_df[bar_column].value_counts(),
        text_auto=''
    )

    fig.update_layout(
        xaxis_title=bar_column,
        yaxis_title='Counts',
        width=1280,
        height=480
    )

    st.write('x: ', bar_column)
    st.write('y: how many values of x')

    st.plotly_chart(fig)

with dv_5:
    col_1, col_2 = st.columns(2)

    with col_1:
        box_col_1 = st.selectbox(
            'Select 1st column (x axis)',
            (fix_df.columns.tolist())
        )

    with col_2:
        box_col_2 = st.selectbox(
            'Select 2nd column (y axis)',
            (fix_df.columns.tolist())
        )

    fig = grouped_boxplot(fix_df, box_col_1, box_col_2)

    fig.update_layout(
        width=1280,
        height=480
    )

    st.plotly_chart(fig)

# conclusion
st.header('Conclusion')