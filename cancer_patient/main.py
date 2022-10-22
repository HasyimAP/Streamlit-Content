import os
import math
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from func import *
from func import grouped_boxplot
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
content_1, content_2 = st.columns(2)

with content_1:
    '''
    Cancer is a disease in which some of the body\'s cells grow uncontrollably and spread to other parts of the body. Cancer is the second leading cause of death worldwide and is responsible for 16.5% of all deaths. Lung cancer is the most common subtype of cancer that causes death. Some of the biggest risk factors for developing cancer include obesity, tobacco smoking, viral infections, UV radiation overexposure, genetic predispositions and regular alcohol consumption. Symptoms of cancer vary and are location-specific but may include a persistent lump, pain, weight loss, fatigue and persistent coughing. 
    '''

with content_2:
    image = Image.open(path + '/lung cancer.jpg')
    st.image(image, width=240)

st.header('Table of Contents')
'''
- [About The Dataset](#about-the-dataset)
- [Data Cleaning](#data-cleaning)
    - Finding correlation between variable
    - Handling outliers
    - Handling missing data
- [Data Visualization](#data-visualization)
- [Tests of Normality](#tests-of-normality)
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
fix_df = fix_df.astype(int)

dv_1, dv_2, dv_3, dv_4, dv_5, dv_6 = st.tabs([
    'Dataframe', #1
    'Statistics', #2
    'Histogram', #3
    'Bar chart', #4
    'Boxplot', #5
    'Scatter plot' #6
])

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
        x=fix_df[bar_column].value_counts().index,
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

with dv_6:
    col_1, col_2 = st.columns(2)

    col_param = fix_df.columns.tolist()
    col_param.remove('Level')

    with col_1:
        scatter_col_1 = st.selectbox(
            'Select 1st variable (x axis)',
            (col_param)
        )

    with col_2:
        scatter_col_2 = st.selectbox(
            'Select 2nd variable (y axis)',
            (col_param)
        )

    fig = px.scatter(fix_df, x=scatter_col_1, y=scatter_col_2, color='Level', trendline='ols')

    st.plotly_chart(fig)

# normality test
st.subheader('Tests of Normality')
'''
The aim of this week’s report is to test if the data follows a normal distribution.

Null hypothesis: (P>0.05) 

The values are sampled from a population that follows a normal distribution.

Alternative hypothesis: (P<=0.05)

The values are not sampled from a population that follows a normal distribution.
'''

y = [
    ('Kolmogorov-Smirnov', 'Statistic'),
    ('Kolmogorov-Smirnov', 'N'),
    ('Kolmogorov-Smirnov', 'p-value'),
    ('Shapiro-Wilk', 'Statistic'),
    ('Shapiro-Wilk', 'N'),
    ('Shapiro-Wilk', 'p-value')
]
col_list = pd.MultiIndex.from_tuples(y)

norm_test_df = pd.DataFrame(
    index=fix_df.columns.tolist(),
    columns=col_list
)

norm_test_df.loc[:, ('Kolmogorov-Smirnov','N')] = 1000
norm_test_df.loc[:, ('Shapiro-Wilk','N')] = 1000

for i in fix_df.columns.tolist():
    ks_stat = stats.kstest(fix_df[i], stats.norm.cdf, alternative='less').statistic
    ks_p = stats.kstest(fix_df[i], stats.norm.cdf, alternative='less').pvalue
    norm_test_df.at[i, ('Kolmogorov-Smirnov', 'Statistic')] = ks_stat
    norm_test_df.at[i, ('Kolmogorov-Smirnov', 'p-value')] = ks_p

    sw_stat = stats.shapiro(fix_df[i]).statistic
    sw_p = stats.shapiro(fix_df[i]).pvalue
    norm_test_df.at[i, ('Shapiro-Wilk', 'Statistic')] = sw_stat
    norm_test_df.at[i, ('Shapiro-Wilk', 'p-value')] = sw_p
    

st.dataframe(norm_test_df)

# conclusion
st.header('Conclusion')
'''
The histogram, skewness, and kurtosis shows that all the distribution of all variables from the dataset is approximately close to normal distribution. BUT, according to the result of the normality tests using Kolmogrov-Smirnov test and Shapiro-Wilk test shows that all variables from the dataset are NOT close/similar to normal distribution.

Current answer *why normality test says data is not normal, but the other analysis says otherwise*:

Looking at a non-significant p-value for any test doesn't lend support to the null hypothesis. In this case, a non-significant SW doesn't show normality-- it just means the sample **doesn't have enough information** to suggest stronger incompatibility with normality which may be **due to sample size or just due to the actual distribution (or some kind of bias)**.

Relying too much on formal tests of normality can lead you astray, as they are often very powerful to detect even the slightest variation from normality which may have zero practical importance. Using subject matter expertise is often useful in statistics, and this is one where **blindly following a p-value is likely to lead you astray**.
'''