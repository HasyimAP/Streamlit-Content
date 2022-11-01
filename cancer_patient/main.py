import os
import math
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import statsmodels.api as smapi
import matplotlib.pyplot as plt
import statsmodels.graphics.gofplots as sm

from func import *
from PIL import Image
from scipy import stats
from func import grouped_boxplot
from statsmodels.formula.api import ols

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
- [Parametric Test](#parametric-tests)
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
        sns.heatmap(df_2.corr(method='pearson'), linewidths=0.1, center=0, annot=True)
        st.pyplot(fig)

    with tab_2:
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(df_2.corr(method='kendall'), linewidths=0.1, center=0, annot=True)
        st.pyplot(fig)

    with tab_3:
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(df_2.corr(method='spearman'), linewidths=0.1, center=0, annot=True)
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

dv_1, dv_2, dv_3, dv_4, dv_5, dv_6, dv_7, dv_8 = st.tabs([
    'Dataframe', #1
    'Statistics', #2
    'Histogram', #3
    'Bar chart', #4
    'Boxplot', #5
    'Scatter plot', #6
    'Q-Q plot', #7
    'P-P plot' #8
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

with dv_7:
    qq_col = st.selectbox(
        'Please choose a column',
        (fix_df.columns.tolist())
    )

    fig = sm.ProbPlot(fix_df[qq_col]).qqplot(line='s')

    st.plotly_chart(fig)

with dv_8:
    pp_col = st.selectbox(
        'Please choose a column ',
        (fix_df.columns.tolist())
    )

    fig = sm.ProbPlot(fix_df[pp_col]).ppplot(line='s')

    st.plotly_chart(fig)

# normality test
st.header('Tests of Normality')
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

# =========================================================
# parametric test
st.header('Parametric tests')
'''
Before we do the parametric test, we should choose how our sample created. The sample will be created according to these 3 parameters below.
'''

sample_df = fix_df.copy()

param_1, param_2 = st.columns(2)

with param_1:
    age_size = st.slider(
        'Age range:',
        int(sample_df['Age'].unique().min()),
        int(sample_df['Age'].unique().max()),
        (int(sample_df['Age'].unique().min()), int(sample_df['Age'].unique().max()))
    )

sample_df = sample_df.query('Age >= @age_size[0] and Age <= @age_size[1]')

with param_2:
    level_size = st.multiselect(
        'Level of the cancer:',
        options=sample_df['Level'].unique(),
        default=sample_df['Level'].unique()
    )

sample_df = sample_df.query('Level == @level_size')

sample_size = st.slider('Choose sample size:', 0, sample_df.shape[0], int(0.2*sample_df.shape[0]))

sample_1df = sample_df.sample(sample_size)

pt_1, pt_2, pt_3, pt_4, pt_5, pt_6 = st.tabs([
    'Sample',
    'One sample T-Test',
    'Independent sample T-Test',
    'Paired-samples T-Test',
    'One-way ANOVA',
    'Two-way ANOVA'
])

with pt_1:
    st.dataframe(sample_1df)

with pt_2:
    '''
    **Hypothesis**

    *Null hypothesis **H0***: The mean value of the population is equal to the specified value of the sample

    *Alternative hypothesis **H1***: The mean value of the population is different from the specified values of the sample

    We can reject H0 in favor of H1 if p-value less than 0.05
    '''
    TTest_1samp = pd.DataFrame(
        index=sample_1df.columns.tolist(),
        columns=['t-statistic', 'p-value']
    )

    TTest_1samp['t-statistic'] = stats.ttest_1samp(fix_df, sample_1df.mean(), alternative='less').statistic
    TTest_1samp['p-value'] = stats.ttest_1samp(fix_df, sample_1df.mean(), alternative='less').pvalue

    st.dataframe(TTest_1samp.T)

with pt_3:
    '''
    *Null hypothesis **H0***: The mean value of the two independent groups is equal
    
    *Alternative hypothesis **H1***: The mean values of the two independent groups are different

    We will create 2 samples with the same parameters that have been specified above, but rest asured because the sample will not be exactly the same to each other.

    We can reject H0 in favor of H1 if p-value less than 0.05
    '''

    sample_2df = sample_df.sample(2*sample_size)
    sample_3df = sample_2df.sample(sample_size)
    sample_2df = sample_2df.drop(sample_3df.index)

    TTest_ind = pd.DataFrame(
        index=sample_df.columns.tolist(),
        columns=['t-statistic', 'p-value']
    )

    TTest_ind['t-statistic'] = stats.ttest_ind(sample_3df, sample_2df, alternative='two-sided').statistic
    TTest_ind['p-value'] = stats.ttest_ind(sample_3df, sample_2df, alternative='two-sided').pvalue

    st.dataframe(TTest_ind.T)

with pt_4:
    '''
    *Null hypothesis **H0***: The mean value of the two dependent groups is equal
    
    *Alternative hypothesis **H1***: The mean values of the two dependent groups are different

    Paired-samples T-Test are not suitable to do for this dataset. The reason why is we don't have any dependent samples from this dataset because the data only measured once.

    Example cases to use paired-samples T-Test:

    We want to compare the test score of students before and after taking classes. Sample 1 is the test scores of the students before taking classes. And sample 2 is the test scores of the students after taking classes. We can use paired-samples T-Test on sample 1 and 2 because those 2 samples came from the same subject, but from different time. 
    '''

with pt_5:
    '''
    *Null hypothesis **H0***: There are no significant differences between the means of the individual groups
    
    *Alternative hypothesis **H1***: At least two group means are significantly different from each other

    There are a total of 3 groups divided by its cancer's level. Group 1 with low level cancer, group 2 with medium level cancer, and group 3 with high level cancer.

    We can reject H0 in favor of H1 if p-value less than 0.05
    '''

    low_df = fix_df[fix_df['Level'] == 1].drop('Level', axis=1)
    mid_df = fix_df[fix_df['Level'] == 2].drop('Level', axis=1)
    high_df = fix_df[fix_df['Level'] == 3].drop('Level', axis=1)

    oneway_ANOVA = pd.DataFrame(
        index=mid_df.columns.tolist(),
        columns=['statistic', 'p-value']
    )

    oneway_ANOVA['statistic'] = stats.f_oneway(low_df, mid_df, high_df).statistic
    oneway_ANOVA['p-value'] = stats.f_oneway(low_df, mid_df, high_df).pvalue

    st.dataframe(oneway_ANOVA.T)

with pt_6:
    '''
    How other variable affect the Level of the cancer?

    *Null hypothesis **H0***: There is a significant effect of the variable to the level of cancer
    
    *Alternative hypothesis **H1***: There is no significant effect of the variable to the level of cancer

    Accept H0 if p-value is less than 0.05.

    *Note*: we don't do any sampling for this test. We directly use the dataset from after the data cleaning process.
    '''
        
    temp_df = fix_df.copy()
    temp_df.columns = temp_df.columns.str.replace(' ', '_')

    string_formula = 'Level ~ '
    col_list = temp_df.columns.tolist()
    col_list.remove('Level')
    for x in col_list:
        string_formula += f'+ C({x})'
    
    model = ols(string_formula, data=temp_df).fit()
    twoway_ANOVA = smapi.stats.anova_lm(model, typ=2)

    st.dataframe(twoway_ANOVA)

# =========================================================
st.header('Non-Parametric Test')

npt_1, npt_2, npt_3, npt_4 = st.tabs([
    'Kruskal-Wallis Test',
    'Wilcoxon-Test',
    'Mann-Whitney U test',
    'Friedman Test'
])

with npt_1:
    '''
    There are a total of 3 groups that divided by the level of patient cancer. Group 1 with low level cancer, group 2 with medium level cancer, and group 3 with high level cancer. From the table below we can see that all of the p-value from all variables are lower than 5%. This means that there are no significant differences in symptoms between the patient with low, medium, and high level cancer. So for the patients to know further about their cancer level it is advisable to ask professional (doctors) and not relying on how severe the symptoms are.
    '''

    low_df = fix_df[fix_df['Level'] == 1].copy()
    mid_df = fix_df[fix_df['Level'] == 2].copy()
    high_df = fix_df[fix_df['Level'] == 3].copy()

    low_df = low_df[low_df['Level'] == 1].drop('Level', axis=1)
    mid_df = mid_df[mid_df['Level'] == 2].drop('Level', axis=1)
    high_df = high_df[high_df['Level'] == 3].drop('Level', axis=1)

    kw_test = pd.DataFrame(
        index=low_df.columns.tolist(),
        columns=['statistic', 'p-value']
    )

    kw_test['statistic'] = stats.kruskal(low_df, mid_df, high_df).statistic
    kw_test['p-value'] = stats.kruskal(low_df, mid_df, high_df).pvalue

    st.dataframe(kw_test.T)

with npt_3:

    mw1_sample = fix_df.copy()
    mw2_sample = fix_df.copy()

    npt3_col_1, npt3_col_2 = st.columns(2)

    with npt3_col_1:
        age1_size = st.slider(
            'Sample 1 age range:',
            int(mw1_sample['Age'].unique().min()),
            int(mw1_sample['Age'].unique().max()),
            (int(mw1_sample['Age'].unique().min()), int(mw1_sample['Age'].unique().max()))
        )

        mw1_sample = mw1_sample.query('Age >= @age1_size[0] and Age <= @age1_size[1]')

        level1_size = st.multiselect(
            'Sample 1 level of the cancer:',
            options=mw1_sample['Level'].unique(),
            default=mw1_sample['Level'].unique()
        )

        mw1_sample = mw1_sample.query('Level == @level1_size')

        sample1_size = st.slider('Choose sample 1 size:', 0, mw1_sample.shape[0], int(0.2*mw1_sample.shape[0]))

        mw1_sample = mw1_sample.sample(sample1_size)

    with npt3_col_2:
        age2_size = st.slider(
            'Sample 2 age range:',
            int(mw2_sample['Age'].unique().min()),
            int(mw2_sample['Age'].unique().max()),
            (int(mw2_sample['Age'].unique().min()), int(mw2_sample['Age'].unique().max()))
        )

        mw2_sample = mw2_sample.query('Age >= @age2_size[0] and Age <= @age2_size[1]')

        level2_size = st.multiselect(
            'Sample 2 level of the cancer:',
            options=mw2_sample['Level'].unique(),
            default=mw2_sample['Level'].unique()
        )

        mw2_sample = mw2_sample.query('Level == @level1_size')

        sample2_size = st.slider('Choose sample 2 size:', 0, mw2_sample.shape[0], int(0.2*mw2_sample.shape[0]))

        mw2_sample = mw2_sample.sample(sample2_size)
    
    mw_test = pd.DataFrame(
        index=mw1_sample.columns.tolist(),
        columns=['statistic', 'p-value']
    )

    mw_test['statistic'] = stats.mannwhitneyu(mw1_sample, mw2_sample, alternative='two-sided').statistic
    mw_test['p-value'] = stats.mannwhitneyu(mw1_sample, mw2_sample, alternative='two-sided').pvalue

    st.dataframe(mw_test.T)

# =========================================================
# conclusion
st.header('Conclusion')
'''
The histogram, skewness, and kurtosis shows that all the distribution of all variables from the dataset is approximately close to normal distribution. BUT, according to the result of the normality tests using Kolmogrov-Smirnov test and Shapiro-Wilk test shows that all variables from the dataset are NOT close/similar to normal distribution.

Current answer *why normality test says data is not normal, but the other analysis says otherwise*:

Looking at a non-significant p-value for any test doesn't lend support to the null hypothesis. In this case, a non-significant SW doesn't show normality-- it just means the sample **doesn't have enough information** to suggest stronger incompatibility with normality which may be **due to sample size or just due to the actual distribution (or some kind of bias)**.

Relying too much on formal tests of normality can lead you astray, as they are often very powerful to detect even the slightest variation from normality which may have zero practical importance. Using subject matter expertise is often useful in statistics, and this is one where **blindly following a p-value is likely to lead you astray**.
'''