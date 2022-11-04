import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
import statsmodels.graphics.gofplots as gg

from func import *
from PIL import Image
from scipy import stats

# dataset
path = os.path.dirname(__file__)
df = pd.read_csv(path + '/owid-co2-data.csv')
clean_df = pd.read_csv(path + '/clean-data-2.csv')

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

# =========================================================
st.header('Data Cleaning')
'''
The first thing to do is remove all country with all of its population data is missing. 
Greenhouse gas emissions are generated from various industrial sectors. 
And these various industrial sectors are run by humans. 
So it feels strange if in a country or area that does not have data on the population living there to produce greenhouse gas emissions.
We can observe/restore the other country/areas later if we want to analyze it individually, seperated from the others.

We then do imputation on the missing values using Deep Learning.
There are 6 variables that we will impute with care, and those are:
- y: population | x: year
- y: gdp | x: year, population
- y: primary_energy_consumption | x: year, population, gdp
- y: co2 |                           x: year, population, gdp, primary_energy_consumption
- y: cumulative_co2 |                x: year, population, gdp, primary_energy_consumption, co2
y: dependent variable (variable we want to do the imputation on its missing values)

x: independent variable (variable we control to find y)

After there are no more missing values on those 6 variables, we will impute the other variables using these 6 variables as the independent variable. 
We will have this new dataset after cleaning process as follows:
'''

deleted_country = []
for country in df['country'].unique().tolist():
    if country not in clean_df['country'].unique().tolist():
        deleted_country.append(country)
        
string = ', '.join(deleted_country)

'1. From ', df.shape[0], ' rows to ', clean_df.shape[0], ' rows. '
'2. From ', df['country'].nunique(), ' countries/areas to ', clean_df['country'].nunique(), ' countries/areas. Those excluded ', (df['country'].nunique() - clean_df['country'].nunique()), 'countries/areas are ', string
'3. From ', df.isna().sum().sum(), ' missing values to ', clean_df.isna().sum().sum(), 'missing values. The only missing values on the new dataset after cleaning process is only on column iso_code'

with st.expander('Cleaned dataset'):
    st.dataframe(clean_df)

# =========================================================
st.header('Data Visualization')

# =========================================================
st.subheader('Cleaned Dataset Statistic')
st.dataframe(clean_df.describe())

# =========================================================
st.subheader('Cumulative CO2 by Years')
line_df = clean_df.copy()
countries = st.multiselect(
    'Choose countries/areas',
    options=clean_df['country'].unique(),
    default=['World', 'Asia', 'Africa', 'Australia', 'Europe', 'North America', 'South America']
)

line_df = line_df.query(
    'country == @countries'
)

fig = px.line(line_df, x='year', y='cumulative_co2', color='country', width=1280, height=480)
st.plotly_chart(fig)

with st.expander('History Fun Fact'):
    '''
    - 1750, the beginning of first industrial revolution.
    - 1802, the first full-scale working railway steam locomotive built by Trevithick.
    - 1822, The Boston Manufacturing Company starts the first large scale factory town in America.
    - 1823, Samuel Brown patented the first internal combustion engine to be applied industrially in the U.S.
    - 1850, the beginning of second industrial revolution.
    - 1879, Thomas Edison demonstrates the first practical incandescent light bulb.
    - 1882, The world\'s first coal-fired power station, the Holborn Viaduct power station, went into operation in London, England.
    - 1886, Carl Benz applied for a patent for his vehicle powered by a gas engine, the first automobile.
    - 1903, the Wright Brothers make their first successful airplane flight.
    - 1908, Ford begins production of the Model T automobile.
    '''

# =========================================================
st.subheader('Histogram')
'''
Because most of the data is interval/ratio it's hard to visualize them using histogram. We only do histogram representation on year column as this variable is an ordinal data, can be sorted, and have good amount of frequencies on each value. By visualize year column on histogram we can see how our data collected. And we can see below here we have a lot of data from year 1950 and above. There are very small size of data before year 1830, the the size of the data is linearly increasing from 1830 to before 1950, then increased exponentially on year 1950, and then the increase on the graph becomes flatter.
'''

fig = the_histogram(clean_df, 'year')
st.plotly_chart(fig)

with st.expander('Notes'):
    '''
    There are 3 reasons why we only plot year column to the histogram:
    1. Some of the column plotted to the histogram showed a flat line distribution of data.
    2. Some of the column plotted to the histogram showed nothing meaningful, even the histogram looks like a mess. It's like looking at the graph of static noise.
    3. Memory problem. The data too varied making there's a lot of unique data that caused the program to crash.
    '''

# =========================================================
st.subheader('Scatter Plot')
'''
Using the scatter plot we can see the growth on each variable. We can also see the trendline on each variable through the years shown in the red line. The observation done only on the value 'World' on country column. 
'''

scatter_columns = clean_df.columns.tolist()
scatter_columns.remove('iso_code')
scatter_columns.remove('country')
scatter_columns.remove('year')

select_scatter_col = st.selectbox(
    'Choose a variable/column (excl. iso_code, year, & country)',
    (scatter_columns)
)

fig = px.scatter(
    clean_df[clean_df['country'] == 'World'],
    x='year',
    y=select_scatter_col,
    trendline='lowess',
    trendline_color_override='red'
)

fig.update_layout(
    width=1280,
    height=480
)
st.plotly_chart(fig)

# =========================================================
st.subheader('Boxplot')
'''
The data shown here is according to value 'World' on column country from the cleaned dataset. Data with value 'World' on the country column are the sum of the other country, so we can say total from all of the data. So we don't need to sum the data from each different country.
'''

bp_columns = clean_df.columns.tolist()
bp_columns.remove('iso_code')
bp_columns.remove('country')
bp_columns.remove('year')

select_bp_col = st.selectbox(
    'Choose a variable/column (excl. iso_code, country, & year)',
    (bp_columns)
)

fig = single_boxplot(df, select_bp_col)
st.plotly_chart(fig)

# =========================================================

qq_side, pp_side = st.columns(2)

with qq_side:
    st.subheader('Q-Q Plot')
    qq_columns = clean_df.columns.tolist()
    qq_columns.remove('iso_code')
    qq_columns.remove('country')
    qq_columns.remove('year')

    select_qq_col = st.selectbox(
        'Choose a variable/column (excl. iso_code, country, & year) ',
        (qq_columns)
    )

    fig = sm.qqplot(clean_df[clean_df['country'] == 'World'][select_qq_col], line='s')
    st.plotly_chart(fig)

with pp_side:
    st.subheader('P-P Plot')
    pp_columns = clean_df.columns.tolist()
    pp_columns.remove('iso_code')
    pp_columns.remove('country')
    pp_columns.remove('year')

    select_pp_col = st.selectbox(
        'Choose a variable/column (excl. iso_code, country, & year)  ',
        (pp_columns)
    )

    fig = gg.ProbPlot(clean_df[clean_df['country'] == 'World'][select_pp_col]).ppplot(line='s')
    st.plotly_chart(fig)

# =========================================================
st.header('Normality Test')
'''
Null Hypothesis (H0): The data under that variable/column following a normal distribution

Alternative Hypothesis (H1): The data under that variable/column **NOT** following a normal distribution
'''

nt_1, nt_2, nt_3 = st.columns([2,2,3])

col_test = clean_df.columns.tolist()
col_test.remove('iso_code')
col_test.remove('country')

with nt_1:
    ''' #### Kolmogrov-Smirnov Test '''
    ks_df = pd.DataFrame(
        index=col_test,
        columns=['statistic', 'p-value']
    )

    for i in col_test:
        ks_stat = stats.kstest(clean_df[i], stats.norm.cdf, alternative='less').statistic
        ks_p = stats.kstest(clean_df[i], stats.norm.cdf, alternative='less').pvalue
        ks_df.at[i, 'statistic'] = ks_stat
        ks_df.at[i, 'p-value'] = ks_p
    
    st.dataframe(ks_df)

    '''
    Conclusion:

    All p-value from all columns are less 5%, this means we can reject H0 in favor of H1. So the conclusion is **all data from all columns/variables are not normally distributed**.
    '''

with nt_2:
    ''' #### Shapiro-Wilk Test '''
    sw_df = pd.DataFrame(
        index=col_test,
        columns=['statistic', 'p-value']
    )
    
    for i in col_test:
        sw_stat = stats.shapiro(clean_df[i]).statistic
        sw_p = stats.shapiro(clean_df[i]).pvalue
        sw_df.at[i, 'statistic'] = sw_stat
        sw_df.at[i, 'p-value'] = sw_p
    
    st.dataframe(sw_df)

    '''
    Conclusion:

    All p-value from all columns are less 5%, this means we can reject H0 in favor of H1. So the conclusion is **all data from all columns/variables are not normally distributed**.
    '''

with nt_3:
    '''#### Anderson-Darling Test'''
    ad_df = pd.DataFrame(
        index=col_test,
        columns=[
            'statistic',
            'cv (15%)',
            'cv (10%)',
            'cv (5%)',
            'cv (2%)',
            'cv (1%)',
        ]
    )

    for i in col_test:
        ad_stat = stats.anderson(clean_df[i], dist='norm').statistic
        ad_cv = stats.anderson(clean_df[i], dist='norm').critical_values
        ad_df.at[i, 'statistic'] = ad_stat
        ad_df.at[i, 'cv (15%)'] = ad_cv[0]
        ad_df.at[i, 'cv (10%)'] = ad_cv[1]
        ad_df.at[i, 'cv (5%)'] = ad_cv[2]
        ad_df.at[i, 'cv (2%)'] = ad_cv[3]
        ad_df.at[i, 'cv (1%)'] = ad_cv[4]
    
    st.dataframe(ad_df)

    '''
    The column cv on table above stands for critical values and there are a total of 5 significance level (\u03B1) for the critical values (15%, 10%, 5%, 2%, 1%). To determine whether we can reject H0 or not, that is by comparing the statistic value with the critical value on the significance level we want. We can reject H0 if statistic value is greater than critical value.
    
    Conclusion:

    The statisticvalue of all variables is greater than the critical value at all significance level. This means we can reject H0 in favor of H1. So the conclusion is **all data from all columns/variables are not normally distributed**.
    '''

'''
#### Final Conclusion

After doing all possible data visualization (lineplot, histogram, scatterplot, boxplot, Q-Q plot, and P-P plot) and all 3 normality test (Kolmogrov-Smirnov Test, Shapiro-Wilk Test, and Anderson-Darling Test) we can make the final conclusion about the distribution of the data we have. By looking both on the result of data visualization and normality test we can conclude that the data we have is **not normally distributed**. So for the next process we can skip the parametric test and go right away to non-parametric test.
'''

# =========================================================
st.header('Non-Parametric Test')