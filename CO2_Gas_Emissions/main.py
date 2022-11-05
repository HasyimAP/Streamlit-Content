import os
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.graphics.gofplots as gg

from func import *
from PIL import Image
from scipy import stats

# dataset
path = os.path.dirname(__file__)
df = pd.read_csv(path + '/owid-co2-data.csv')
clean_df = pd.read_csv(path + '/clean-data-mae-sgd.csv')

# page title
icon = Image.open(path + '/co2.png')

st.set_page_config(
    page_title='CO2 Emissions',
    page_icon=icon,
    layout='wide'
)

st.title('CO2 and Greenhouse Gas Emissions')

content_1, content_2 = st.columns(2)

with content_1:
    '''
    Greenhouse gases trap heat and make the planet warmer. Human activities are responsible for almost all of the increase in greenhouse gases in the atmosphere over the last hundred years. Carbon dioxide (CO2) makes up the vast majority of greenhouse gas emissions from the sector, but smaller amounts of methane (CH4) and nitrous oxide (N2O) are also emitted. These gases are released during the combustion of fossil fuels, such as coal, oil, and natural gas, to produce electricity. Greenhouse gas emissions from human activities strengthen the greenhouse effect, contributing to climate change.
    '''

with content_2:
    image = Image.open(path + '/factory.jpg')
    st.image(image, width=240)

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
'''
After we find out that our data is not normally distributed we don't have to do parametric test and can go straight away to non-parametric test.
'''

# =========================================================
st.subheader('Kruskal-Wallis Test')
'''
In Kruskal-Wallis Test we will compare more than 2 countries/areas to check if they have similar distribution or not. We will define our hypothesis as follows:

Null Hypothesis (H0): The data distribution between all samples/groups are **NOT** stochastically the same

Alternative Hypothesis (H0): The data distribution between all samples/groups are stochastically the same

We can reject H0 if the p-value on the variables we observed are less than 5%.
'''

sample_size = st.number_input(
    'Number of samples (max. 10): ',
    min_value=2,
    max_value=10,
    step=1
)

kw_country = st.multiselect(
    'Choose the countries/areas:',
    options=clean_df['country'].unique(),
    max_selections=sample_size
)

kw_dict = []

for country in kw_country:
    kw_dict.append(clean_df[clean_df['country'] == country].drop(['iso_code', 'country'], axis=1))

ks_df_index = clean_df.columns.tolist()
ks_df_index.remove('iso_code')
ks_df_index.remove('country')

kw_df = pd.DataFrame(
    index=ks_df_index,
    columns=['statistic', 'p-value']
)

if sample_size == 2:
    kw_df['statistic'] = stats.kruskal(kw_dict[0], kw_dict[1]).statistic
    kw_df['p-value'] = stats.kruskal(kw_dict[0], kw_dict[1]).pvalue
elif sample_size == 3:
    kw_df['statistic'] = stats.kruskal(kw_dict[0], kw_dict[1], kw_dict[2]).statistic
    kw_df['p-value'] = stats.kruskal(kw_dict[0], kw_dict[1], kw_dict[2]).pvalue
elif sample_size == 4:
    kw_df['statistic'] = stats.kruskal(kw_dict[0], kw_dict[1], kw_dict[2], kw_dict[3]).statistic
    kw_df['p-value'] = stats.kruskal(kw_dict[0], kw_dict[1], kw_dict[2], kw_dict[3]).pvalue
elif sample_size == 5:
    kw_df['statistic'] = stats.kruskal(
        kw_dict[0], 
        kw_dict[1],
        kw_dict[2],
        kw_dict[3],
        kw_dict[4]
    ).statistic
    kw_df['p-value'] = stats.kruskal(
        kw_dict[0], 
        kw_dict[1],
        kw_dict[2],
        kw_dict[3],
        kw_dict[4]
    ).pvalue
elif sample_size == 6:
    kw_df['statistic'] = stats.kruskal(
        kw_dict[0], 
        kw_dict[1],
        kw_dict[2],
        kw_dict[3],
        kw_dict[4], 
        kw_dict[5]
    ).statistic
    kw_df['p-value'] = stats.kruskal(
        kw_dict[0], 
        kw_dict[1],
        kw_dict[2],
        kw_dict[3],
        kw_dict[4], 
        kw_dict[5]
    ).pvalue
elif sample_size == 7:
    kw_df['statistic'] = stats.kruskal(
        kw_dict[0], 
        kw_dict[1],
        kw_dict[2],
        kw_dict[3],
        kw_dict[4], 
        kw_dict[5],
        kw_dict[6]
    ).statistic
    kw_df['p-value'] = stats.kruskal(
        kw_dict[0], 
        kw_dict[1],
        kw_dict[2],
        kw_dict[3],
        kw_dict[4], 
        kw_dict[5],
        kw_dict[6]
    ).pvalue
elif sample_size == 8:
    kw_df['statistic'] = stats.kruskal(
        kw_dict[0], 
        kw_dict[1],
        kw_dict[2],
        kw_dict[3],
        kw_dict[4], 
        kw_dict[5],
        kw_dict[6],
        kw_dict[7]
    ).statistic
    kw_df['p-value'] = stats.kruskal(
        kw_dict[0], 
        kw_dict[1],
        kw_dict[2],
        kw_dict[3],
        kw_dict[4], 
        kw_dict[5],
        kw_dict[6],
        kw_dict[7]
    ).pvalue
elif sample_size == 9:
    kw_df['statistic'] = stats.kruskal(
        kw_dict[0], 
        kw_dict[1],
        kw_dict[2],
        kw_dict[3],
        kw_dict[4], 
        kw_dict[5],
        kw_dict[6],
        kw_dict[7],
        kw_dict[8]
    ).statistic
    kw_df['p-value'] = stats.kruskal(
        kw_dict[0], 
        kw_dict[1],
        kw_dict[2],
        kw_dict[3],
        kw_dict[4], 
        kw_dict[5],
        kw_dict[6],
        kw_dict[7],
        kw_dict[8]
    ).pvalue
elif sample_size == 10:
    kw_df['statistic'] = stats.kruskal(
        kw_dict[0], 
        kw_dict[1],
        kw_dict[2],
        kw_dict[3],
        kw_dict[4], 
        kw_dict[5],
        kw_dict[6],
        kw_dict[7],
        kw_dict[8],
        kw_dict[9]
    ).statistic
    kw_df['p-value'] = stats.kruskal(
        kw_dict[0], 
        kw_dict[1],
        kw_dict[2],
        kw_dict[3],
        kw_dict[4], 
        kw_dict[5],
        kw_dict[6],
        kw_dict[7],
        kw_dict[8],
        kw_dict[9]
    ).pvalue

st.dataframe(kw_df.T)

# =========================================================
st.subheader('Wilcoxon Test')
'''
In Wilcoxon test we need to determine 2 parameter year. These 2 parameters will create 2 sample with different year from all of data regarding the country/area. We then define our hypothesis:

Null Hypothesis (H0): There's difference in data between those 2 years

Alternative Hypothesis (H1): There's NO difference in data between those 2 years

We can reject H0 in favor of H1 if the p-value on the variables we observed are less than 5%.
'''

w_col_1, w_col_2 = st.columns(2)

with w_col_1:
    w_year_1 = st.number_input(
        'Year 1 for Wilcoxon Test',
        min_value=clean_df['year'].min(),
        max_value=clean_df['year'].max(),
        value=clean_df['year'].min(),
        step=1
    )

with w_col_2:
    w_year_2 = st.number_input(
        'Year 2 for Wilcoxon Test',
        min_value=clean_df['year'].min(),
        max_value=clean_df['year'].max(),
        value=clean_df['year'].max(),
        step=1
    )

wt_sample_1 = clean_df[clean_df['year'] == w_year_1].drop(['iso_code', 'country', 'year'], axis=1)
wt_sample_1.reset_index(inplace=True, drop=True)
wt_sample_2 = clean_df[clean_df['year'] == w_year_2].drop(['iso_code', 'country', 'year'], axis=1)
wt_sample_2.reset_index(inplace=True, drop=True)

resize = min(wt_sample_1.shape[0], wt_sample_2.shape[0])
wt_sample_1 = wt_sample_1.sample(resize)
wt_sample_2 = wt_sample_2.sample(resize)

wt_df = pd.DataFrame(
    index=wt_sample_1.columns.tolist(),
    columns=['statistic', 'p-value']
)

wt_df['statistic'], wt_df['p-value'] = stats.wilcoxon(wt_sample_1, wt_sample_2, alternative='two-sided', zero_method='zsplit')

st.dataframe(wt_df.T)

# =========================================================
st.subheader('Mann-Whitney U Test')
'''
In the Mann-Whitney U Test we are going to compare 2 countries/areas and we will see if there are any differents between those 2 countries/areas. We will define our hypothesis as follows:

Null Hypothesis (H0): The data distribution underlying area 1 is **NOT** stochastically less than the distribution underlying area 2.

Alternative Hypothesis (H0): The data distribution underlying area 1 is stochastically less than the distribution underlying area 2.

We can reject H0 in favor of H1 if the p-value on the variables we observed are less than 5%.
'''

mw_side_1, mw_side_2 = st.columns(2)

with mw_side_1:
    mw_country_1 = st.selectbox(
        'Select country/area 1:',
        (clean_df['country'].unique())
    )

with mw_side_2:
    mw_country_2 = st.selectbox(
        'Select country/area 2:',
        (clean_df['country'].unique())
    )

mw_sample_1 = clean_df[clean_df['country'] == mw_country_1].copy()
mw_sample_1 = mw_sample_1.drop('iso_code', axis=1)
mw_sample_1 = mw_sample_1.drop('country', axis=1)

mw_sample_2 = clean_df[clean_df['country'] == mw_country_2].copy()
mw_sample_2 = mw_sample_2.drop('iso_code', axis=1)
mw_sample_2 = mw_sample_2.drop('country', axis=1)

mw_test = pd.DataFrame(
    index=mw_sample_1.columns.tolist(),
    columns=['statistic', 'p-value']
)

mw_test['statistic'] = stats.mannwhitneyu(mw_sample_1, mw_sample_2, alternative='less').statistic
mw_test['p-value'] = stats.mannwhitneyu(mw_sample_1, mw_sample_2, alternative='less').pvalue

st.dataframe(mw_test.T)

# =========================================================
st.subheader('Friedman Test')
'''
Friedman test is similar to wilcoxon, the difference is wilcoxon only use 2 dependent samples and friedman use 3 or more dependent test. 

Null Hyptothesis (H0): There is **NO** significant difference between data of the dependent samples.

Alternative Hyptothesis (H0): There is significant difference between data of the dependent samples.

We can reject H0 in favor of H1 if p-value are less than 5%.
'''

ft_col_1, ft_col_2, ft_col_3 = st.columns(3)

with ft_col_1:
    ft_year_1 = st.number_input(
        'Year 1 for Friedman Test',
        min_value=clean_df['year'].min(),
        max_value=clean_df['year'].max(),
        value=clean_df['year'].min(),
        step=1
    )

with ft_col_2:
    ft_year_2 = st.number_input(
        'Year 2 for Friedman Test',
        min_value=clean_df['year'].min(),
        max_value=clean_df['year'].max(),
        value=int((clean_df['year'].min() + clean_df['year'].max())/2),
        step=1
    )

with ft_col_3:
    ft_year_3 = st.number_input(
        'Year 3 for Friedman Test',
        min_value=clean_df['year'].min(),
        max_value=clean_df['year'].max(),
        value=clean_df['year'].max(),
        step=1
    )


ft_sample_1 = clean_df[clean_df['year'] == ft_year_1].drop(['iso_code', 'year', 'country'], axis=1)
ft_sample_1.reset_index(inplace=True, drop=True)
ft_sample_2 = clean_df[clean_df['year'] == ft_year_2].drop(['iso_code', 'year', 'country'], axis=1)
ft_sample_2.reset_index(inplace=True, drop=True)
ft_sample_3 = clean_df[clean_df['year'] == ft_year_3].drop(['iso_code', 'year', 'country'], axis=1)
ft_sample_3.reset_index(inplace=True, drop=True)

ft_df = pd.DataFrame(
    index=ft_sample_1.columns.tolist(),
    columns=['statistic', 'p-value']
)

resize = min(ft_sample_1.shape[0], ft_sample_2.shape[0], ft_sample_3.shape[0])
ft_sample_1 = ft_sample_1.sample(resize)
ft_sample_2 = ft_sample_2.sample(resize)
ft_sample_3 = ft_sample_3.sample(resize)

for col in ft_sample_1.columns.tolist():
    ft_df.at[col, 'statistic'], ft_df.at[col, 'p-value'] = stats.friedmanchisquare(
        [ft_sample_1.loc[:, col]],
        [ft_sample_2.loc[:, col]],
        [ft_sample_3.loc[:, col]],
    )

# ft_df['statistic'], ft_df['p-value'] = stats.friedmanchisquare(ft_sample_1, ft_sample_2, ft_sample_3)

st.dataframe(ft_df.T)

with st.expander('Non-Parametrics Test Note'):
    '''
    Notice if we refresh the page, you can see that the p-value on the wilcoxon and friedman test changing. The reason behind this is because the size of the data is different between those year. Wilcoxon and friedman can only be done if all the testing data have the same size, if not it will return an error. So to handle this we do sampling according to the smaller data's size and the sampling is done randomly. 
    
    For example: the size of the data are 67 and 71. We will reduce the 2nd data (data's size 71) to 67, and that 67 values will be chosen randomly from the total of that 71 data.
    '''

# =========================================================
st.header('Correlation')
'''
We are going to use Spearman correlation because the data is not normally distributed.
Below here we can see the correlation table plotted on a heatmap from the cleaned dataset.
We can also see correlation heatmap of the raw dataset below so we can compare the correlation between and after data cleaning.
The color orange represents positive correlation while color blue represents negative correlation.
The brighter the color, the closer the correlation coefficient is to value 1 (+1 for light orange, -1 for light blue).
While the darker the color, the closer the correlation coefficient is to value 0.
There are no correlation between variables if the correlation coefficient is 0.

From the correlation heatmap we can make conclusions as follows:
1. We can see better correlation on the cleaned dataset than on the raw dataset.
2. Only flaring_co2_percapita and trade_co2_share have negative correlation with most of the other variables.
3. Variable year and co2_growth_prct have the least correlation with other variables. 
'''

with st.expander('Correlation using cleaned dataset'):
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(clean_df.corr(method='spearman'), linewidths=0.1, center=0)
    st.pyplot(fig)

with st.expander('Correlation using raw dataset'):
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(df.corr(method='spearman'), linewidths=0.1, center=0)
    st.pyplot(fig)

# =========================================================
st.header('Sources & References')
'''
Column descriptions: [https://github.com/owid/co2-data](https://github.com/owid/co2-data)

Data cleaning documentation code: [https://www.kaggle.com/code/hasyimabdillah/data-cleaning-world-co2-emission/notebook](https://www.kaggle.com/code/hasyimabdillah/data-cleaning-world-co2-emission/notebook)

Streamlit code: [https://github.com/HasyimAP/Streamlit-Content/blob/main/CO2_Gas_Emissions/main.py](https://github.com/HasyimAP/Streamlit-Content/blob/main/CO2_Gas_Emissions/main.py)

Full documentation: [https://github.com/HasyimAP/Streamlit-Content/tree/main/CO2_Gas_Emissions](https://github.com/HasyimAP/Streamlit-Content/tree/main/CO2_Gas_Emissions)

Author profile: [https://github.com/HasyimAP](https://github.com/HasyimAP),
[https://www.linkedin.com/in/m-hasyim-abdillah-p-391079237/](https://www.linkedin.com/in/m-hasyim-abdillah-p-391079237/)
'''
