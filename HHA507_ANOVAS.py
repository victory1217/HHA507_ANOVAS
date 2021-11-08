#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 20:58:14 2021

@author: victoria_rodriguez
"""

## Step 1 - Import needed packages 
import pandas as pd 
import seaborn as sns
import scipy.stats as stats
import numpy as np 
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, bartlett
import pingouin as pg

## Step 2 - Import dataframe to be analyzed 
## Note: This dataframe is presenting suicide rates from 1985 to 2016 for multiple countries in the world. 
suicide_rates = pd.read_csv('/Users/victoria_rodriguez/Downloads/suicide.csv')

## Step 3 - Generate a list of columns within the dataframe to identify variables for ANOVA tests 
list(suicide_rates)
"""
'country' 
 'year'
 'sex' - (2 levels: male and female)
 'age' - (6 levels: 5-14yrs, 15-24yrs, 25-34yrs, 35-54yrs, 55-74yrs, 75+yrs)
 'suicides_no' - total number of suicides for information within row 
 'population' = total population for information within row 
 'suicides/100k pop' = total number of suicides per 100,000 people 
 'country-year' = categorial value with country and year 
 ' gdp_for_year ($) ' = total gdp for that year 
 'gdp_per_capita ($)' = total gdp per capita 
 'generation' - (6 levels: Boomers, GI Generation, Generation X, Generation Z, Millenials, Silent)
"""

## Step 4 - Select variables of interest for 1-way ANOVA tests 
"""
Dependent variable (continuous value) = suicides/100k pop
Indepdendent variable 1 (categorical value) = age 
Indepdendent variable 2 (categorical value) = gdp_for_year
Independent variable  3 (categorical value) = sex
Independent variable  4 (categorical value) = generation 
"""
##Renamed columns to avoid white space errors
suicide_rates = suicide_rates.rename(columns={ 'suicides/100k pop' : 'suicide_per_pop'})
suicide_rates = suicide_rates.rename(columns={ ' gdp_for_year ($) ' : 'gdp_per_year'})


## Step 5 - Create visuals to see data distribution and differences between groups 

##Boxplots to see differences and outliers 
suicide_age_boxplot = sns.boxplot(x='age', y= 'suicide_per_pop', data=suicide_rates, palette="Set3")
suicide_gdp_boxplot = sns.boxplot(x='gdp_per_year', y= 'suicide_per_pop', data=suicide_rates, palette="Set3") 
suicide_sex_boxplot = sns.boxplot(x='sex', y= 'suicide_per_pop', data=suicide_rates, palette="Set3") 
suicide_gen_boxplot = sns.boxplot(x='generation', y= 'suicide_per_pop', data=suicide_rates, palette="Set3") 

##Barplots to see distribution and value counts 
suicides_vs_age = sns.barplot(x='age', y= 'suicide_per_pop, data=suicide_rates, palette="Set3") 
suicides_vs_gdp = sns.barplot(x='gdp_per_year', y= 'suicide_per_pop', data=suicide_rates, palette="Set3") 
suicides_vs_sex = sns.barplot(x='sex', y= 'suicide_per_pop', data=suicide_rates, palette="Set3") 
suicides_vs_gen = sns.barplot(x='generation', y= 'suicide_per_pop', data=suicide_rates, palette="Set3") 


## Step 6 - Create a working dataframe where only columns of interest are visible 
workingdf = suicide_rates[['suicide_per_pop', 'age','gdp_per_year','sex', 'generation']]


## Step 7 - Get value counts to determine if the values are unbalanced or balanced 

age_counts = workingdf['age'].value_counts().reset_index()
##Note: all categories for age are balanced except for the 5-14 age group, so ultimately the column is UNBALANCED

gdp_counts = workingdf['gdp_per_year'].value_counts().reset_index()
##Note: all categories have the same value, so ultimately the column is BALANCED

sex_counts = workingdf['sex'].value_counts().reset_index()
##Note: both categories have the same value, so ultimately the column is BALANCED

gen_counts = workingdf['generation'].value_counts().reset_index()
##Note: all categories have different values, so ultimately the column is UNBALANCED

## Step 8 - Perform one-way ANOVA tests 

## First test is trying to figure out if there is a difference between the total number of suicides per 100,000 people and the documented age groups
model = ols('suicide_per_pop ~ C(age)', data=suicide_rates).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table
"""
                sum_sq       df           F  PR(>F)
C(age)    1.405886e+06      5.0  909.788944     0.0
Residual  8.596127e+06  27814.0         NaN     NaN
"""                                  

## Second test is trying to figure out if there is a difference between the total number of suicides per 100,000 people and gdp earned per year
model = ols('suicide_per_pop ~ C(gdp_per_year)', data=suicide_rates).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table
"""
sum_sq       df         F  PR(>F)
C(gdp_per_year)  2.645360e+06   2320.0  3.952207     0.0
Residual         7.356653e+06  25499.0       NaN     NaN
"""

## Third test is trying to figure out if there is a difference between the total number of suicides per 100,000 people and the gender of an individual 
model = ols('suicide_per_pop ~ C(sex)', data=suicide_rates).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table
"""
                sum_sq       df            F  PR(>F)
C(sex)    1.533003e+06      1.0  5035.427899     0.0
Residual  8.469009e+06  27818.0          NaN     NaN
"""

## Fourth test is trying to figure out if there is a difference between the total number of suicides per 100,000 people and the documented generation category
model = ols('suicide_per_pop ~ C(generation)', data=suicide_rates).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table
"""
       sum_sq       df           F  PR(>F)
C(generation)  1.131614e+06      5.0  709.657446     0.0
Residual       8.870398e+06  27814.0         NaN     NaN
"""

## Conclusions for each 1-way ANOVA test

## According to the p-value for the first ANOVA test, there is a significant difference between the number of suicides and the associated age groups.

## According to the p-value for the second ANOVA test, there is a significant difference between the number of suicides and the gdp earned per year.

## According to the p-value for the third ANOVA test, there is a significant difference between the number of suicides and an individual's gender.

## According to the p-value for the fourth ANOVA test, there is a significant difference between the number of suicides and the generation an individual is classified under.


## EXTRA: 2-way ANOVA test practice 
model = ols('suicide_per_pop ~ C(age) + C(sex) + C(age):C(sex)', data=suicide_rates).fit()
anova_table = sm.stats.anova_lm(model, typ=3)
anova_table
"""
     sum_sq       df           F         PR(>F)
Intercept      4.347637e+04      1.0  184.406190   7.195354e-42
C(age)         1.153800e+05      5.0   97.877480  1.275860e-102
C(sex)         9.904366e+04      1.0  420.096323   1.126998e-92
C(age):C(sex)  5.069939e+05      5.0  430.085645   0.000000e+00
Residual       6.556130e+06  27808.0         NaN            NaN
"""

## Conclusion for 2-way ANOVA test 

## As shown previously, there are significant differences between number of suicides for age and sex.
## According to the interaction effect, there seems to be a significant difference between the associated age groups and gender. 