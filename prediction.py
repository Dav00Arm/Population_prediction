#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pmdarima as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

wp = pd.read_excel('World population database.xls')
wp = wp.set_index(['Country Name'])
wp = wp.drop(['Indicator Code', 'Country Code', 'Indicator Name'],axis=1)

countries = wp.drop(['Africa Eastern and Southern', 'Africa Western and Central', 'Arab World',
'Central Europe and the Baltics', 'Caribbean small states', 'East Asia & Pacific (excluding high income)',
'Early-demographic dividend','East Asia & Pacific',
'Europe & Central Asia (excluding high income)','Europe & Central Asia', 'Euro area', 
'European Union','Fragile and conflict affected situations','Heavily indebted poor countries (HIPC)',
'IBRD only','IDA & IBRD total','IDA total','IDA blend','IDA only','Not classified',
'Latin America & Caribbean (excluding high income)','Least developed countries: UN classification',
'Low income','Liechtenstein','Sri Lanka','Lower middle income','Low & middle income','Lesotho',
'Late-demographic dividend','Middle East & North Africa','Middle income','Middle East & North Africa (excluding high income)',
'OECD members','Other small states','Pre-demographic dividend','Pacific island small states','Post-demographic dividend',
'South Asia','Sub-Saharan Africa (excluding high income)','Sub-Saharan Africa','East Asia & Pacific (IDA & IBRD countries)',
'Europe & Central Asia (IDA & IBRD countries)','Latin America & the Caribbean (IDA & IBRD countries)',
'Middle East & North Africa (IDA & IBRD countries)','South Asia (IDA & IBRD)','Sub-Saharan Africa (IDA & IBRD countries)',
'Upper middle income','North America'])

correction = {'Congo, Dem. Rep.': 'Dem. Rep. Congo',
        'Congo, Dem.':'Dem. Congo',
        'Egypt, Arab Rep.': 'Arab Rep. Egypt',
        'Micronesia, Fed. Sts.': 'Fed. Sts. Micronesia',
        'Korea, Rep.': 'Rep. Korea',
        "Korea, Dem. People's Rep.": 'North Korea',
        'Venezuela, RB': 'Venezuela',
        'Yemen, Rep.': 'Rep. Yemen'}

countries = countries.rename(index=correction)
countries.loc['Eritrea'] = countries.loc['Eritrea'].fillna(countries.loc['Eritrea','1969'])
countries.loc['West Bank and Gaza'] = countries.loc['West Bank and Gaza'].fillna(countries.loc['West Bank and Gaza','1990'])
kuwait_avg = (countries.loc['Kuwait','1991'] + countries.loc['Kuwait','1995'])/2
countries.loc['Kuwait'] = countries.loc['Kuwait'].fillna(kuwait_avg)

countries_transpose = countries.T

countries_transpose.index = pd.to_datetime(countries_transpose.index).year

def prediction(country, n_periods=10):
    
    model = pm.auto_arima(countries_transpose[country], start_p=1, start_q=1,
                      test='adf',       
                      max_p=4, max_q=4, m=1,
                      seasonal=False,   
                      start_P=0, 
                      D=0, 
                      trace=False,
                      error_action='ignore',  
                      suppress_warnings=True)
    
    forecast, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    forecast_id = pd.date_range(start='2021/01/01', periods=n_periods,freq='YS').year
    forecast_series = pd.Series(forecast.round(decimals=0), index=forecast_id)
    lower_series = pd.Series(confint[:, 0].round(decimals=0), index=forecast_id)
    upper_series = pd.Series(confint[:, 1].round(decimals=0), index=forecast_id)

    plt.figure(figsize=(10,8))
    plt.plot(countries_transpose[country])
    plt.plot(forecast_series)
    plt.fill_between(lower_series.index, 
                     lower_series, 
                     upper_series, 
                     color='k', alpha=.15)

    plt.title("Final Forecast")
    plt.show()

    preds = pd.concat({'forecast':forecast_series,'lower border': lower_series, 'upper border': upper_series}, axis=1)
    print('------------------------------------------------------------')
    print('============================================================')
    print('---------------------FINAL FORECAST-------------------------')
    print('============================================================')
    print('------------------------------------------------------------')
    print(preds.astype('int'))
    
country = input("Enter country: ")

if __name__ == '__main__':
    prediction(country)






