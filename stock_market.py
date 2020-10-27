#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 02:56:08 2020

@author: bhweshgaur1
"""

import resource, sys
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)    
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import SVG
import time
#import plotly.express as px

import streamlit as st
from pandas_datareader import DataReader

from datetime import datetime,date



st.title('Stock Market Analysis and Risk Prediction')
#@st.cache
def get_data(start, end):
    
    data = []
    for stock in tech_list:
        globals()[stock] = DataReader(stock, 'yahoo', start, end)
        #data.append(DataReader(stock, 'yahoo', start, end))
        #return data      

def get_name(value):
    for k,v in companies.items():
        if v == value:
            return k
    
    return value
        

companies = {'Apple':'AAPL','Google':'GOOGL','Microsoft':'MSFT','Nikola':'NKLA','Boeing':'BA',
             'Amazon':'AMZN','Banque Cantonale':'BQCNF','Facebook':'FB','Alibaba':'BABA',
             'Visa':'V','Johnson & Johnson':'JNJ','Walmart':'WMT','Nestle':'NSRGF',
             'JP Morgan':'JPM','Mastercard':'MA','Cisco':'CSCO','PayPal':'PYPL',
             'PepsiCo':'PEP','Toyota':'TM','Tesla':'TSLA','Oracle':'ORCL','Nike':'NKE',
             'IBM':'IBM','Citigroup':'C','Philip Morris':'PM','QUALCOMM':'QCOM',
             'Sony Corp':'SNEJF','Sony Corporation':'SNE','CNB':'CNBW'}

elements = ['Analysis','Risk Prediction']
option = st.sidebar.selectbox('What do you want to see?', elements)

end = date.today()
start_date = st.sidebar.date_input('Start Date', min_value = date(end.year-5,end.month,end.day), max_value=date(end.year,end.month-1,end.day))
end_date = st.sidebar.date_input('End Date', min_value = date(end.year-4,end.month,end.day) , max_value = date(end.year,end.month,end.day))


if end_date<=start_date:
    st.error('Start Date should be less than End Date')
    
else:

    #years = st.sidebar.slider('How old data do you want(in years, from today)?',1,5,1)
             
    st.sidebar.markdown('### Companies')
    
    tech_list = [
        comp_code for comp,comp_code in companies.items()
        if st.sidebar.checkbox(comp)]
    
    if len(tech_list)<2:
        st.error("Please Select atleast two companies")
    
    else:
    
        #tech_list = ['AAPL','GOOGL','MSFT','AMZN']
    
        #end = datetime.now()
        #start = datetime(end.year-years, end.month, end.day)
    
        get_data(start_date, end_date)
        
        company_name = [get_name(i) for i in tech_list]
        company = st.selectbox(
            'Select the Company: ',
            company_name)
        option,'of', company,':'
    
        if option == elements[0]:
            if st.checkbox('Analysis of Data'):
                st.write(company +' Data')
                if st.checkbox('Show Data'):
                    st.write('First 15 data:')
                    st.dataframe(eval(companies[company]).head(10))
                
                st.write('**Closing Price:**')
                st.line_chart(eval(companies[company])['Close'])
                
                st.write('**Volume:**')
                st.line_chart(eval(companies[company])['Volume'])
                    
                MA_day = [10,20]
                for ma in MA_day:
                    column_name = 'MA for %s days' %(str(ma))
                    eval(companies[company])[column_name] = eval(companies[company])['Close'].rolling(ma).mean()
                if st.checkbox('Show Moving Average Data'):
                    st.write('First 15 data:')
                    st.dataframe(eval(companies[company]).head(15))
                            
                st.write('**Show Moving Average Data:**')
                st.line_chart(eval(companies[company])[['Close','MA for 10 days',
                                    'MA for 20 days']])
                
                eval(companies[company])['Daily Return'] = eval(companies[company])['Close'].pct_change()
                st.write('**Percent change of each day:**')
                eval(companies[company])['Daily Return'].plot(figsize=(12,4), legend=True, linestyle='--', marker='o')
                st.pyplot(clear_figure = True)
                    
                st.bar_chart(eval(companies[company])['Daily Return'])
                    
                st.write('Distplot:')
                sns.distplot(eval(companies[company])['Daily Return'].dropna(), bins=100, color='magenta')
                st.pyplot(clear_figure = True)
                        
                
                
                
            # Grab all the closing prices for the tech stock list into one DataFrame
            closingprice_df = DataReader(tech_list, 'yahoo', start_date, end_date)['Close']
            # make a new tech returns DataFrame
            tech_returns = closingprice_df.pct_change()
                
            if st.checkbox('Show Comparison between two companies'):
            
                st.write('***Comparison***')
                if len(tech_list)==2:
                    st.write(get_name(tech_list[0])+' and '+get_name(tech_list[1])+' by jointplot:')
                    sns.jointplot(tech_list[0],tech_list[1],tech_returns, kind='hex',height=8, color='skyblue'
                                 ).set_axis_labels(get_name(tech_list[0]),get_name(tech_list[1]))
                    st.pyplot(clear_figure= True)
                    
                    st.write("Let us use Regression for "+get_name(tech_list[0])+" and "+get_name(tech_list[1])+" graph:" )
                    sns.jointplot(tech_list[0],tech_list[1],tech_returns, kind='reg', size=8, color='skyblue'
                                 ).set_axis_labels(get_name(tech_list[0]),get_name(tech_list[1]))
                    st.pyplot(clear_figure=True)
                    
                else:
                    compare_company = st.multiselect(
                        'Select any two companies: ',
                        company_name)
                    if len(compare_company)==2:
                        st.write(compare_company[0]+' and '+ compare_company[1]+' by jointplot:')
                        sns.jointplot(companies[compare_company[0]],companies[compare_company[1]],tech_returns, 
                                      kind='hex',height=8, color='skyblue').set_axis_labels(compare_company[0],
                                                                                            compare_company[1])
                        st.pyplot(clear_figure= True)
                        
                        st.write("Let us use Regression for"+compare_company[0]+" and "+
                                 compare_company[1]+" graph:" )
                        sns.jointplot(companies[compare_company[0]],companies[compare_company[1]],tech_returns, 
                                      kind='reg', size=8, color='skyblue').set_axis_labels(compare_company[0],
                                                                                            compare_company[1])
                        st.pyplot(clear_figure=True)
                        
                    else:
                        st.error('Please select only/any two companies...')
                        
            
            if st.checkbox('Show Correlations between different companies'):    
                st.write('**Correlation between different Companies:**')
                if st.checkbox('PairGrid'):
                    st.write('PairGrid of Daily Return:')                
                    # Set up the figure by naming it returns_fig, call PairGrid on the DataFrame
                    returns_fig = sns.PairGrid(tech_returns.dropna())
        
                    # Using map_upper we can specify what the upper triangle will look like.
                    returns_fig.map_upper(plt.scatter,color='purple')
                    
                    # We can also define the lower triangle in the figure, including the plot type (kde) & the color map (BluePurple)
                    returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
        
                    # Finally we'll define the diagonal as a series of histogram plots of the daily return
                    returns_fig.map_diag(plt.hist,bins=30)
                    
                    #change the labels of the plot
                    replacements = {code:get_name(code) for code in tech_list}
                    for i in range(len(tech_list)):
                        for j in range(len(tech_list)):
                            xlabel, ylabel = returns_fig.axes[i][j].get_xlabel(),returns_fig.axes[i][j].get_ylabel()
                            if xlabel in replacements.keys():
                                returns_fig.axes[i][j].set_xlabel(replacements[xlabel])
                            if ylabel in replacements.keys():
                                returns_fig.axes[i][j].set_ylabel(replacements[ylabel])
                                
                    st.pyplot()    
            
                    st.write('PairGrid of Closing Price: ')
                    # Set up the figure by naming it returns_fig, call PairGrid on the DataFrame
                    returns_fig = sns.PairGrid(closingprice_df.dropna())
        
                    # Using map_upper we can specify what the upper triangle will look like.
                    returns_fig.map_upper(plt.scatter,color='purple')
            
                    # We can also define the lower triangle in the figure, including the plot type (kde) & the color map (BluePurple)
                    returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
        
                    # Finally we'll define the diagonal as a series of histogram plots of the daily return
                    returns_fig.map_diag(plt.hist,bins=30)
                    
                    #Change the label of the plot
                    for i in range(len(tech_list)):
                        for j in range(len(tech_list)):
                            xlabel, ylabel = returns_fig.axes[i][j].get_xlabel(),returns_fig.axes[i][j].get_ylabel()
                            if xlabel in replacements.keys():
                                returns_fig.axes[i][j].set_xlabel(replacements[xlabel])
                            if ylabel in replacements.keys():
                                returns_fig.axes[i][j].set_ylabel(replacements[ylabel])
                    
                    st.pyplot()
            
                if st.checkbox('HeatMap'):
                    st.write('Correlation by Heatmap:')
                    st.write('--> Daily Returns:')
                    
                    labels = [get_name(i) for i in tech_list]
                    
                    sns.heatmap(tech_returns.corr(),annot = True,fmt = '.2g',cmap='YlGnBu',
                                              xticklabels = labels, yticklabels = labels )                
                    st.pyplot(clear_figure=True)
                    st.write('--> Closing Price:')
                    sns.heatmap(closingprice_df.corr(),annot=True,fmt = '.3g',cmap='YlGnBu',
                                              xticklabels = labels, yticklabels = labels)#,fmt=".3g"
                    st.pyplot(clear_figure=True)
            
            if st.checkbox('Show Risk Analysis graph'):
                rets = tech_returns.dropna()
            
                # Defining the area for the circles of scatter plot to avoid tiny little points
                area = np.pi*20
        
                plt.scatter(rets.mean(),rets.std(),s=area)
        
                # Set the x and y limits of the plot (optional, remove this if you don't see anything in your plot)
                plt.xlim([-0.0025,0.0025])
                plt.ylim([0.001,0.025])
            
                #Set the plot axis titles
                plt.xlabel('Expected returns')
                plt.ylabel('Risk')
            
                # Label the scatter plots, for more info on how this is done, chekc out the link below
                # http://matplotlib.org/users/annotations_guide.html
                for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
                    plt.annotate(
                            get_name(label), 
                            xy = (x, y), xytext = (50, 50),
                            textcoords = 'offset points', ha = 'right', va = 'bottom',
                            arrowprops = dict(arrowstyle = 'fancy', connectionstyle = 'arc3,rad=-0.3'))
                st.write('**Risk Analysis**')
                st.pyplot(clear_figure=True)
            
        if option == elements[1]:
            
            # Grab all the closing prices for the tech stock list into one DataFrame
            closingprice_df = DataReader(tech_list, 'yahoo', start_date, end_date)['Close']
            # make a new tech returns DataFrame
            tech_returns = closingprice_df.pct_change()
            rets = tech_returns.dropna()
            
            if st.checkbox('Quantile Method'):
                st.write('**Quantile Method:**')
                risk = rets[companies[company]].quantile(0.05)
                risk_amount = (-1)*risk * 1000000
                st.write('The 0.05 empirical quantile of daily returns is at ',risk,'.'+'That means that with', 100-((-1)*risk)*100,'% confidence, our worst daily loss will not exceed ', ((-1)*risk)*100,'%.'+ 'If we have a 1 million dollar investment, our one-day 5% VaR is',(-1)*risk,' * 1,000,000 = $',risk_amount)
                    
                
            if st.checkbox('Monte Carlo Method'):
                st.write('**Monte Carlo Method:**')
                
                # Set up our time horizon
                days = 365
        
                # Now our delta
                dt = 1/days
        
                # Now let's grab our mu (drift) from the expected return data we got for GOOGL
                mu = rets.mean()[companies[company]]
        
                # Now let's grab the volatility of the stock from the std() of the average return for GOOGL
                sigma = rets.std()[companies[company]]
                
                def stock_monte_carlo(start_price,days,mu,sigma):
                    ''' This function takes in starting stock price, days of simulation,mu,sigma, and returns simulated price array'''
                    
                    
                    # Define a price array
                    price = np.zeros(days)
                    price[0] = start_price
            
                    # Shock and Drift
                    shock = np.zeros(days)
                    drift = np.zeros(days)
            
                    # Run price array for number of days
                    for x in range(1,days):
                
                        # Calculate Shock
                        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
                        # Calculate Drift
                        drift[x] = mu * dt
                        # Calculate Price
                        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
                
                    return price
                                
                start_price = st.number_input("Enter any amount")
                
                if st.button('Predict'):
                    st.write('**Risk Analysis at**',start_price)
                    for run in range(100):
                        plt.plot(stock_monte_carlo(start_price, days, mu, sigma))
                        
                    plt.xlabel("Days")
                    plt.ylabel("Price")  
                    plt.title('Monte Carlo Analysis for '+company)
                    st.pyplot(clear_figure = True)
                    
                    st.write('**Risk Prediction at**',start_price)
                    # Set a large numebr of runs
                    runs = 10000
            
                    # Create an empty matrix to hold the end price data
                    simulations = np.zeros(runs)
            
                    for run in range(runs):    
                        # Set the simulation data point as the last stock price for that run
                        simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
                        
                    # Now we'll define q as the 1% empirical quantile, this basically means that 99% of the values should fall between here
                    q = np.percentile(simulations,1)
            
                    # Now let's plot the distribution of the end prices
                    plt.hist(simulations, bins=200)
            
                    # Using plt.figtext to fill in some additional information onto the plot
            
                    # starting price
                    plt.figtext(0.6,0.8, s='Start Price: $%.2f' % start_price)
            
                    # mean ending price
                    plt.figtext(0.6,0.7, s='Mean Final Price: $%.2f' % simulations.mean())
            
                    # Variance of the price (within 99% confidence interval)
                    plt.figtext(0.6,0.6, s='VaR(0.99): $%.2f' % (start_price - q))
            
                    # To display 1% quantile
                    plt.figtext(0.15, 0.6, s="q(0.99): $%.2f" % q)
            
                    # Plot a line at the 1% quantile result
                    plt.axvline(x=q, linewidth=4, color='r')
            
                    # For plot title
                    plt.title(label ="Final price distribution for "+company+" after %s days" % days, weight='bold', color='Y')
                    
                    st.pyplot(clear_figure = True)
                    
                    for i in range(2):
                        time.sleep(1)
                        st.balloons()    