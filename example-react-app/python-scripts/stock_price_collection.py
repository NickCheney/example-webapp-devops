#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
from datetime import date, timedelta
import matplotlib as plt
import pandas as pd
import numpy as np
#import yfinance as yf
import finnhub


# In[ ]:


#f = open('symbols.txt')
f = open('S+P_Symbols.txt')
symbols = f.read().split(',')
f.close()
print(len(symbols))


# In[ ]:


date = np.array([],dtype='datetime64')
symbol = np.array([],dtype='object')
close = np.array([])


# In[ ]:


finnhub_client = finnhub.Client(api_key="c3o7claad3ia07uemr30")


# In[ ]:


for i in range(len(symbols)):
    #stock = yf.Ticker(symbols[i])
    #hist = stock.history(period='5y')['Close']
    
    res = finnhub_client.stock_candles(symbols[i], 'D', start, now)
    hist = pd.DataFrame(res)
    
    
    
    #print(i,'/',len(symbols))
    
    #dates = hist.index.to_series().to_numpy(dtype='datetime64')
    deltas = pd.Series(pd.to_timedelta((hist.t / 86400).astype(int),unit='days'))
    dates = pd.Series(pd.Timestamp('1970-01-01')).repeat(len(deltas)).reset_index()[0] + deltas
    dates = dates.values.astype('datetime64[D]')
    
    date = np.concatenate([date,dates])
    
    l = len(dates)
    symbol = np.concatenate([symbol,np.array(l*[symbols[i]])])
    
    close = np.concatenate([close,hist.c.to_numpy()])
    
    time.sleep(1.0)
    


# In[131]:


records = pd.DataFrame({'Date':date,'Symbol':symbol,'Closing Price':close})
records = records.drop_duplicates()
records.Symbol.value_counts()


# In[132]:


accept = records.Symbol.value_counts() == records.Symbol.value_counts().mode()[0]

records = records.loc[records.Symbol.isin(accept.loc[accept].index)]
records


# In[133]:


records.to_csv('s+p_500_records.csv',index=False)


# In[ ]:




