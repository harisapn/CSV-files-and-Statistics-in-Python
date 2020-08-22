#!/usr/bin/env python
# coding: utf-8

# In[7]:


import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd

x = [8.0, 1, 2.5, 4, 28.0]
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]
x #without NaN value

x_with_nan # with NaN value
#from lists to arrays, series and DataFrame
y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
y
y_with_nan
z
z_with_nan


# In[21]:


#calculating mean of x
mean_=sum(x)/len(x)
mean_
#or built in function
statistics.mean(x)

#for data with nan alues
statistics.mean(x_with_nan)
np.mean(x_with_nan)

np.mean(y_with_nan) #functions
y_with_nan.mean() #method
# in order to ignore a nan value use .nanmean()
np.nanmean(y_with_nan)
#series object with mean
z.mean()


# In[31]:


#CALCULATING WEIGHTED MEAN
x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15] #weights

weighted_mean=sum(w[i]*x[i] for i in range(len(x)))/sum(w)
weighted_mean

#built-in function of numpy or pandas series
y,z,w=np.array(x), pd.Series(x), np.array(w)
wmean=np.average(x,weights=w)
wmean
np.average(z,weights=w)


# In[36]:


#CALCULATING HARMONIC MEAN
hmean=len(x)/sum(1/item for item in x)
hmean
#using bulit in function
hmean=statistics.harmonic_mean(x)
statistics.harmonic_mean(x_with_nan)
statistics.harmonic_mean([1, 0, 2])
statistics.harmonic_mean([1, 2, -2]) #error with negative numbers


# In[48]:


#CALCULATING GEOMETRIC MEAN
n=len(x)
product=1
for i in range(len(x)):
 product*=x[i]
gmean=product**(1/n)
gmean


# In[51]:


statistics.median(x[:-1])
statistics.median_low(x[:-1])
statistics.median_high(x[:-1])
statistics.median(x_with_nan)
np.median(y) #with numpy
np.nanmedian(y_with_nan)
z.median() #with series


# In[59]:


#MODE
u = [2, 3, 2, 8, 12]
mode_=max((u.count(item),item) for item in set(u))[1]
mode_
statistics.mode(u)
u=np.array(u)
scipy.stats.mode(u)#with numpy
s=pd.Series(u) #with pandas series
s.mode()


# In[69]:


#VARIANCE
n=len(x)
mean_=statistics.mean(x)
#sample varinace
var_=sum((item-mean_)**2 for item in x)/(n-1)
var_
var_ = statistics.variance(x) #using built in 
var_ = np.var(y, ddof=1) #with numpy
#population variance
pvar_=statistics.variance(x)
pvar_


# In[73]:


#STANDARD DEVIATION
std_=var_**0.5
std_
statistics.stdev(x)
np.std(np.array(x)) #using with numpy


# In[75]:


#SKEWNESS

skew_ = (sum((item - mean_)**3 for item in x)
         * n / ((n - 1) * (n - 2) * std_**3))
skew_


# In[78]:


#PERCENTILES
x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
statistics.quantiles(x, n=2)

statistics.quantiles(x, n=4, method='inclusive')


# In[79]:


y = np.array(x)
np.percentile(y, 5)
np.percentile(y, 95)


# In[80]:


#ranges of data
np.ptp(z)
np.ptp(y_with_nan)
np.ptp(z_with_nan)
z.kurtosis()


# In[83]:


#MEASURES OF CORRELATION
x=list(range(-10,11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_,y_=np.array(x),np.array(y)
x__,y__=pd.Series(x_),pd.Series(y_)

#COVARIANCe
n = len(x)
mean_x, mean_y = sum(x) / n, sum(y) / n
cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n))
          / (n - 1))
cov_xy

cov_matrix=np.cov(x_,y_) #with  numpy
cov_matrix
cov_xy = x__.cov(y__) #with scipy
cov_xy = y__.cov(x__)


# In[85]:


#CORRELATION COEFFICIENT R
var_x = sum((item - mean_x)**2 for item in x) / (n - 1)
var_y = sum((item - mean_y)**2 for item in y) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
r

r, p = scipy.stats.pearsonr(x_, y_) #BULIT-IN FNCTIO
r
#correlation coefficient matrix
corr_matrix = np.corrcoef(x_, y_)
corr_matrix



# In[100]:


#WORKING WITH 2D DATA
a=np.array([[1,2,3],[4,5,6],[7,8,9],[11,12,13]])
np.mean(a)
np.median(a)
np.var(a)
np.mean(a,axis=0) #rows
np.mean(a,axis=1) #calomuns
a.mean(axis=1)
np.median(a, axis=0)
np.median(a, axis=1)
a.var(axis=0, ddof=1)
a.var(axis=1, ddof=1)
a


# In[124]:


#WORKING WITH DATAFRAMES
aDF=pd.DataFrame(a,index=["first","second","third",],columns=["a","b","c"])
aDF
aDF.mean()
aDF.var()
aDF.mean(axis=1)
aDF["a"].mean()
aDF.describe()
aDF.info()
aDF.to_numpy()


# In[132]:


#VISUALIZING DATA
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#get random numbers
np.random.seed(seed=0)
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)
fig, ax = plt.subplots()
#boxplot
ax.boxplot((x, y, z),vert=False, showmeans=True, meanline=True,
           labels=('x', 'y', 'z'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})
plt.show()

#histogram
hist, bin_edges = np.histogram(x, bins=10)
hist

bin_edges


# In[136]:


#histogram
hist,bin_edges=np.histogram(x,bins=10)
hist 
bin_edges


# In[134]:


fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()


# In[137]:


#PIE CHARTS
x,y,z=128,256,1028
fif,ax=plt.subplots()
ax.pie((x,y,z),labels=("x","y","z"),autopct='%1.1f%%')
plt.show()


# In[142]:


#BAR CHARTS
x = np.arange(21)
y = np.random.randint(21, size=21)
err = np.random.randn(21)
fig, ax=plt.subplots()
ax.bar(x,y,yerr=err)
plt.show()


# In[146]:


#X ANS Y PLOTS
x=np.arange(21)
y=5+2*x+2*np.random.randn(21)
slope, intercept, r, *__ = scipy.stats.linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
fig,ax=plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x,intercept + slope * x,label=line)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(facecolor='white')
plt.show()


# In[156]:


#HEAT MAP
#Covariance
matrix=np.cov(x,y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()
#Correlation coefficient
matrix = np.corrcoef(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()


# In[ ]:




