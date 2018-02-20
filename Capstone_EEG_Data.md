

```python
# Exploratory Analysis of the EEG data; Total features: 178, Class: 1; Instances: 11500
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
# reading the data file
eeg = pd.read_csv('eeg_data_1.csv')
```


```python
# Information on data file
eeg.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11500 entries, 0 to 11499
    Columns: 179 entries, X1 to y
    dtypes: int64(179)
    memory usage: 15.7 MB
    


```python
# First 5 rows
eeg.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>X10</th>
      <th>...</th>
      <th>X170</th>
      <th>X171</th>
      <th>X172</th>
      <th>X173</th>
      <th>X174</th>
      <th>X175</th>
      <th>X176</th>
      <th>X177</th>
      <th>X178</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>135</td>
      <td>190</td>
      <td>229</td>
      <td>223</td>
      <td>192</td>
      <td>125</td>
      <td>55</td>
      <td>-9</td>
      <td>-33</td>
      <td>-38</td>
      <td>...</td>
      <td>-17</td>
      <td>-15</td>
      <td>-31</td>
      <td>-77</td>
      <td>-103</td>
      <td>-127</td>
      <td>-116</td>
      <td>-83</td>
      <td>-51</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>386</td>
      <td>382</td>
      <td>356</td>
      <td>331</td>
      <td>320</td>
      <td>315</td>
      <td>307</td>
      <td>272</td>
      <td>244</td>
      <td>232</td>
      <td>...</td>
      <td>164</td>
      <td>150</td>
      <td>146</td>
      <td>152</td>
      <td>157</td>
      <td>156</td>
      <td>154</td>
      <td>143</td>
      <td>129</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-32</td>
      <td>-39</td>
      <td>-47</td>
      <td>-37</td>
      <td>-32</td>
      <td>-36</td>
      <td>-57</td>
      <td>-73</td>
      <td>-85</td>
      <td>-94</td>
      <td>...</td>
      <td>57</td>
      <td>64</td>
      <td>48</td>
      <td>19</td>
      <td>-12</td>
      <td>-30</td>
      <td>-35</td>
      <td>-35</td>
      <td>-36</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-105</td>
      <td>-101</td>
      <td>-96</td>
      <td>-92</td>
      <td>-89</td>
      <td>-95</td>
      <td>-102</td>
      <td>-100</td>
      <td>-87</td>
      <td>-79</td>
      <td>...</td>
      <td>-82</td>
      <td>-81</td>
      <td>-80</td>
      <td>-77</td>
      <td>-85</td>
      <td>-77</td>
      <td>-72</td>
      <td>-69</td>
      <td>-65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-9</td>
      <td>-65</td>
      <td>-98</td>
      <td>-102</td>
      <td>-78</td>
      <td>-48</td>
      <td>-16</td>
      <td>0</td>
      <td>-21</td>
      <td>-59</td>
      <td>...</td>
      <td>4</td>
      <td>2</td>
      <td>-12</td>
      <td>-32</td>
      <td>-41</td>
      <td>-65</td>
      <td>-83</td>
      <td>-89</td>
      <td>-73</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 179 columns</p>
</div>




```python
# Summary of attributes
eeg.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>X10</th>
      <th>...</th>
      <th>X170</th>
      <th>X171</th>
      <th>X172</th>
      <th>X173</th>
      <th>X174</th>
      <th>X175</th>
      <th>X176</th>
      <th>X177</th>
      <th>X178</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11500.000000</td>
      <td>11500.000000</td>
      <td>11500.000000</td>
      <td>11500.000000</td>
      <td>11500.000000</td>
      <td>11500.000000</td>
      <td>11500.000000</td>
      <td>11500.00000</td>
      <td>11500.00000</td>
      <td>11500.000000</td>
      <td>...</td>
      <td>11500.000000</td>
      <td>11500.000000</td>
      <td>11500.000000</td>
      <td>11500.000000</td>
      <td>11500.000000</td>
      <td>11500.000000</td>
      <td>11500.000000</td>
      <td>11500.000000</td>
      <td>11500.000000</td>
      <td>11500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-11.581391</td>
      <td>-10.911565</td>
      <td>-10.187130</td>
      <td>-9.143043</td>
      <td>-8.009739</td>
      <td>-7.003478</td>
      <td>-6.502087</td>
      <td>-6.68713</td>
      <td>-6.55800</td>
      <td>-6.168435</td>
      <td>...</td>
      <td>-10.145739</td>
      <td>-11.630348</td>
      <td>-12.943478</td>
      <td>-13.668870</td>
      <td>-13.363304</td>
      <td>-13.045043</td>
      <td>-12.705130</td>
      <td>-12.426000</td>
      <td>-12.195652</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>165.626284</td>
      <td>166.059609</td>
      <td>163.524317</td>
      <td>161.269041</td>
      <td>160.998007</td>
      <td>161.328725</td>
      <td>161.467837</td>
      <td>162.11912</td>
      <td>162.03336</td>
      <td>160.436352</td>
      <td>...</td>
      <td>164.652883</td>
      <td>166.149790</td>
      <td>168.554058</td>
      <td>168.556486</td>
      <td>167.257290</td>
      <td>164.241019</td>
      <td>162.895832</td>
      <td>162.886311</td>
      <td>164.852015</td>
      <td>1.414275</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1839.000000</td>
      <td>-1838.000000</td>
      <td>-1835.000000</td>
      <td>-1845.000000</td>
      <td>-1791.000000</td>
      <td>-1757.000000</td>
      <td>-1832.000000</td>
      <td>-1778.00000</td>
      <td>-1840.00000</td>
      <td>-1867.000000</td>
      <td>...</td>
      <td>-1867.000000</td>
      <td>-1865.000000</td>
      <td>-1642.000000</td>
      <td>-1723.000000</td>
      <td>-1866.000000</td>
      <td>-1863.000000</td>
      <td>-1781.000000</td>
      <td>-1727.000000</td>
      <td>-1829.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-54.000000</td>
      <td>-55.000000</td>
      <td>-54.000000</td>
      <td>-54.000000</td>
      <td>-54.000000</td>
      <td>-54.000000</td>
      <td>-54.000000</td>
      <td>-55.00000</td>
      <td>-55.00000</td>
      <td>-54.000000</td>
      <td>...</td>
      <td>-55.000000</td>
      <td>-56.000000</td>
      <td>-56.000000</td>
      <td>-56.000000</td>
      <td>-55.000000</td>
      <td>-56.000000</td>
      <td>-55.000000</td>
      <td>-55.000000</td>
      <td>-55.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-8.000000</td>
      <td>-8.000000</td>
      <td>-7.000000</td>
      <td>-8.000000</td>
      <td>-8.000000</td>
      <td>-8.000000</td>
      <td>-8.000000</td>
      <td>-8.00000</td>
      <td>-7.00000</td>
      <td>-7.000000</td>
      <td>...</td>
      <td>-9.000000</td>
      <td>-10.000000</td>
      <td>-10.000000</td>
      <td>-10.000000</td>
      <td>-10.000000</td>
      <td>-9.000000</td>
      <td>-9.000000</td>
      <td>-9.000000</td>
      <td>-9.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>34.000000</td>
      <td>35.000000</td>
      <td>36.000000</td>
      <td>36.000000</td>
      <td>35.000000</td>
      <td>36.000000</td>
      <td>35.000000</td>
      <td>36.00000</td>
      <td>36.00000</td>
      <td>35.250000</td>
      <td>...</td>
      <td>34.000000</td>
      <td>34.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>34.000000</td>
      <td>34.000000</td>
      <td>34.000000</td>
      <td>34.000000</td>
      <td>34.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1726.000000</td>
      <td>1713.000000</td>
      <td>1697.000000</td>
      <td>1612.000000</td>
      <td>1518.000000</td>
      <td>1816.000000</td>
      <td>2047.000000</td>
      <td>2047.00000</td>
      <td>2047.00000</td>
      <td>2047.000000</td>
      <td>...</td>
      <td>1777.000000</td>
      <td>1472.000000</td>
      <td>1319.000000</td>
      <td>1436.000000</td>
      <td>1733.000000</td>
      <td>1958.000000</td>
      <td>2047.000000</td>
      <td>2047.000000</td>
      <td>1915.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 179 columns</p>
</div>




```python
# Last 5 rows
eeg.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>X10</th>
      <th>...</th>
      <th>X170</th>
      <th>X171</th>
      <th>X172</th>
      <th>X173</th>
      <th>X174</th>
      <th>X175</th>
      <th>X176</th>
      <th>X177</th>
      <th>X178</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11495</th>
      <td>-22</td>
      <td>-22</td>
      <td>-23</td>
      <td>-26</td>
      <td>-36</td>
      <td>-42</td>
      <td>-45</td>
      <td>-42</td>
      <td>-45</td>
      <td>-49</td>
      <td>...</td>
      <td>15</td>
      <td>16</td>
      <td>12</td>
      <td>5</td>
      <td>-1</td>
      <td>-18</td>
      <td>-37</td>
      <td>-47</td>
      <td>-48</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11496</th>
      <td>-47</td>
      <td>-11</td>
      <td>28</td>
      <td>77</td>
      <td>141</td>
      <td>211</td>
      <td>246</td>
      <td>240</td>
      <td>193</td>
      <td>136</td>
      <td>...</td>
      <td>-65</td>
      <td>-33</td>
      <td>-7</td>
      <td>14</td>
      <td>27</td>
      <td>48</td>
      <td>77</td>
      <td>117</td>
      <td>170</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11497</th>
      <td>14</td>
      <td>6</td>
      <td>-13</td>
      <td>-16</td>
      <td>10</td>
      <td>26</td>
      <td>27</td>
      <td>-9</td>
      <td>4</td>
      <td>14</td>
      <td>...</td>
      <td>-65</td>
      <td>-48</td>
      <td>-61</td>
      <td>-62</td>
      <td>-67</td>
      <td>-30</td>
      <td>-2</td>
      <td>-1</td>
      <td>-8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11498</th>
      <td>-40</td>
      <td>-25</td>
      <td>-9</td>
      <td>-12</td>
      <td>-2</td>
      <td>12</td>
      <td>7</td>
      <td>19</td>
      <td>22</td>
      <td>29</td>
      <td>...</td>
      <td>121</td>
      <td>135</td>
      <td>148</td>
      <td>143</td>
      <td>116</td>
      <td>86</td>
      <td>68</td>
      <td>59</td>
      <td>55</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11499</th>
      <td>29</td>
      <td>41</td>
      <td>57</td>
      <td>72</td>
      <td>74</td>
      <td>62</td>
      <td>54</td>
      <td>43</td>
      <td>31</td>
      <td>23</td>
      <td>...</td>
      <td>-59</td>
      <td>-25</td>
      <td>-4</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>-2</td>
      <td>2</td>
      <td>20</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 179 columns</p>
</div>




```python
#Separating 2nd and 3rd rows
print (eeg[1:3])
```

        X1   X2   X3   X4   X5   X6   X7   X8   X9  X10 ...  X170  X171  X172  \
    1  386  382  356  331  320  315  307  272  244  232 ...   164   150   146   
    2  -32  -39  -47  -37  -32  -36  -57  -73  -85  -94 ...    57    64    48   
    
       X173  X174  X175  X176  X177  X178  y  
    1   152   157   156   154   143   129  1  
    2    19   -12   -30   -35   -35   -36  5  
    
    [2 rows x 179 columns]
    


```python
# Separating first 3 rows
e1 = eeg[:3]
```


```python
print(e1)
```

        X1   X2   X3   X4   X5   X6   X7   X8   X9  X10 ...  X170  X171  X172  \
    0  135  190  229  223  192  125   55   -9  -33  -38 ...   -17   -15   -31   
    1  386  382  356  331  320  315  307  272  244  232 ...   164   150   146   
    2  -32  -39  -47  -37  -32  -36  -57  -73  -85  -94 ...    57    64    48   
    
       X173  X174  X175  X176  X177  X178  y  
    0   -77  -103  -127  -116   -83   -51  4  
    1   152   157   156   154   143   129  1  
    2    19   -12   -30   -35   -35   -36  5  
    
    [3 rows x 179 columns]
    


```python
# Information on columns
eeg.columns
```




    Index(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10',
           ...
           'X170', 'X171', 'X172', 'X173', 'X174', 'X175', 'X176', 'X177', 'X178',
           'y'],
          dtype='object', length=179)




```python
# Distribution of first column
sns.distplot(eeg.X1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2c86530b7f0>




![png](output_10_1.png)



```python
# Distribution of last column before the class attribute
sns.distplot(eeg.X178)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2c86dae6400>




![png](output_11_1.png)



```python
# Distribution of counts in one time stamp
sns.countplot(x='X1', data=eeg)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2c802be2c18>




![png](output_12_1.png)



```python
# spread of positive and negative values over all attributes
sns.stripplot(data=eeg, jitter=True, palette='Set1')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2c802bed518>




![png](output_13_1.png)



```python
# outlier check on attribute X178
sns.boxplot(x='X178', data=eeg)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2c8030d6908>




![png](output_14_1.png)



```python
# qualitative statistic w.r.t. different class
sns.boxplot(x='X178', y='y', data=eeg, palette='coolwarm')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2c8774eb2e8>




![png](output_15_1.png)



```python
sns.factorplot(x='X178', y='y', data=eeg);
```


![png](output_16_0.png)



```python
# Separating 3rd row only
E2 = eeg.iloc[2]
```


```python
# distribution in the 3rd row 
sns.distplot(E2, bins=20, kde=True, rug=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2c861324d30>




![png](output_18_1.png)



```python
# frequency across 3rd row
plt.plot(E2, lw=2)
```




    [<matplotlib.lines.Line2D at 0x2c864bafeb8>]




![png](output_19_1.png)



```python
# Histogram of 3rd row
plt.hist(E2, bins=20)
```




    (array([  2.,  10.,  14.,  10.,  14.,  11.,  18.,  12.,  26.,  11.,  11.,
             10.,   6.,   4.,   6.,   5.,   4.,   1.,   3.,   1.]),
     array([-126. , -115.7, -105.4,  -95.1,  -84.8,  -74.5,  -64.2,  -53.9,
             -43.6,  -33.3,  -23. ,  -12.7,   -2.4,    7.9,   18.2,   28.5,
              38.8,   49.1,   59.4,   69.7,   80. ]),
     <a list of 20 Patch objects>)




![png](output_20_1.png)



```python
# Separating 100th row 
E100= eeg.iloc[100]
```


```python
# scatter plot of 100th row 
plt.plot(E100, 'ro')
```




    [<matplotlib.lines.Line2D at 0x2c873129780>]




![png](output_22_1.png)



```python
# Class attribute and its distribution
Y = eeg.y
```


```python
# Total number of cases in all classes; shows equal number of entries in each class
plt.hist(Y, bins=50)
```




    (array([ 2300.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,
                0.,     0.,     0.,     0.,  2300.,     0.,     0.,     0.,
                0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,
                0.,  2300.,     0.,     0.,     0.,     0.,     0.,     0.,
                0.,     0.,     0.,     0.,     0.,  2300.,     0.,     0.,
                0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,
                0.,  2300.]),
     array([ 1.  ,  1.08,  1.16,  1.24,  1.32,  1.4 ,  1.48,  1.56,  1.64,
             1.72,  1.8 ,  1.88,  1.96,  2.04,  2.12,  2.2 ,  2.28,  2.36,
             2.44,  2.52,  2.6 ,  2.68,  2.76,  2.84,  2.92,  3.  ,  3.08,
             3.16,  3.24,  3.32,  3.4 ,  3.48,  3.56,  3.64,  3.72,  3.8 ,
             3.88,  3.96,  4.04,  4.12,  4.2 ,  4.28,  4.36,  4.44,  4.52,
             4.6 ,  4.68,  4.76,  4.84,  4.92,  5.  ]),
     <a list of 50 Patch objects>)




![png](output_24_1.png)



```python
# Distribution of class values
sns.distplot(Y, hist=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2c864f2ea58>




![png](output_25_1.png)



```python
# Mean values w.r.t. different class at timestamp X1
eeg.groupby('y').mean()['X1']
```




    y
    1   -21.936522
    2    -7.710000
    3    -9.207391
    4   -12.726087
    5    -6.326957
    Name: X1, dtype: float64




```python
# Mean values w.r.t. different class at timestamp X178
eeg.groupby('y').mean()['X178']
```




    y
    1   -24.016522
    2    -8.147391
    3    -8.935217
    4   -12.914783
    5    -6.964348
    Name: X178, dtype: float64




```python
# Mean values w.r.t. different class at timestamp X100
eeg.groupby('y').mean()['X100']
```




    y
    1    -4.706957
    2    -3.762609
    3    -7.367391
    4   -13.959565
    5    -6.549565
    Name: X100, dtype: float64




```python
# Counting total number of epileptic cases over on a single timestamp
Epi_X1 = sum(eeg[eeg['y']==1]['X1'].value_counts()==1)
```


```python
Epi_X1
```




    473




```python
Epi_X178 = sum(eeg[eeg['y']==1]['X178'].value_counts()==1)
```


```python
Epi_X178
```




    461




```python
Epi_X100 = sum(eeg[eeg['y']==1]['X100'].value_counts()==1)
```


```python
Epi_X100
```




    518




```python
# Regrouping the data in terms of class variables
cls = eeg.groupby('y')
```


```python
# mean values of individual class
Mean = cls.mean()
```


```python
# standard dviation of individual class
Std = cls.std()
```


```python
Mean
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>X10</th>
      <th>...</th>
      <th>X169</th>
      <th>X170</th>
      <th>X171</th>
      <th>X172</th>
      <th>X173</th>
      <th>X174</th>
      <th>X175</th>
      <th>X176</th>
      <th>X177</th>
      <th>X178</th>
    </tr>
    <tr>
      <th>y</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-21.936522</td>
      <td>-19.049130</td>
      <td>-15.293913</td>
      <td>-9.836087</td>
      <td>-3.707391</td>
      <td>2.730000</td>
      <td>6.899565</td>
      <td>7.425652</td>
      <td>8.380870</td>
      <td>10.773913</td>
      <td>...</td>
      <td>-7.266522</td>
      <td>-9.714783</td>
      <td>-15.909565</td>
      <td>-22.383043</td>
      <td>-26.424783</td>
      <td>-26.221739</td>
      <td>-25.830870</td>
      <td>-25.043913</td>
      <td>-24.548261</td>
      <td>-24.016522</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-7.710000</td>
      <td>-8.148696</td>
      <td>-8.020000</td>
      <td>-8.050435</td>
      <td>-7.776957</td>
      <td>-8.220870</td>
      <td>-8.597826</td>
      <td>-9.210000</td>
      <td>-9.217826</td>
      <td>-9.347826</td>
      <td>...</td>
      <td>-7.979565</td>
      <td>-8.483043</td>
      <td>-8.967391</td>
      <td>-9.067391</td>
      <td>-8.924783</td>
      <td>-8.592174</td>
      <td>-8.524348</td>
      <td>-8.558696</td>
      <td>-8.187391</td>
      <td>-8.147391</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-9.207391</td>
      <td>-9.276522</td>
      <td>-9.968261</td>
      <td>-10.634348</td>
      <td>-11.172174</td>
      <td>-11.559130</td>
      <td>-11.797826</td>
      <td>-12.013043</td>
      <td>-12.132609</td>
      <td>-12.249130</td>
      <td>...</td>
      <td>-13.590000</td>
      <td>-13.332174</td>
      <td>-13.218696</td>
      <td>-12.789565</td>
      <td>-12.366087</td>
      <td>-11.320000</td>
      <td>-10.455217</td>
      <td>-9.757391</td>
      <td>-9.418696</td>
      <td>-8.935217</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-12.726087</td>
      <td>-12.410435</td>
      <td>-12.272609</td>
      <td>-12.096957</td>
      <td>-12.083043</td>
      <td>-12.349130</td>
      <td>-13.064348</td>
      <td>-13.419565</td>
      <td>-13.441739</td>
      <td>-13.147826</td>
      <td>...</td>
      <td>-13.428696</td>
      <td>-13.636522</td>
      <td>-14.114783</td>
      <td>-14.404783</td>
      <td>-14.251304</td>
      <td>-14.109565</td>
      <td>-13.653913</td>
      <td>-13.082174</td>
      <td>-12.791739</td>
      <td>-12.914783</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-6.326957</td>
      <td>-5.673043</td>
      <td>-5.380870</td>
      <td>-5.097391</td>
      <td>-5.309130</td>
      <td>-5.618261</td>
      <td>-5.950000</td>
      <td>-6.218696</td>
      <td>-6.378696</td>
      <td>-6.871304</td>
      <td>...</td>
      <td>-5.229565</td>
      <td>-5.562174</td>
      <td>-5.941304</td>
      <td>-6.072609</td>
      <td>-6.377391</td>
      <td>-6.573043</td>
      <td>-6.760870</td>
      <td>-7.083478</td>
      <td>-7.183913</td>
      <td>-6.964348</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 178 columns</p>
</div>




```python
Std
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>X10</th>
      <th>...</th>
      <th>X169</th>
      <th>X170</th>
      <th>X171</th>
      <th>X172</th>
      <th>X173</th>
      <th>X174</th>
      <th>X175</th>
      <th>X176</th>
      <th>X177</th>
      <th>X178</th>
    </tr>
    <tr>
      <th>y</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>342.361939</td>
      <td>343.398782</td>
      <td>337.489643</td>
      <td>332.354833</td>
      <td>332.211163</td>
      <td>332.941638</td>
      <td>332.899545</td>
      <td>334.375084</td>
      <td>334.534814</td>
      <td>331.071853</td>
      <td>...</td>
      <td>342.095308</td>
      <td>341.534693</td>
      <td>345.740615</td>
      <td>351.503084</td>
      <td>351.184524</td>
      <td>347.475865</td>
      <td>339.650467</td>
      <td>335.747017</td>
      <td>335.244512</td>
      <td>339.819309</td>
    </tr>
    <tr>
      <th>2</th>
      <td>95.525136</td>
      <td>94.734737</td>
      <td>93.716055</td>
      <td>92.509248</td>
      <td>91.004234</td>
      <td>90.880780</td>
      <td>91.679224</td>
      <td>92.097291</td>
      <td>91.138175</td>
      <td>89.759109</td>
      <td>...</td>
      <td>92.959490</td>
      <td>89.569320</td>
      <td>86.961108</td>
      <td>86.397728</td>
      <td>87.752018</td>
      <td>90.156202</td>
      <td>92.433147</td>
      <td>94.311338</td>
      <td>96.013765</td>
      <td>96.710559</td>
    </tr>
    <tr>
      <th>3</th>
      <td>59.924231</td>
      <td>60.240543</td>
      <td>59.905387</td>
      <td>59.697773</td>
      <td>59.502204</td>
      <td>59.635346</td>
      <td>60.043847</td>
      <td>59.865027</td>
      <td>59.463238</td>
      <td>59.110605</td>
      <td>...</td>
      <td>60.209589</td>
      <td>60.715733</td>
      <td>60.541729</td>
      <td>61.036980</td>
      <td>60.658758</td>
      <td>60.726974</td>
      <td>60.273966</td>
      <td>60.178156</td>
      <td>59.736182</td>
      <td>59.849148</td>
    </tr>
    <tr>
      <th>4</th>
      <td>69.729925</td>
      <td>70.708016</td>
      <td>71.425314</td>
      <td>71.519947</td>
      <td>71.055967</td>
      <td>70.532877</td>
      <td>70.027232</td>
      <td>69.902928</td>
      <td>69.674909</td>
      <td>69.296696</td>
      <td>...</td>
      <td>70.359338</td>
      <td>70.353789</td>
      <td>70.441038</td>
      <td>70.135134</td>
      <td>69.881104</td>
      <td>69.803366</td>
      <td>69.846537</td>
      <td>70.480867</td>
      <td>70.720585</td>
      <td>70.646346</td>
    </tr>
    <tr>
      <th>5</th>
      <td>47.553062</td>
      <td>47.903503</td>
      <td>48.135729</td>
      <td>48.560703</td>
      <td>48.568777</td>
      <td>48.906275</td>
      <td>49.121173</td>
      <td>49.197125</td>
      <td>48.996716</td>
      <td>48.979947</td>
      <td>...</td>
      <td>47.512691</td>
      <td>47.424105</td>
      <td>47.793875</td>
      <td>47.799416</td>
      <td>47.660365</td>
      <td>47.505155</td>
      <td>47.687482</td>
      <td>47.977213</td>
      <td>48.238124</td>
      <td>48.276043</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 178 columns</p>
</div>




```python
# separating Class-1 data
Class_1 = Mean[:1]
```


```python
Class_1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>X10</th>
      <th>...</th>
      <th>X169</th>
      <th>X170</th>
      <th>X171</th>
      <th>X172</th>
      <th>X173</th>
      <th>X174</th>
      <th>X175</th>
      <th>X176</th>
      <th>X177</th>
      <th>X178</th>
    </tr>
    <tr>
      <th>y</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-21.936522</td>
      <td>-19.04913</td>
      <td>-15.293913</td>
      <td>-9.836087</td>
      <td>-3.707391</td>
      <td>2.73</td>
      <td>6.899565</td>
      <td>7.425652</td>
      <td>8.38087</td>
      <td>10.773913</td>
      <td>...</td>
      <td>-7.266522</td>
      <td>-9.714783</td>
      <td>-15.909565</td>
      <td>-22.383043</td>
      <td>-26.424783</td>
      <td>-26.221739</td>
      <td>-25.83087</td>
      <td>-25.043913</td>
      <td>-24.548261</td>
      <td>-24.016522</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 178 columns</p>
</div>




```python
Std_1 = Std[:1]
```


```python
Std_1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>X10</th>
      <th>...</th>
      <th>X169</th>
      <th>X170</th>
      <th>X171</th>
      <th>X172</th>
      <th>X173</th>
      <th>X174</th>
      <th>X175</th>
      <th>X176</th>
      <th>X177</th>
      <th>X178</th>
    </tr>
    <tr>
      <th>y</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>342.361939</td>
      <td>343.398782</td>
      <td>337.489643</td>
      <td>332.354833</td>
      <td>332.211163</td>
      <td>332.941638</td>
      <td>332.899545</td>
      <td>334.375084</td>
      <td>334.534814</td>
      <td>331.071853</td>
      <td>...</td>
      <td>342.095308</td>
      <td>341.534693</td>
      <td>345.740615</td>
      <td>351.503084</td>
      <td>351.184524</td>
      <td>347.475865</td>
      <td>339.650467</td>
      <td>335.747017</td>
      <td>335.244512</td>
      <td>339.819309</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 178 columns</p>
</div>




```python
# Distribution of mean and standard deviation of class-1 (ictal data for epileptic cases)
sns.distplot(Class_1)
sns.distplot(Std_1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x20c14f60780>




![png](output_44_1.png)



```python
# Separating mean values of other classes
Class_2 = Mean[:2]
Class_3 = Mean[:3]
Class_4 = Mean[:4]
Class_5 = Mean[:5]
```


```python
# histograms of mean values 
fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs[0, 0].hist(Class_2)
axs[1, 0].hist(Class_3)
axs[0, 1].hist(Class_4)
axs[1, 1].hist(Class_5)
plt.show()
```


![png](output_46_0.png)



```python
sns.stripplot(data=Mean, jitter=True, palette='Set1')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x20c18f93208>




![png](output_47_1.png)



```python
sns.violinplot(data=Mean, palette='Set1')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x20c1b9f98d0>




![png](output_48_1.png)

