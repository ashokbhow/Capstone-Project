

```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# reading the data file
eeg_df = pd.read_csv('eeg_data_1.csv')
```


```python
# first 5 lines of the data 
eeg_df.head()
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
# removing and storing the class attribute for later use
Class = list(eeg_df.pop('y'))
```


```python
# Extracting the measurements as a numpy array
samples = eeg_df.values
```


```python
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
```


```python
from sklearn.cluster import KMeans
```


```python
kmeans = KMeans(n_clusters=5)
```


```python
kmeans.fit(samples)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=5, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0)




```python
kmeans.labels_
```




    array([3, 0, 3, ..., 0, 3, 0])




```python
plt.scatter(samples[:,0],samples[:,177],c=kmeans.labels_,cmap='rainbow')
plt.title("K Means")
```




    Text(0.5,1,'K Means')




![png](output_9_1.png)



```python
#To visualise real clusters and idetified clusters, plot them together
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,5))

ax1.set_title('K Means')
ax1.scatter(samples[:,0],samples[:,177],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(samples[:,0],samples[:,177],c=Class,cmap='rainbow')
```




    <matplotlib.collections.PathCollection at 0x1e783ae0ef0>




![png](output_10_1.png)



```python
# Class =1 Ictal Data ; Has much higher voltage values as plotted separately
eeg_1 = pd.read_csv('eeg_class1.csv')
```


```python
Class1 = list(eeg_1.pop('y'))
```


```python
s1 = eeg_1.values
```


```python
eeg_2 = pd.read_csv('eeg_class2.csv')
```


```python
Class2 = list(eeg_2.pop('y'))
```


```python
s2 = eeg_2.values
```


```python
eeg_3 = pd.read_csv("eeg_class3.csv")
Class3 = list(eeg_3.pop('y'))
s3 = eeg_3.values
```


```python
eeg_4 = pd.read_csv("eeg_class4.csv")
Class4 = list(eeg_4.pop('y'))
s4 = eeg_4.values
```


```python
eeg_5 = pd.read_csv("eeg_class5.csv")
Class5 = list(eeg_5.pop('y'))
s5 = eeg_5.values
```


```python
#To visualise real clusters and idetified clusters, plot them together
f, (ax3, ax4, ax5, ax6, ax7)= plt.subplots(1, 5, sharey=True,figsize=(12,3))

ax3.set_title('1:Ictal, seizure')
ax3.scatter(s1[:,0],s1[:,177],c=Class1,cmap='rainbow')
ax4.set_title("2:probe at tumor")
ax4.scatter(s2[:,0],s2[:,177],c=Class2,cmap='rainbow')
ax5.scatter(s3[:,0],s3[:,177],c=Class3,cmap='rainbow')
ax5.set_title("3:other area")
ax6.scatter(s4[:,0],s4[:,177],c=Class4,cmap='rainbow')
ax6.set_title("4:eyes closed")
ax7.scatter(s5[:,0],s5[:,177],c=Class5,cmap='rainbow')
ax7.set_title("5:eyes open")
```




    Text(0.5,1,'5:eyes open')




![png](output_20_1.png)



```python
# Normalizing all class data
norm_samples = normalize(samples)
```


```python
# Hirearchical analogue of K-means clustering with the normalized data
mergings_norm = linkage(norm_samples, method='ward')
```


```python
plt.figure(figsize=(12,6))
dendrogram(mergings_norm, labels=Class, leaf_rotation=90, leaf_font_size=10)
plt.show()
```


![png](output_23_0.png)



```python
# comment: at level 15 there are 5 large nested clusters
```


```python
kmeans2 = KMeans(n_clusters=5)
```


```python
# Investigating KMeans clustering on normalized data
kmeans2.fit(norm_samples)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=5, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0)




```python
kmeans2.labels_
```




    array([3, 4, 1, ..., 4, 1, 4])




```python
plt.scatter(norm_samples[:,0],norm_samples[:,177],c=kmeans2.labels_,cmap='rainbow')
plt.title("K Means_Normalized")
```




    Text(0.5,1,'K Means_Normalized')




![png](output_28_1.png)



```python
eeg_nonictal = pd.read_csv('class_nonictal.csv')
```


```python
Class_nonictal = list(eeg_nonictal.pop('y'))
```


```python
sample_nonictal = eeg_nonictal.values
```


```python
from sklearn.cluster import KMeans
kmeans_2 = KMeans(n_clusters=4)
kmeans_2.fit(sample_nonictal)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=4, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0)




```python
kmeans_2.labels_
```




    array([0, 3, 2, ..., 3, 3, 3])




```python
plt.figure(figsize=(12,6))
plt.scatter(sample_nonictal[:,0],sample_nonictal[:,177],c=kmeans_2.labels_,cmap='rainbow')
plt.title("K Means Nonictal")
```




    Text(0.5,1,'K Means Nonictal')




![png](output_34_1.png)



```python
# Normalizing the non-ictal classes 2, 3, 4, 5
norm_nonictal = normalize(sample_nonictal)
```


```python
kmeans_non = KMeans(n_clusters=4)
```


```python
# KMeans clustering of normalized non-ictal data
kmeans_non.fit(norm_nonictal)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=4, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0)




```python
kmeans_non.labels_
```




    array([1, 1, 2, ..., 0, 2, 3])




```python
plt.scatter(norm_nonictal[:,0],norm_nonictal[:,177],c=kmeans_non.labels_,cmap='rainbow')
plt.title("nonictal Classes")
```




    Text(0.5,1,'nonictal Classes')




![png](output_39_1.png)



```python
# Investigating nesting by hirearchical analogue of KMeans on normalized non-ictal data
mergings_nonictal = linkage(norm_nonictal, method='ward')
```


```python
plt.figure(figsize=(12,6))
dendrogram(mergings_nonictal, labels=Class_nonictal, leaf_rotation=90, leaf_font_size=10)
plt.show()
```


![png](output_41_0.png)



```python
# at level 15 there are 4 nested clusters
# The data can be viewed as a nested cluster. However it is not good for clustering studies
# End
```
