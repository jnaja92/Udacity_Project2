
import pandas as pd

#y_train = y_all.loc[X_all.index.isin(X_train.index)]

d = {'one' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),'two' : pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd']),'three' : pd.Series([4., 5., 6.,7.], index=['a', 'b', 'c','d'])}
df = pd.DataFrame(d)
print 'df:'
print df

#df1=df[['one','two']]

df1=df.iloc[0:2]
print 'df1:'
print df1

print df.loc[~df.index.isin(df1.index)]