import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys

#1.0 数据导入操作：读取保单、赔款数据集
df_pol = pd.read_csv('/mnt/d/prg_lnx/anaconda/envs/tensorflow/data/policy_sample.csv',index_col=False)
df_clm = pd.read_csv('/mnt/d/prg_lnx/anaconda/envs/tensorflow/data/claim_sample.csv',index_col=False)


df_clm.head()


print(df_pol.shape,df_clm.shape)

#1.1 发现导入的数据中，第一列是原有数据的序号列，没有任何意义，这时可以通过以下方式删除这一列。
df_pol.drop('Unnamed: 0',axis=1,inplace=True)
df_clm.drop('Unnamed: 0',axis=1,inplace=True)


df_pol.head()


print(df_pol.dtypes)

#2.1 检验保单库中的保单号是否有重复，并且输出重复项
print ("There are %s records in policy dataset with duplicated policy no"%df_pol.duplicated('pol_no').sum())

#2.2 检查保单号是否一样长
pol_length = df_pol.pol_no.map(lambda x: len(np.str(x)))
print ("保单号的长度为%s"%pol_length.unique())


plt.boxplot(df_clm.ultloss,sym='r+',whis=[5,95],meanline=True)
plt.show()