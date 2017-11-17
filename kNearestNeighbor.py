import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('./train.csv')
#df.drop('SalePrice', axis = 1, inplace = True)
#test = pd.read_csv('../input/test.csv')
#df = df.append(test, ignore_index = True)
df.head()