import numpy as np
import pandas as pd

df = pd.read_csv('data.csv', sep=';')
m_0 = 0
m_1 = 0
m_2 = 0


for index, row in df.iterrows():
    m_0 += np.abs(row['mean'] - row['stddv'])
    m_1 += np.abs(row['mean'] - row['contrast'])
    m_2 += np.abs(row['stddv'] - row['contrast'])

print("mean vs stddv: ", m_0/150)
print("mean vs contrast: ", m_1/150)
print("stddv vs contrast: ", m_2/150)
