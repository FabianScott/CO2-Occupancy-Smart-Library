import pandas as pd
filename = 'data/N_30_11.csv'
df = pd.read_csv(filename, sep=';')

temp = []
for el in df['Time']:
    temp.append('2022.11.30.' + el)
df['Time'] = temp

df.to_csv(filename[:-4] + 'new.csv')
