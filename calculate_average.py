from Functions import load_data, str_to_dt
import pandas as pd
import numpy as np
import time

device_data_list = load_data('data/big_data.txt', sep='\t', format_time='%d/%m/%Y %H.%M.%S', digits_to_remove=0)

sums = np.ones((28, int(24*4)))
counts = np.ones((28, int(24*4)))

for row, device in enumerate(device_data_list):
    t = time.time()
    print(f'Starting element {row}')
    for j, el in enumerate(device):
        if el:
            column = int(el[0].hour*4+el[0].minute/15)
            coords = (row, column)
            sums[coords] += el[1]
            counts[coords] += 1
    print(f'Took {time.time() - t}')
average = sums/counts
print(f'The sums are: \n{sums}')
print(f'The counts are: \n{counts}')
print(f'The averages are: \n{average}')

df = pd.DataFrame(average)
