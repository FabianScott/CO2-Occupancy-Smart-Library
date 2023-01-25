import numpy as np

id_map = {'DA00110043': 1, 'DA00110044': 2, 'DA00110045': 3, 'DA00130002': 4, 'DA00130001': 5, 'DA00110031': 6,
          'DA00110041': 7, 'DA00110047': 8, 'DA00110035': 9, 'DA00110049': 10, 'DA00110032': 11, 'AMNO-03': 12,
          'DA00110033': 13, 'DA00130004': 14, 'AMNO-04': 15, 'DA00110037': 16, 'DA00130003': 17, 'AMNO-01': 18,
          'AMNO-02': 19, 'DA00110040': 20, 'DA00100001': 21, 'DA00110036': 22, 'DA00110034': 23, 'DA00110046': 24,
          'DA00110039': 25, 'DA00110042': 26, 'DA00110038': 27, }

n_map = {0: [], 1: [2, 10], 2: [1, 3], 3: [2, 4], 4: [2, 5], 5: [4, 6], 6: [5, 7], 7: [6, 8], 8: [7, 9], 9: [8, 10],
         10: [1, 9], 11: [12, 20], 12: [11, 13], 13: [12, 14], 14: [13, 15], 15: [14, 16], 16: [15, 17],
         17: [16, 18], 18: [17, 19], 19: [18, 20], 20: [11, 19], 21: [22, 27], 22: [21, 23], 23: [22, 24],
         24: [23, 25], 25: [24, 26], 26: [25, 27], 27: [21, 26]
         }
#         Q_adj        Q_out    C_out        m in m^3/s
bounds = ((0, 1), (0, 1), (300, 500), (0, 0.0001))
# V in m^3:
V = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              11.5 * 14.6, 10.2 * 11.5, 8.8 * 14.2, 8.8 * 14.8, 9.2 * 11.3, 15.5 * 11.6, 4.8 * 28]) * 3 #* 1000
