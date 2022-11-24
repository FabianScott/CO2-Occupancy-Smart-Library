import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from Functions import exponential_moving_average, kalman_estimates

length = 200
min_error, error_proportion = 50, 0.03
mean = 100
measurements = np.random.normal(mean, min_error, length)

estimates, errors = kalman_estimates(measurements, min_error, error_proportion)

time_step = 1   # in minutes
times = [datetime(year=2000, month=1, day=1, hour=1, minute=0) + timedelta(minutes=i + time_step - 1) for i in range(length)]
x = np.column_stack((times, measurements))
exp_avg1 = exponential_moving_average(x, tau=60)
exp_avg2 = exponential_moving_average(x, tau=600)
exp_avg3 = exponential_moving_average(x, tau=6000)

line_scale = 0.5
point_scale = 0.2
plt.plot(range(length), estimates, color='c')
plt.scatter(range(length), measurements, s=point_scale, color='b')
plt.plot(range(length), [mean for _ in range(length)])
plt.plot(range(length), exp_avg1, color='r', linewidth=line_scale)
plt.plot(range(length), exp_avg2, color='g', linewidth=line_scale)
# plt.plot(range(length), exp_avg3, color='y', linewidth=line_scale)
plt.title('Kalman filter and ema')
plt.show()
