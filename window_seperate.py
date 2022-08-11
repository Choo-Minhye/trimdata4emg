import pandas as pd
from trim_n_filter import bp_filter, notch_filter, plot_2_signal,plot_t_signal, clipdata, signal_analysis
import matplotlib.pyplot as plt

# Load data from Excel file
signal_path = '220810_1.xlsx'

signal_1 = pd.read_excel(signal_path, usecols = [1]).values
signal_2 = pd.read_excel(signal_path, usecols = [2]).values
channel_name_1 = 'B Data'
channel_name_2 = 'C Data'

# Sampling Frequency of 1000 (1000 Samples per second)
sampling_frequency = 1



# plot_2_signal(signal_1, signal_2, sampling_frequency, channel_name_1, channel_name_2)
# plot_2_signal(signal_2, signal_1, sampling_frequency, channel_name_1, channel_name_2)

signal_x = signal_1.reshape((signal_1.size,))
signal_y = signal_2.reshape((signal_2.size,))


print(signal_y[0])
print(signal_y[1])
print(signal_y[2])
print(signal_y[3])
print(signal_y[4])
print(signal_y[5])
print(signal_y[6])
print(signal_y[7])

info = []
b_arr, c_arr, x_t_arr, y_t_arr, info = signal_analysis(signal_x,signal_y, sampling_frequency, 5, True)
# c_arr, b_arr, y_t_arr, x_t_arr, info = signal_analysis(signal_y,signal_x, sampling_frequency, 5, True)
plot_t_signal(b_arr, c_arr,  channel_name_1, channel_name_2)



# 최종적으로 계산되어 반환되어야 하는 요소는 

# window의 크기 

# b의 0이 아닌 데이터 수
# c의 0이 아닌 데이터 수 

# b_s의 데이터 수
# c_s의 데이터 수

# b_s 행렬
# c_s 행렬


plt.show()



dataframe = {'b_t':x_t_arr,'b_s':b_arr}
# export result data in excel
_to_excel = pd.DataFrame(dataframe)
_to_excel.to_excel("analysis_b_s.xlsx", index = False)
dataframe = {'c_t':y_t_arr,'c_s':c_arr}
# export result data in excel
_to_excel = pd.DataFrame(dataframe)
_to_excel.to_excel("analysis_c_s.xlsx", index = False)