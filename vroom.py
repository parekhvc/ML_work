import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import csv
import math


data_to_output=[]
dates_to_output=[]
def prediction(col1,col2):
	fields = [col1, col2]
	df = pd.read_csv('case_study_a.csv', skipinitialspace=True, usecols=fields)
	df[col1] = pd.DatetimeIndex(df[col1])
	df = df.rename(columns={col1: 'ds', col2: 'y'})
	my_model = Prophet(interval_width=0.99)
	my_model.fit(df)
	future_dates = my_model.make_future_dataframe(periods=53, freq='w')
	forecast = my_model.predict(future_dates)
	for i in forecast['ds']:
		dates_to_output.append(i)
	for i in forecast['yhat']:
		if i >= 0:
			data_to_output.append(math.floor(i))
		else:
			data_to_output.append(0)
	my_model.plot(forecast, uncertainty=True)
	plt.savefig('part_25.png')
	

prediction('Week','part_25')

'''
f = open('case_study_a.csv', 'rb')
reader = csv.reader(f)
headers = reader.next()

for i in headers[1:]:
	prediction('Week',i)

output_data = [data_to_output[i:i + 159] for i in range(0, len(data_to_output), 159)]
output_dates = [dates_to_output[i:i + 159] for i in range(0, len(dates_to_output), 159)]

def csv_writer(data,path):
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)

csv_writer(zip(
	output_dates[0],
	output_data[0],
	output_data[1],
	output_data[2],
        output_data[3],
	output_data[4],
        output_data[5],
        output_data[6],
        output_data[7],
	output_data[8],
        output_data[9],
        output_data[10],
        output_data[11],
	output_data[12],
        output_data[13],
        output_data[14],
        output_data[15],
	output_data[16],
        output_data[17],
        output_data[18],
        output_data[19],
	output_data[20],
        output_data[21],
        output_data[22],
        output_data[23],
	output_data[24],
        output_data[25]),
	'tmp_test.csv')

'''













