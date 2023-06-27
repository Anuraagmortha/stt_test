import pandas as pd 
import os
import glob
import re

REPORTS_DIR = "D:/sp/stt_test/reports/"
CLIENT_NAME = "amc"
DATE = "2023-06-26"   # In YYYY-MM-DD format only

final_report_df = pd.DataFrame()
pattern = r'^report.*.csv$'

client_date_dir_path = os.path.join(REPORTS_DIR, CLIENT_NAME, DATE)
files = glob.glob(os.path.join(client_date_dir_path, '*'))

for file_path in files:
    csv_files = glob.glob(os.path.join(file_path, '*'))
    for file in csv_files:
        file_name = file.split("\\")[-1]
        if re.match(pattern, file_name):
            csv_path = os.path.join(file_path, file_name)
            df = pd.read_csv(csv_path)
            final_report_df = pd.concat([final_report_df, df], ignore_index=True)

final_report_df = final_report_df.drop('Unnamed: 0', axis=1)
print(final_report_df)








'''
for dir in os.scandir(some_dir_path):
    client_path = dir.path
    #print(client_path)
    final_path = client_path + '/' + 'report.*.csv'
    print(final_path)
    
    csv_files = glob.glob(client_path + '/' + 'report.*.csv')
    print(csv_files)
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(df)
'''
