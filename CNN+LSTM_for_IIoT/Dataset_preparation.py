import requests
import pandas as pd
import numpy as np
import os


# one hot function to be used when the app features is need.
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


print("Allocating dataset parts vectors, constants and variables...")
orignal_dataset_csv_filepath = r'./Traffics/'

# will get pre defined values from a xlsx file stored in gdrive
try:
    r = requests.get(
        'https://drive.google.com/uc?export=download&id=157jtLqUtpmGp085ZbysSjG_777hHgI5P')
    with open('IoT-IIoT.Definitions.xlsx', 'wb') as f:
        f.write(r.content)
except:
    print("Could not get predefined data from drive. Change the code to use a cached file.")
    exit()


sensors = pd.read_excel('IoT-IIoT.Definitions.xlsx', sheet_name='Sensors') # Gets all IP sensors and they edge server IP
attackers = pd.read_excel('IoT-IIoT.Definitions.xlsx',sheet_name='Attackers')  # Gets all IP attackers
UnwFeatures = pd.read_excel('IoT-IIoT.Definitions.xlsx', sheet_name='Unwanted_Features') # List of columns names to be droped

# will be used to reduce all dataset in subsets as this percentage values
#maxsamples = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.70, 0.85, 1]
maxsamples = [0.01, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1]

dataset = []
for i in range(0, len(maxsamples)):
    dataset.append(pd.DataFrame())

fulldf = pd.DataFrame()
fulltest = pd.DataFrame()
ret_stat = pd.DataFrame()
test_size= float(input("Type a float value for test size ( 0.2 ):"))

print("Starting...")
dropappfeatures = input('Drop APP layer features? Dropping the app feature is our goal. Type Y for yes:')

for dirname, _, filenames in os.walk(orignal_dataset_csv_filepath):
    for filename in filenames:  # loop for the csv file that is the classes
        if filename.endswith('.csv'):
            filepath = os.path.join(dirname, filename)
            if filepath.upper().find('MODBUS') > 0:  # The ModBus csv file is broken, skips this file
                continue
            print('Oppening file:', filepath)
            data = pd.read_csv(filepath, low_memory=False )#,nrows=30000)
            datashape = data.shape
            print('Initial size:', data.shape[0])
            if dropappfeatures == 'Y':  # Will drop the app features, wich is our goal
                cols = UnwFeatures.loc[UnwFeatures['Type'] != 'OTHERS']['Feature_name'].values
                print('Dropping unwanted columns of High layer Network.')
                data.drop(cols, axis=1, inplace=True)
            else:  # leaves the app features, like the dataset authors
                data.drop(['http.file_data', 'http.request.full_uri', 'http.request.uri.query'], axis=1, inplace=True)
                cols = UnwFeatures.loc[UnwFeatures['Type'] == 'MQTT']['Feature_name'].values
                data.drop(cols, axis=1, inplace=True)
                cols = UnwFeatures.loc[UnwFeatures['Type'] == 'MODBUS']['Feature_name'].values
                data.drop(cols, axis=1, inplace=True)
                cols = UnwFeatures.loc[UnwFeatures['Type'] == 'DNS']['Feature_name'].values
                data.drop(cols, axis=1, inplace=True)

                encode_text_dummy(data, 'http.request.method')
                encode_text_dummy(data, 'http.referer')
                encode_text_dummy(data, "http.request.version")

            print('Dropping rows with null and duplicates')
            data.dropna(axis=0, how='any', inplace=True)
            data.drop_duplicates(subset=None, keep="first", inplace=True)
            print('Actual size:', data.shape[0])

            traffictype = ''
            if filepath.upper().find('NORMAL') > 0:
                allowed = sensors.loc[sensors['Folder'] == os.path.basename(dirname)]['IoT_node'].values  # Get the IP address of the IoT node
                IoT_Device = sensors.loc[sensors['Folder'] == os.path.basename(dirname)]['IoT_Device'].values[0]
                traffictype = 'Normal'
                print('Cleaning sensors  IPs.')
                data = data.loc[(data['ip.src_host'].isin(allowed)) | (data['ip.dst_host'].isin(allowed))]
            if filepath.upper().find('ATTACK') > 0:
                IoT_Device = data['Attack_type'][0]# Zero is not in the sensor list, so we use to keep the column in data frame
                traffictype = 'Attack'


            print('Dropping last part of unwanted columns, such as IPs.')
            cols = UnwFeatures.loc[UnwFeatures['Type']== 'OTHERS']['Feature_name'].values
            data.drop(cols, axis=1, inplace=True)

            # the .values instruction is need becouse some how the values come null to data[] variable
            print('Converting hex values to float, and 0 to 0.0')
            data["tcp.checksum"] = pd.Series(
                map(float.fromhex, map(str, data["tcp.checksum"].values))).values
            data["icmp.checksum"] = pd.Series(
                map(float.fromhex, map(str, data["icmp.checksum"].values))).values
            data["tcp.connection.fin"] = pd.Series(
                map(float.fromhex, map(str, data["tcp.connection.fin"].values))).values
            data["tcp.flags"] = pd.Series(
                map(float.fromhex, map(str, data["tcp.flags"].values))).values
            data["icmp.seq_le"] = pd.Series(
                map(float.fromhex, map(str, data["icmp.seq_le"].values))).values
            data = data.replace('0', 0.0)
            print('Dealing with ips in wrong knew columns..')
            data['arp.opcode'] = data.apply(lambda r: r['arp.opcode'] if type(r['arp.opcode']) == float else np.nan, axis=1)
            data['arp.hw.size'] = data.apply(lambda r: r['arp.hw.size'] if type(r['arp.hw.size']) == float else np.nan, axis=1)

            print('Dropping rows with null and duplicates after application features removal.')
            data.dropna(axis=0, how='any', inplace=True)
            data.drop_duplicates(subset=None, keep="first", inplace=True)

            print('Setting sensor id...')
            data['IoT_Device'] = IoT_Device
            print("shuffling...")
            data = data.sample(frac=1, random_state=40).reset_index(drop=True)
            dfteste =data.sample(frac=test_size,random_state=43)
            subdata = data.drop(dfteste.index, inplace=False)

            row = pd.DataFrame().append({'Traffic_type': traffictype, 
                                        'Label': IoT_Device,
                                        'Original_size': datashape[0],
                                        'Preprocessed_size': data.shape[0],
                                        'test_size':dfteste.shape[0],
                                        'train_size':subdata.shape[0]
                                        },ignore_index=True)
            print('Actual size:', data.shape[0])

            print('Concatting and shuffling Test set...')
            fulltest = pd.concat([dfteste,fulltest], ignore_index=True).sample(frac=1).reset_index(drop=True)

            for i in range(len(maxsamples)-1, -1, -1):  # loop to make subsets
                print(i, '-Splitting, shuffling, and concating ', maxsamples[i], ' samples.')
                
                frac_val = (maxsamples[i] if i == len( maxsamples)-1 else  (maxsamples[i]/(1-test_size))/(maxsamples[i+1]/(1-test_size)) )#maxsamples[i]/maxsamples[i+1])
                #frac_val = frac_val/(1-test_size)
                # will leave at least 2000 samples or the amout that already exists for the working class
                if (subdata.shape[0]*frac_val) < 2000:
                    subdata = subdata.sample(n=2000 if subdata.shape[0] > 1999 else subdata.shape[0], random_state=43).reset_index(drop=True)
                else:
                    subdata = subdata.sample(frac=frac_val, random_state=43).reset_index(drop=True)

                merged = pd.concat([subdata, dataset[i]], ignore_index=True)
                dataset[i] = merged.sample(frac=1, random_state=9).reset_index(drop=True)
                row[f'Subset_{i}/{maxsamples[i]}'] = subdata.shape[0]
                # data=subdata
            ret_stat = pd.concat([ret_stat, row], ignore_index=True)


for i in range(len(maxsamples)-1, -1, -1):
    print('Writing subset ', i)
    dataset[i].sample(frac=1, random_state=99).reset_index(drop=True).to_parquet(
        f'./SubSets/iiot-full.SubSet-p{i}.gzip', compression='gzip')

print("Writing the test subset...")
fulltest.to_parquet('./SubSets/iiot-full.TestSet.gzip',compression='gzip')

ret_stat.to_csv('./SubSets/Edge-IIoT-SubSet_Process.Statistics.csv', index=False)

print('done!')
exit()
