# A Privacy-aware Intrusion Detection Architecture for IIoT Edge Based on a Hybrid CNN-LSTM Approach

This repository contains the code of a neural network model based on a hybrid CNN-LSTM architecture to detect several attacks in the network traffic at the Edge of IIoT using only features from the transport and network layers. It was used to validate the hypothesis of the paper

Elias, Erik Miguel de and Carriel, Vinicius Sanches and De Oliveira, Guilherme Werneck and Dos Santos, Aldri Luiz and Nogueira, Michele and Junior, Roberto Hirata and Batista, Daniel Macêdo, "A Hybrid CNN-LSTM Model for IIoT Edge Privacy-Aware Intrusion Detection," 2022 IEEE Latin-American Conference on Communications (LATINCOM), Rio de Janeiro, Brazil, 2022, pp. 1-6, doi: 10.1109/LATINCOM56090.2022.10000468.

In case of questions, please write to Prof. Daniel Macêdo Batista <batista@ime.usp.br>.

## Requirements
Use the **requirements.txt** to install all dependencies before running the project scripts.
```
$ pip install -r requirements.txt
```

You must have the following expected folders structure from the project root folder:
```
|- Project Folder
|  |-figures
|  |-results 
|  |-SubSets
|  |
|  |-Trafics
|  |  |-Normal
|  |  |  |-Sensor 1
|  |  |  |-Sensor 2
|  |  |  |-(...)
|  |  |-Attack
|  |  |  |-Attack 1
|  |  |  |-Attack 2
|  |  |  |- (...)
```

An excel format file named **IoT-IIoT.Definitions.xlsx** with some specific content is needed. This file will be automatically downloaded to the root folder project when it is used if does not exists. This file has the IP addresses of devices, classes name for classification and other information.

## Dataset preparation and processing
**Dataset_preparation.py** will find all csv files and perform some processes to prepare the data and split its subsets.
The original csv files of the dataset must be in a subfolder "./Trafics". The original dataset is found at <https://phc.st/edgeiiotsetdataport> or <https://phc.st/edgeiiotsetkag>.

After processing the results will be in the "SubSets" folder. Each file represents a part of the dataset, in which the {x} will represent one integer number of the part in the name "iiot-full.SubSet-p{x}.gzip". Each part from 0 to 7 are approximately equivalent to: 0.9%, 2.6%, 5%, 10%, 20%, 40%, 60%, 80% of the preprocessed data from the dataset. A file named **Edge-IIoT-SubSet_Process.Statistics.csv** will be generated with the statistics.

The preprocessed data removes the features, cleans the source IP and destination IP traffic, and fixes values to float type or removes invalid records.

## Models building, training and evaluation
The **MultiModels.py** has the model's representation, training process and evaluation processes. It will use the **figures**, **SubSets**, **results** and **TrainedModels** folders at the root of the project. The figures folder will be saved the Confusion Matrix picture of each test made for each model saved on **TrainedModels** folder. The results of the evaluation, which means, the metrics scores, will be saved as csv file in the **results** folder. The **dataviz.py** has some suggestions for visualization of the results, that are not saved, which means, you can view and save only if you want.

- Actually, this repository has the **figures**, **results** and **TrainedModels** folders filled with the last running outputs used to write the paper. The subsets (processed records) used to train and test the models are available by the link <https://phc.st/edgeiiotset>. The paper URL will be provided after publication at [LATINCOM 2022](https://latincom2022.ieee-latincom.org/program/technical-sessions/).
