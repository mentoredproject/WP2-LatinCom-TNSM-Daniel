import requests
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


import tensorflow as tf
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Flatten, Conv1D, LSTM, Dropout
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# will get pre defined values from a xlsx file stored in gdrive

r = requests.get(
    'https://drive.google.com/uc?export=download&id=157jtLqUtpmGp085ZbysSjG_777hHgI5P')
with open('IoT-IIoT.Definitions.xlsx', 'wb') as f:
    f.write(r.content)

plt.clf()
pltshow = ''
classes = pd.read_excel('./IoT-IIoT.Definitions.xlsx', sheet_name='Classes')


def Multi_Binary(Y_uniques):
    if len(Y_uniques) == 2:
        lossfun = 'binary_crossentropy'
        lastlayernodes = 1
        activfun = 'sigmoid'
    else:
        lossfun = 'sparse_categorical_crossentropy'
        activfun = 'softmax'
        lastlayernodes = len(Y_uniques)
    return lossfun, activfun, lastlayernodes


def Models_proposed(pset='undefined', Y_uniques=[0, 1], inputshape=[]):
    lr = 5e-4
    kernelsize = 5
    print('Build the proposed neural network')

    lossfun, activfun, lastlayernodes = Multi_Binary(Y_uniques)

    NN_model = Sequential(name=f'Proposed_Hybrid_CNN_LSTM.p{pset}.{activfun}')
    NN_model.add(Conv1D(64, kernel_size=kernelsize,
                 input_shape=inputshape, activation='relu'))
    NN_model.add(LSTM(128))
    NN_model.add(Dense(128, activation='relu'))
    NN_model.add(Dropout(0.5))
    NN_model.add(Dense(32, activation='relu'))
    NN_model.add(Flatten())
    NN_model.add(Dense(lastlayernodes, activation=activfun))
    NN_model.compile(loss=lossfun, optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr), metrics=['accuracy'])
    NN_model.build()
    # print(NN_model.summary())
    return NN_model


def Models_Inspired(pset='undefined', Y_uniques=[0, 1], inputshape=[]):
    lr = 1e-3  # default from keras
    kernelsize = 5
    print('Building the inspierated neural network')

    lossfun, activfun, lastlayernodes = Multi_Binary(Y_uniques)

    NN_model = Sequential(name=f'Inspired_CNN_LSTM.p{pset}.{activfun}')
    NN_model.add(Conv1D(64, kernel_size=kernelsize,
                 input_shape=inputshape, activation='relu'))
    NN_model.add(LSTM(128))
    NN_model.add(Dropout(0.5))
    NN_model.add(Dense(32, activation='relu'))
    NN_model.add(Dense(lastlayernodes, activation=activfun))
    NN_model.compile(loss=lossfun, optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr), metrics=['accuracy'])
    NN_model.build()
    # print(NN_model.summary())
    return NN_model


def Models_AuthorsDataset(pset='undefined', Y_uniques=[0, 1], inputshape=[]):
    lr = 1e-2
    print('Building the authors dataset neural network')
    lossfun, activfun, lastlayernodes = Multi_Binary(Y_uniques)

    # NN_model . add( Input( shape =i nputshape))
    NN_model = Sequential(name=f'Dataset_Authors.p{pset}.{activfun}')
    NN_model.add(Dense(inputshape[0], input_dim=inputshape[0],
                 activation='relu', kernel_regularizer=l2()))
    NN_model.add(Dense(60, activation='relu', kernel_regularizer=l2()))
    NN_model.add(Dense(30, activation='relu', kernel_regularizer=l2()))
    NN_model.add(Dense(lastlayernodes, activation=activfun))
    NN_model.compile(loss=lossfun, optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr), metrics=['accuracy'])
    NN_model.build()
    # print(NN_model.summary())
    return NN_model


def Models_Train(model, X_train, y_train, Batchsize=768, Epochs=3, ):
    if model.name.upper().find('CNN') > 0 or model.name.upper().find('HYBRID') > 0 or model.name.upper().find('LSTM'):
        print('Reshapping...')
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    print('Fitting the neural network')

    # (it is better use the same name to have the best model on top/last)#   model.name+'.Weights-{epoch:03d}-{val_accuracy:.5f}.hdf5'
    checkpoint_name = 'TrainedModels/'+model.name+'.hdf5'

    McheckpointLOSS = ModelCheckpoint(
        filepath=checkpoint_name, monitor='loss', mode='auto', save_best_only=True, verbose=1)
    McheckpointACC = ModelCheckpoint(
        filepath=checkpoint_name, monitor='accuracy', mode='auto', save_best_only=True, verbose=1)
    MearlystopLOSS = EarlyStopping(
        monitor='loss', mode='auto', patience=4, min_delta=0.002, verbose=1)  # 0.0652
    MearlystopACC = EarlyStopping(
        monitor='accuracy', mode='auto', patience=4, min_delta=0.0005, verbose=1)  # 0.0652
    callbacks_list = [McheckpointLOSS,
                      MearlystopACC, MearlystopLOSS, MearlystopACC]

    #model = Models_proposed(len(yuniques), inputshape=[X_train.shape[1], 1])
    mhistory = model.fit(X_train, y_train, epochs=Epochs, batch_size=Batchsize,
                         validation_split=0.2, verbose=1, callbacks=callbacks_list)

    # print(mhistory.history)
    plt.style.use("ggplot")
    plt.figure()
    N = len(mhistory.history['loss'])
    plt.plot(np.arange(0, N), mhistory.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), mhistory.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), mhistory.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N),
             mhistory.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.tight_layout()
    # plt.savefig(f'./figures/{model.name}_ModelHistory.png')
    plt.clf()
    return mhistory


def Models_Evaluate(model, x_test, y_test, y_uniques, y_target):
    if model.name.upper().find('CNN') > 0 or model.name.upper().find('HYBRID') > 0 or model.name.upper().find('LSTM'):
        print('Reshapping...')
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    y_pred = model.predict(x_test, verbose=1)
    if len(y_uniques) > 2:
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = (y_pred.reshape(y_pred.shape[0]) > 0.5).astype('int32')

    reversefactor = dict(zip(range(len(y_uniques)), y_target))
    y_testvec = np.vectorize(reversefactor.get)(y_test)
    y_predvec = np.vectorize(reversefactor.get)(y_pred)
    ctnn = pd.crosstab(y_testvec, y_predvec, rownames=[
                       'Actual Classes'], colnames=['Predicted Classes'])
    print(ctnn)
    creport = classification_report(
        y_test, y_pred, target_names=y_target, zero_division=0, output_dict=True, digits=2)
    # print(creport)
    hmx = sns.heatmap(ctnn/np.sum(ctnn), annot=True,
                      cbar=False, fmt='.2f', cmap='Blues')
    plt.xticks(rotation=45)
    if len(y_uniques) > 2:
        hmx.set_title("Multi-Class Classification Confusion Matrix")
    else:
        hmx.set_title("Binary Classification Confusion Matrix")
    #hmx.set_ylabel("Real Labels")
    #hmx.set_xlabel("Predicted Labels")
    plt.tight_layout()
    plt.savefig(f'./figures/{model.name}_CMatrix.png')
    plt.clf()
    return(creport)


def LoadSetPart(Set_number=0, TestSet=False):
    print('loadding subset:', Set_number)
    gzipfile = f'iiot-full.SubSet-p{Set_number}.gzip'

    if TestSet:
        gzipfile = f'iiot-full.TestSet.gzip'

    start = time.time()
    set_df = pd.DataFrame()
    try:
        set_df = pd.read_parquet('../SubSets/'+gzipfile)
    except:
        set_df = pd.read_parquet('./SubSets/'+gzipfile)
    print('Time to load file:', time.time() - start)
    print("Shuffling..")
    set_df = set_df.sample(frac=1, random_state=43).reset_index(drop=True)

    return set_df

# *****************************************************************************************************


def SaveReport(resultreport, mode, name, set, res):
    name = name.split('.')[0]
    resultreport.append({'mode': mode, 'name': name, 'set': set, 'accuracy': res['accuracy'], 'precision': res['macro avg']['precision'],
                        'recall': res['macro avg']['recall'], 'f1-score': res['macro avg']['f1-score'], 'support': res['macro avg']['support'], 'report': res})
    pd.DataFrame(resultreport).to_csv(
        f'result_report.{mode}-Parcial.csv', index=False)
    print(name, 'accuracy:', res['accuracy'])


def PrepareData(set_df, YClass):
    # if it is Normal or Attack
    set_df_label = set_df['Attack_label']

    # Normal and others 14 type of attacks
    set_df_type = set_df['Attack_type']

    # Same of attack_type, but the Normal class is splitted with the IoT device name
    set_df_device = set_df['IoT_Device']

    clabel = classes.loc[classes.Class_Type == 'Binary']
    ctype = classes.loc[classes.Class_Type == 'Attack']
    cdevice = classes.loc[classes.Class_Type == 'Device']

    if(YClass.upper() == 'LABEL'):  # label feature is already number values
        #    set_df.replace(classes['Class_Name'],classes['Class_Code'], inplace=True)
        y_uniques = clabel['Class_abbr'].values  # set_label_yuniques
        Y_data = set_df['Attack_label']
        y_target = y_uniques  # Classes_Binary
    elif(YClass.upper() == 'TYPE'):
        y_uniques = ctype['Class_abbr'].values  # set_types_yuniques
        Y_data = set_df['Attack_type'].replace(
            ctype['Class_Name'].values, ctype['Class_Code'].values)
        y_target = y_uniques  # Classes_Multi
    elif(YClass.upper() == 'DEVICE' or YClass.upper() == 'IOT_DEVICE' or YClass.upper() == 'IOT'):
        #print("***********************************arrumar aki")
        set_df.loc[set_df.Attack_label == 1, ['IoT_Device']
                   ] = 'Attack'  # set all attack class as the same

        y_uniques = cdevice['Class_abbr'].values  # set_devce_yuniques
        Y_data = set_df['IoT_Device'].replace(
            cdevice['Class_Name'].values, cdevice['Class_Code'].values)
        Y_data.replace(ctype['Class_Name'].values, cdevice.loc[cdevice.Class_Name ==
                       'Attack']['Class_Code'].values[0], inplace=True)
        y_target = y_uniques  # set_devce_yuniques

    set_df.drop(['Attack_label', 'Attack_type',
                 'IoT_Device'], axis=1, inplace=True)

    X_data = set_df.to_numpy()
    return X_data, Y_data, y_target, y_uniques


def TrainModels(p_init=0, p_fim=9, mode="LABEL"):
    print(f'Starting to train all models with Y={mode}.')
    resultreport = []
    # load the test subset to evaluate the models
    test_df = LoadSetPart(TestSet=True)
    x_test, y_test, y_target, y_uniques = PrepareData(test_df, mode)
    for i in range(p_init, p_fim+1):
        row_report = []
        set_df = LoadSetPart(i)  # load the subset to train the models
        # no problem to set again the y_target and y_uniques since they are the same
        X_train, y_train, y_target, y_uniques = PrepareData(set_df, mode)

        print(i, 'Proposed. ********************************************************')
        mdls_proposed_B = Models_proposed(
            i, y_uniques, inputshape=[X_train.shape[1], 1])
        Models_Train(mdls_proposed_B, X_train=X_train,
                     y_train=y_train, Batchsize=768, Epochs=30)
        res = Models_Evaluate(mdls_proposed_B, x_test,
                              y_test, y_uniques, y_target)
        SaveReport(resultreport, mode, mdls_proposed_B.name, i, res)

        print(i, 'Inspired. ********************************************************')
        mdls_Inspired_B = Models_Inspired(
            i, y_uniques, inputshape=[X_train.shape[1], 1])
        Models_Train(mdls_Inspired_B, X_train=X_train,
                     y_train=y_train, Batchsize=768, Epochs=30)
        res = Models_Evaluate(mdls_Inspired_B, x_test,
                              y_test, y_uniques, y_target)
        SaveReport(resultreport, mode, mdls_Inspired_B.name, i, res)

        print(i, 'Authors. ********************************************************')
        mdls_authorsdataset_B = Models_AuthorsDataset(
            i, y_uniques, inputshape=[X_train.shape[1]])
        Models_Train(mdls_authorsdataset_B, X_train=X_train,
                     y_train=y_train, Batchsize=800, Epochs=30)
        res = Models_Evaluate(mdls_authorsdataset_B,
                              x_test, y_test, y_uniques, y_target)
        SaveReport(resultreport, mode, mdls_authorsdataset_B.name, i, res)

    print('Saving complete report...')
    pd.DataFrame(resultreport).to_csv(
        f'result_report.{mode}-Final.csv', index=False)
    return resultreport


# will train and test the binray models
print("Will train and test all models with all subsets for binary classification.")
binary = TrainModels(0, 7, 'Label')

# will train and test the multiclass models
print("Will train and test all models with all subsets for multiclass classification.")
multiclass = TrainModels(0, 7, 'Type')

exit()
