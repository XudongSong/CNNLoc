from New2_encoder_multi_model import EncoderDNN
import numpy as np
import data_helper
import time
import keras
import csv
import matplotlib.pyplot as plt
import os
import pandas as pd

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]='2'
from keras.backend.tensorflow_backend import set_session
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.6
set_session(tf.Session(config=config))


rng=np.random.RandomState(666)
base_dir= os.getcwd()
# train_csv_path = os.path.join(base_dir,'trainingData.csv')
test_csv_path=os.path.join(base_dir,'validationData.csv')
valid_csv_path=os.path.join(base_dir,'AllValuationData.csv')
train_csv_path=os.path.join(base_dir,'arrAllTrainingData.csv')
allTrain_csv_path=os.path.join(base_dir,'trainingData.csv')
expandedTrain_csv_path=os.path.join(base_dir,'all_training_data.csv')
log_dir='New_Auto_DEEPLEARNING_MODEL_log.txt'



compare_list=[]
if __name__ == '__main__':
    # Load data
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data_helper.load_data_all(train_csv_path, valid_csv_path,test_csv_path)
    # # train_x, train_y, valid_x, valid_y, test_x, test_y=data_helper.load(train_csv_path, test_csv_path)
    # (train_x,train_y)=data_helper.load_data_perspective(allTrain_csv_path)
    # (train_x, train_y) = data_helper.load_data_perspective(expandedTrain_csv_path)
    encode_dnn_model = EncoderDNN()

    SAE=[
        [[256,128, 64], [128,256, 520]],
        [[256, 128, 64], [128, 256, 520]],
        [[256, 128, 64], [128, 256, 520]],
        [[256, 128, 64], [128, 256, 520]],
        [[256, 128, 64], [128, 256, 520]],
        [[256, 128, 64], [128, 256, 520]],
        [[256, 128, 64], [128, 256, 520]],
        [[256, 128, 64], [128, 256, 520]],
        [[256, 128, 64], [128, 256, 520]],
        [[256, 128, 64], [128, 256, 520]],
        [[256, 128, 64], [128, 256, 520]],
        [[256, 128, 64], [128, 256, 520]],
        [[256, 128, 64], [128, 256, 520]],
        [[256, 128, 64], [128, 256, 520]],
        [[256, 128, 64], [128, 256, 520]],
        [[256, 128, 64], [128, 256, 520]],

         # [[128,64],[128,520]],
         # [[256,64],[256,520]],
         # [[256,128,64],[128,256,520]],
         # [[512,128],[128,520]],
         # [[512,256,128],[256,512,520]],
         # [[512,256,64],[256,512,520]],
         # [[512,64],[512,520]],
         # [[512,256,128,64],[128,256,512,520]],
         # [[256,128],[256,520]],

         ]
    CNN=[
        [[99,22],[66,22],[33,22]],
         [[129,22],[96,22],[63,22]],
         [[99,33],[66,32]],
         [[66,33],[99,32]],
         [[99,53],[66,12]],
         [[119,22],[86,22],[53,22]],
         [[81,33],[81,32]],
         [[333,33],[66,32]],
         [[139,22],[106,22],[73,22]],
         [[99,13],[66,52]],
         [[66,22],[44,22],[22,22]],
         [[333,33],[33,32]],
         [[66,33],],
         [[66,6],],
         [[33,6],],
         [[149,22],[116,22],[83,22]],
        #
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],
         # [[99, 22], [66, 22], [33, 22]],



         ]

    # SAE = [
    #     [[128, 64], [128, 520]],
    # ]
    # CNN = [
    #     [[99, 22], [66, 22], [33, 22]],
    # ]

    for i in range(len(SAE)):
        sae, cnn=SAE[i],CNN[i]
        with open(log_dir, 'a') as file:
            file.write('\n' + "**" * 10 + 'SAE:'+str(sae)+'CNN:'+str(cnn)+' Start' + "**" * 10)


        lr=0.001
        bb=[2.7,2.7,2.7]
        p=3
        epoch_sae=40
        epoch_floor=20
        epoch_position=60
        dp=1
        regular=1
        results = []
        # echo = 1
        # while echo>0:
        #     echo-=1
        for echo,b in enumerate(bb):
            # Load data
            #********fixed verification data**************
            # type_data="fixed"
            # (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data_helper.load_data_all(train_csv_path,valid_csv_path,test_csv_path)
            #********random verification data**********
            # type_data="random"
            # train_x, train_y, valid_x, valid_y, test_x, test_y=data_helper.load(train_csv_path, test_csv_path)


            name="SAE:"+str(sae)+"CNN:"+str(cnn)+"No."+str(echo)
            # Training

            encode_dnn_model.patience=int(p)
            encode_dnn_model.b=b
            encode_dnn_model.epoch_AE=epoch_sae
            encode_dnn_model.epoch_floor=epoch_floor
            encode_dnn_model.epoch_position=epoch_position
            encode_dnn_model.dropout=dp
            encode_dnn_model.adam_lr=lr
            encode_dnn_model.regularlization=regular
            strat = time.time()


            h=encode_dnn_model.fitAuto(train_x, train_y, valid_x=valid_x, valid_y=valid_y,SAE_list=sae,CNN_list=cnn)#,tensorbd=tbCallBack)
            end=time.time()

            floor_right=encode_dnn_model.error(test_x, test_y)

            print('floor_accuracy',str((floor_right/1111.0)*100)+'%')

            print('time:',end-strat)
            results.append((floor_right/1111.0)*100)
            with open(log_dir, 'a') as f:
                f.write('\n\ndropout rate=' + name)
                f.write('\n\nFloor_training_acc_log bpde' + name + '\n')
                f.write('training acc:\n' + str(h.history['acc'])[1:-1])
                f.write('\nvalid acc:\n' + str(h.history['val_acc'])[1:-1])
            plt.plot(h.history['acc'])
            plt.plot(h.history['val_acc'])
            plt.title('Floor Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig('pictures/' + '/bpde' + name + 'flooracc.png')
            plt.clf()

        with open(log_dir,'a') as file:
            file.write('\n'+str(results)+','+str(np.mean(results)))

        print(results)
        print(np.mean(results))
        compare_list.append([str(sae),str(cnn),str(results[0]),str(results[1]),str(results[2]),str(np.mean(results))])
        with open(log_dir, 'a') as file:
            file.write('\n' + "**" * 10 + 'SAE:'+str(sae)+'CNN:'+str(cnn)+' end' + "**" * 10)
    compare = pd.DataFrame(data=compare_list,columns=['sae', 'cnn', 'th1', 'th2', 'th3', 'mean'])
    compare.to_csv('compare.csv')