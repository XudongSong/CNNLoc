from encoder_model import EncoderDNN
import numpy as np
import data_helper
import time
import keras
import csv
import matplotlib.pyplot as plt
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]='0'
from keras.backend.tensorflow_backend import set_session
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.5
set_session(tf.Session(config=config))


rng=np.random.RandomState(888)
# np.random.seed(423453)
base_dir= os.getcwd()
# train_csv_path = os.path.join(base_dir,'trainingData.csv')
test_csv_path=os.path.join(base_dir,'validationData.csv')
valid_csv_path=os.path.join(base_dir,'AllValuationData.csv')
train_csv_path=os.path.join(base_dir,'arrAllTrainingData.csv')

log_dir='DEEPLEARNING_MODEL_log.txt'

if __name__ == '__main__':
    # Load data
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data_helper.load_data_all(train_csv_path, valid_csv_path,test_csv_path)
    # train_x, train_y, valid_x, valid_y, test_x, test_y=data_helper.load(train_csv_path, test_csv_path)

    # patience=[i for i in range(1,50,2)]
    # patience=[21,]

    # B=[i for i in np.linspace(3.0,3.1,2)]
    # for b in B:
    # for p in patience:



    # dropout=[i for i in np.linspace(0.4,0.7,4)]
    dropout=[0.7,]
    for dp in dropout:

        # save_picture_dir = "pict_dp="+str(dp)+"fixeddropout=0.7,comparelr32"
        # os.mkdir('pictures/all/' + save_picture_dir)
        with open(log_dir, 'a') as file:
            file.write('\n' + "**" * 10 + '%%%%%ALL%%%%%dp='+str(dp)+' Start' + "**" * 10)

        # adam_lr = [i for i in np.linspace(0.00001, 0.001, 6)]
        adam_lr=[0.0001,]

        for lr in adam_lr:
            # dp = 0.7
            b=2.8
            p=3
            epoch_sae=10
            epoch_floor=10
            epoch_position=10

            results = []
            position_results=[]
            echo = 1
            while echo>0:
                echo-=1
                name="fixed validdata, beta:"+str(b)+"patience:"+str(p)+"dropout rate:"+str(dp)+"No."+str(echo)+"learning rate:"+str(lr)
                # Training
                encode_dnn_model = EncoderDNN()
                encode_dnn_model.patience=int(p)
                encode_dnn_model.b=b
                encode_dnn_model.epoch_AE=epoch_sae
                encode_dnn_model.epoch_floor=epoch_floor
                encode_dnn_model.epoch_position=epoch_position
                encode_dnn_model.dropout=dp
                encode_dnn_model.adam_lr=lr
                strat = time.time()
                # tbCallBack=keras.callbacks.TensorBoard(log_dir='./Graph',
                #                                        histogram_freq=1,
                #                                        write_graph=True,
                #                                        write_images=True)

                h=encode_dnn_model.fit(train_x, train_y, valid_x=valid_x, valid_y=valid_y)#,tensorbd=tbCallBack)
                end=time.time()

                if not isinstance(h[0],int):

                    # Plot training & validation loss values
                    with open(log_dir,'a') as f:
                        f.write('\n\nSAE_training_log bpde'+name+'\n')
                        f.write('training loss:\t'+str(h[0].history['loss']))
                        f.write('\nvalid loss:\t'+str(h[0].history['val_loss']))
                    # plt.plot(h[0].history['loss'])
                    # plt.plot(h[0].history['val_loss'])
                    # plt.title('SAE Model loss')
                    # plt.ylabel('Loss')
                    # plt.xlabel('Epoch')
                    # plt.legend(['Train', 'Test'], loc='upper left')
                    # plt.savefig('pictures/all/'+save_picture_dir+'/bpde'+name+'saeLoss.png')
                    # plt.clf()

                if not isinstance(h[1],int):
                    # Plot training & validation accuracy values
                    with open(log_dir,'a') as f:
                        f.write('\n\ndropout rate='+name)
                        f.write('\n\nFloor_training_acc_log bpde'+name+'\n')
                        f.write('training acc:\n'+str(h[1].history['acc'])[1:-1])
                        f.write('\nvalid acc:\n'+str(h[1].history['val_acc'])[1:-1])
                    # plt.plot(h[1].history['acc'])
                    # plt.plot(h[1].history['val_acc'])
                    # plt.title('Floor Model accuracy')
                    # plt.ylabel('Accuracy')
                    # plt.xlabel('Epoch')
                    # plt.legend(['Train', 'Test'], loc='upper left')
                    # plt.savefig('pictures/all/'+save_picture_dir+'/bpde'+name+'flooracc.png')
                    # plt.clf()
                    # Plot training & validation loss values
                    with open(log_dir,'a') as f:
                        f.write('\n\nFloor_training_loss_log bpde'+name+'\n\n')
                        f.write('training loss:\n'+str(h[1].history['loss'])[1:-1])
                        f.write('\nvalid loss:\n'+str(h[1].history['val_loss'])[1:-1])
                    # plt.plot(h[1].history['loss'])
                    # plt.plot(h[1].history['val_loss'])
                    # plt.title('Floor Model loss')
                    # plt.ylabel('Loss')
                    # plt.xlabel('Epoch')
                    # plt.legend(['Train', 'Test'], loc='upper left')
                    # plt.savefig('pictures/all/'+save_picture_dir+'/bpde'+name+'floorLoss.png')
                    # plt.clf()

                if not isinstance(h[2], int):
                    # Plot training & validation loss values
                    with open(log_dir,'a') as f:
                        f.write('\n\nLocation_training_log bpde'+name+'\n\n')
                        f.write('training loss:\n'+str(h[2].history['loss'])[1:-1])
                        f.write('\nvalid loss:\n'+ str(h[2].history['val_loss'])[1:-1])
                    plt.plot(h[2].history['loss'])
                    plt.plot(h[2].history['val_loss'])
                    plt.title('Location Model loss')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.legend(['Train', 'Test'], loc='upper left')
                    plt.savefig('LocationLoss.png')
                    plt.clf()

                building_right, floor_right, longitude_error, latitude_error, mean_error,right_error=encode_dnn_model.error(test_x, test_y)


                print('build accuracy:',str((building_right/1111.0)*100)+'%')
                print('floor_accuracy',str((floor_right/1111.0)*100)+'%')
                print('position_error',str(mean_error)+'m')
                print('floor_right_position_error',str(right_error)+'m')
                print('long_err:',longitude_error)
                print('lati_err:',latitude_error)
                print('time:',end-strat)
                results.append((floor_right/1111.0)*100)
                position_results.append([longitude_error,latitude_error,mean_error])
                # if (floor_right/1111.0)*100>96:
                #     break

                del encode_dnn_model

            with open(log_dir,'a') as file:
                file.write('\nlog bpde'+name)
                file.write('\nfloor_result')
                file.write('\n'+str(results)+','+str(np.mean(results)))
                file.write('\nlocation_result')
                file.write('\n'+str(position_results))
                file.write('\n'+"##"*10+'Dropout='+str(dp)+"End"+"##"*10)
            print(results)
            print(np.mean(results))
            print(position_results)