########################################################################################################################
# PINN method for HRTF interpolation 
#  
# Given HRTF from limited number of directions       (330 training hrtf), 
# we wish to interpolate the HRTF over a large direction (1260 total hrtf )
# 
# Fei Ma,   
# feima1024@gmail.com
# 4th, Oct, 2023 
########################################################################################################################
# import the python packages 
import tensorflow as tf
from tensorflow.keras import activations
from keras import backend as K
import logging 
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
from   time import time as now
from datetime import datetime
import scipy.optimize
import scipy.io as sio
import random
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)
for ii in range(50): 
    print(">>>")
########################################################################################################################












######################################################################################################################## 
###  the definiton of the PINN model 
############################################################# 
### neural network initialization 
def init_model(num_input=3, layers = 3, neurons=3):     ## (x,y,z) input number is 3, layer number =  depth L,  neuron number =  width W 
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(num_input))                ## input layer 
    for ii in range(layers):                            ## hidden layers 
        model.add(tf.keras.layers.Dense(neurons,activation=tf.keras.activations.get('tanh'),kernel_initializer='glorot_normal'))
    model.add(tf.keras.layers.Dense(1))                 ## output layer 
    return model
############################################################# 
### calculate the PDE loss 
def get_pde(model,pde_input,wave_num1):
    with tf.GradientTape(persistent=True) as tape:
        x1, x2, x3 = pde_input[:,0:1], pde_input[:,1:2], pde_input[:,2:3] # (x1,x2,x3) = (x,y,z) 
        tape.watch(x1)                                  ## notify tensorflow that we care about the gradient with respect to x1, x2, x3
        tape.watch(x2)
        tape.watch(x3)
        pde_pred  = model(tf.stack([x1[:,0],x2[:,0],x3[:,0]],axis=1))   # mode prediction 
        x1_d1 = tape.gradient(pde_pred,x1)              ## first order gradient with respect to x, y, z
        x2_d1 = tape.gradient(pde_pred,x2)
        x3_d1 = tape.gradient(pde_pred,x3)
    x1_d2 = tape.gradient(x1_d1,x1)                     ## second order gradient with respect to x, y, z
    x2_d2 = tape.gradient(x2_d1,x2)
    x3_d2 = tape.gradient(x3_d1,x3)
    del tape
    Laplacian = ( x1_d2 + x2_d2 + x3_d2 )*wave_num1              ##  wave_num1 = (1/(omega/c))^2, this normalize the laplacian 
    loss_pde = tf.reduce_mean(tf.square(Laplacian + pde_pred))  ## this line of code calculate the PDE loss  
    return loss_pde                                             ## return the PDE loss 
#############################################################
### calculate the data loss                                     
def get_data(model,data_input,data_target):                     
    data_pred  = model(data_input)                                ## mode prediction 
    loss_data  = tf.reduce_mean(tf.square(data_pred-data_target)) ## calcualte the difference between training data and the prediction, result in the data loss 
    return loss_data                                              ## return the data loss  
############################################################# 
### calculat the gradient with respect to the loss  
def get_grad(model,data_input,data_target,pde_input,wave_num1):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        loss_data  = get_data(model,data_input,data_target)     ## data loss   
        loss_pde  = get_pde(model,pde_input,wave_num1)          ## pde loss  
        loss      = loss_data + loss_pde                        ## total loss 
    g = tape.gradient(loss,model.trainable_variables)           ## take the gradient of the trainable parameters with respect to the loss
    del tape
    return loss_data, loss_pde, g                               ## return the data_loss, pde_loss, and the gradient  
############################################################# 
### predict the HRTF for the test_input coordiantes 
def get_test(model,test_input):
    test_pred  = model(test_input)                              ## model prediction 
    return test_pred                                            ## return the prediction 
##############################################################################################################







########################################################################################################################
# some global variables 
speed      = 343;               # speed of sound 
lr         = 1e-3;              # adam learning rate            
num_epochs = 100*1000;          # training epoches 
layers     = 3                  # PINN depth
########################################################################################################################







########################################################################################################################
## this is the training process 
for human in range(40,41):      ##  human denote the ID of subjects, iterative over subject 11 to 50 
    ################################################################################################################
    ### prepare the training data 
    file       = str(human)+'.mat' 
    data       = sio.loadmat(file);
    #######################################
    total_hrtf = data['total_hrtf'];   ## known HRTF + unknown HRTF = total HRTF, 
                                       ## total HRTF is a [7 , 2 , 1260] tensor
                                       ## 7 is the number of frequency bins of interest 

                                       ## total_hrtf[ ff, 0, 0:1260] is the real part of total HRTF at frequency ff
                                       ## total_hrtf[ ff, 1, 0:1260] is the imag part of total HRTF at frequency ff

                                       ## total_hrtf[ ff, 0, 0:630] is the real and left part of total HRTF at frequency ff
                                       ## total_hrtf[ ff, 0, 0:630] is the real and right part of total HRTF at frequency ff
                                       ## total_hrtf[ ff, 1, 630:1260] is the imag and left part of total HRTF at frequency ff
                                       ## total_hrtf[ ff, 1, 630:1260] is the imag and right  part of total HRTF at frequency ff

    total_est  = np.copy(total_hrtf);  ## a tensor the same size as total_hrtf, it will store the total_hrtf estimation


    train_hrtf = data['train_hrtf'];   ## known HRTF only 
                                       ## train HRTF is a [7 , 2 , 330] tensor
                                       ## 7 is the number of frequency bins of interest 

                                       ## train_hrtf[ ff, 0, 0:330] is the real part of train HRTF at frequency ff
                                       ## train_hrtf[ ff, 1, 0:330] is the imag part of train HRTF at frequency ff

                                       ## train_hrtf[ ff, 0, 0:165] is the real and left part of train HRTF at frequency ff
                                       ## train_hrtf[ ff, 0, 0:165] is the real and right part of train HRTF at frequency ff
                                       ## train_hrtf[ ff, 1, 165:330] is the imag and left part of train HRTF at frequency ff
                                       ## train_hrtf[ ff, 1, 165:330] is the imag and right  part of train HRTF at frequency ff



    total_coor = data['total_coor'];   ## coordiantes of the known and unkown HRTFs 
                                       ## total_coor is a [1260,8] tensor 
                                       ## total_coor[ii,0:8] is the coordiantes of the ii-th hrtf  
                                       ## total_coor[ii,0:3] = [x, y,  z]
                                       ## total_coor[ii,3:6] = [r, theta, phi] in radian 
                                       ## total_coor[ii,6:8] = [   theta, phi] in degree 


    train_coor = data['train_coor'];   ## coordiantes of the known HRTF only  
                                       ## train_coor is a [330,8] tensor 
                                       ## train_coor[ii,0:8] is the coordiantes of the ii-th hrtf  
                                       ## train_coor[ii,0:3] = [x, y,  z]
                                       ## train_coor[ii,3:6] = [r, theta, phi] in radian 
                                       ## train_coor[ii,6:8] = [   theta, phi] in degree 
    ########################################
    freq_bb    = data['freq_bins'];    ## vectors saved by matlab into mat format will be 2D arrays when read by python
                                       ## var[0] will get the value of the vector
    freq_bins  = freq_bb[0]            ## a vector of frequency bins we are going to evaluate 
                                       ## [2.1 4.1 6.2 8.2 10.3 12.3 14.4] kHz 
    ########################################
    total_num  = data['total_num'];    ## total number of known and unknown HRTFs = 1260 
    train_num  = data['train_num'];    ##                   number of known  HRTF = 330 
    total_num  = total_num[0][0]       ## variables saved by matlab into mat format will be 2D arrays when read by python
    train_num  = train_num[0][0]       ## var[0][0] will get the value of the variable 

    total_mid  = total_num//2          ## total_hrtf[ ff, 0,         0:total_mid]  real and left  part of total hrtf    
                                       ## total_hrtf[ ff, 0, total_mid:total_num]  real and right part of total hrtf    
                                       ## total_hrtf[ ff, 1,         0:total_mid] imagl and left  part of total hrtf    
                                       ## total_hrtf[ ff, 1, total_mid:total_num] imagl and right part of total hrtf    


    train_mid  = train_num//2          ## train_hrtf[ ff, 0,         0:train_mid]  real and left  part of train hrtf    
                                       ## train_hrtf[ ff, 0, train_mid:train_num]  real and right part of train hrtf    
                                       ## train_hrtf[ ff, 1,         0:train_mid] imagl and left  part of train hrtf    
                                       ## train_hrtf[ ff, 1, train_mid:train_num] imagl and right part of train hrtf    
    ################################################################################################################
    for ff in range(0,7):             ## iterative over frequency 
        ###############################
        ### the wave number 
        freq       = freq_bins[ff];             ## the current frequency 
        wave_num   = 2*np.pi*freq/speed;        ## the wave number 
        wave_num1  = 1.0/(wave_num**2);         ## a factor used for normalizing the Laplacian 
        ###############################
        ### the node number                     ## calculate the number of  neurons in hidden layer according the frequency  
        nodes = 0;
        if freq<3000:                           ##  f<3000, neuron = f/500 
            nodes = int(np.ceil(freq/500));
        elif freq>6000:
            nodes = int(np.ceil(freq/1000));    ##  f>6000, neuron = f/1000 
        else:
            nodes = 6;                          ## else neuron = 6 
    ##########################################################################################################################################
        for dd in range(4):                     ## 4 pinn methods to model the [real left], [real right], [imaginary left], [imaginar right] part of HRTF 
            if dd==0:                           ## real left
                ####################################################################################################
                data_train = np.zeros((train_mid,1));                   ###  use the real and left known HRTF as training data 
                xyz_train  = np.zeros((train_mid,3))                    ###  the real and left known HRTF's cartesian coordinates  
                xyz_total  = np.zeros((total_mid,3))                    ###  the real and left [known + unknown] total HRTF's cartesian coordinates  
                for ii in range(train_mid):
                    data_train[ii,0]  = train_hrtf[ff,0,ii]             ###  get the traning data 
                    xyz_train[ii,0:3] = train_coor[ii,0:3]              ###  and the corresponding coordinates  
                for ii in range(total_mid):
                    xyz_total[ii,0:3] = total_coor[ii,0:3]              ###  get the  coordiantes used for PDE loss calculation 
                ####################################################################################################
            elif dd==1:                         ## real right
                ####################################################################################################
                data_train  = np.zeros((train_num-train_mid,1));        ### same as above but for the real and right part 
                xyz_train  = np.zeros((train_num-train_mid,3))
                xyz_total  = np.zeros((total_num-total_mid,3))
                for ii in range(train_mid,train_num):
                    data_train[ii-train_mid,0]   = train_hrtf[ff,0,ii] 
                    xyz_train[ii-train_mid,0:3] = train_coor[ii,0:3] 
                for ii in range(total_mid,total_num):
                    xyz_total[ii-total_mid,0:3] = total_coor[ii,0:3] 
                ####################################################################################################
            elif dd==2:                         ## imag left
                ####################################################################################################
                data_train  = np.zeros((train_mid,1));                  ### same as above but for the imaginary and left part 
                xyz_train  = np.zeros((train_mid,3))
                xyz_total  = np.zeros((total_mid,3))
                for ii in range(train_mid):
                    data_train[ii,0]   = train_hrtf[ff,1,ii] 
                    xyz_train[ii,0:3] = train_coor[ii,0:3] 
                for ii in range(total_mid):
                    xyz_total[ii,0:3] = total_coor[ii,0:3] 
                ####################################################################################################
            else:                               ## imag right
                ####################################################################################################
                data_train  = np.zeros((train_num-train_mid,1));        ### same as above but for the imaginary and right part 
                xyz_train  = np.zeros((train_num-train_mid,3))
                xyz_total  = np.zeros((total_num-total_mid,3))
                for ii in range(train_mid,train_num):
                    data_train[ii-train_mid,0]   = train_hrtf[ff,1,ii] 
                    xyz_train[ii-train_mid,0:3] = train_coor[ii,0:3] 
                for ii in range(total_mid,total_num):
                    xyz_total[ii-total_mid,0:3] = total_coor[ii,0:3] 
    #########################################################################################################################################
            xyz_train  = tf.convert_to_tensor(xyz_train, dtype=tf.float32)              ### transfer the numpy data into tensorflow data, float32 format 
            data_train = tf.convert_to_tensor(data_train, dtype=tf.float32)
            xyz_total  = tf.convert_to_tensor(xyz_total, dtype=tf.float32)
            wave_num1  = tf.convert_to_tensor(wave_num1, dtype=tf.float32)
    #########################################################################################################################################
    ## the core training process 
            now_err = 0                                                                 ### record the current data loss 
            ### the PINN training is sensitive to network initialization, the training is repeated five times
            ### we select the training with the least data loss as the training result 
            for cc in range(5):                                                         
                #####################################################################################################
                ### this line of code will clear the memory used by a model after it finish 
                ### with out it, you will run out of memory quickly 
                tf.keras.backend.clear_session()            
                #####################################################################################################
                ###  tell tensorflow to build up a static graph for the model_fit function 
                ###  the core model fitting/traning function 
                @tf.function
                def model_fit(model,data_input,data_target,pde_input,wave_num1):
                    loss_data,loss_pde,grad=get_grad(model,data_input,data_target,pde_input,wave_num1)
                    optim.apply_gradients(zip(grad,model.trainable_variables))
                    return loss_data, loss_pde
                #####################################################################################################
                model      = init_model(3,layers,nodes)   ## initialize a model with  input number, hidden layer number, and nodes number 
                optim      = tf.keras.optimizers.legacy.Adam(learning_rate = lr)       ### we use the ADAM optimizer
                db_true    = 10*np.log10(tf.reduce_mean(tf.square(data_train)))/0.1/10 ### energy of the training HRTF in dB 
                #####################################################################################################  
                ### let us train 
                for jj in range(1,num_epochs):
                    loss_data,loss_pde=model_fit(model,xyz_train,data_train,xyz_total,wave_num1)
                #####################################################################################################  
                loss_data_db = (10*np.log10(loss_data) - db_true)//0.1/10           ## the data loss 
                loss_pde_db  = 10*np.log10(loss_pde)//0.1/10                        ## pde loss 
                now          = datetime.now()                                       ## current date and time
                now          = now.strftime("%H:%M:%S")
                ## print  data loss,  pde loss, layer number, neuron number, for subject huamn at frequency ff
                print(ff,'ID',human,'T',now,'L',layers,'N',nodes,'R',dd,loss_data_db,loss_pde_db) 
                #######################################################################################################
                ### if the current data loss is smaller, we store the data loss 
                if loss_data_db < now_err:
                    #########################################################
                    now_err   = loss_data_db;               ### store the data loss 
                    pinn_pred = get_test(model,xyz_total);  ### predict the HRTF at [known + unknown] total HRTF coordiantes 
                    pinn_pred = pinn_pred.numpy();          ### transfer the prediction into numpy format 
                    #########################################################
                    if dd==0:
                        for ii in range(total_mid):                      ### store the real left HRTF prediction into the total_est
                            total_est[ff,0,ii] = pinn_pred[ii][0]
                    elif dd==1: 
                        for ii in range(total_mid,total_num):            ### store the real right HRTF prediction into the total_est
                            total_est[ff,0,ii] = pinn_pred[ii-total_mid][0]
                    elif dd==2:
                        for ii in range(total_mid):                      ### store the imag left HRTF prediction into the total_est
                            total_est[ff,1,ii] = pinn_pred[ii][0]
                    else:
                        for ii in range(total_mid,total_num):            ### store the imag right HRTF prediction into the total_est
                            total_est[ff,1,ii] = pinn_pred[ii-total_mid][0]
                #######################################################################################################
                ### good enough, stop 
                if now_err<-29.0:           ## if the current data loss is small enough, we stop traning 
                    break; 
        print('------------------------------------------') 
    ################################################################################################## 
    ### save the result into a file 
    newfile    = str(human) + '_L' + str(layers) + '.mat' 
    sio.savemat(newfile,{'total_hrtf':total_hrtf,'total_est':total_est,'total_coor':total_coor,'train_coor':train_coor}); 
    #################################################################################################################### 




















