# -*- coding: utf-8 -*-
"""
Guillaume laval: 31/08/2026
A Deep learning alforithm to jointly estimate
- s (selection coefficient)
- T	(osnset of selection)
- p0(frequency at the onset of selection)
"""


import os
print( os.getcwd() )


import sys
print ('Number of arguments:', len(sys.argv), 'arguments.');
print ('Argument List:', str(sys.argv));

jobfolder=sys.argv[1];
snp_name=sys.argv[2];
Ngenot=int(sys.argv[3]);

print ('jobfolder:', jobfolder);
print ('snp_name:', snp_name);
print ('Nstat:', Ngenot);
#exit();

# Import numpy& tensorflow
import numpy as np
import tensorflow as tf
# Import library to view images
import matplotlib.pyplot as plt
from scipy import stats






#############   LOADING MAIN SETTINGS       ##############################################
#
settings_main   ='SETTINGS_MAIN.txt'
myfile = open(settings_main, "r")
header = myfile.readline();
##### Reading the number of epochs for training
myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
SLICE=temp[1]; ### 'ALL' or 'diHH-Fst'; 

myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
first=int( temp[1] );

myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
last=int( temp[1] );

myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
POINTESTIM=temp[1]; ### 'AVERAGE' or 'MODE'

myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
pseudoempirical_only=temp[1]; ## 'TRAINING' or 'PSEUDOEMPIRICAL_ONLY';

myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
analyse=temp[1]; ## '100K' 

myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
file_location=temp[1]; ##

myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
output_location=temp[1]; ## 

myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
lineage=temp[1]; ## 'wAGR' etc...;

myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
pop=temp[1]; ## 'wAGR' etc...;


myfile.close()

##########################################################################################


###SLICE = 'ALL'; 
#SLICE = 'diHH-Fst'; 

if SLICE == 'ALL':
	#ALL stats
	input_shape = (Ngenot, 1); 

else:
	#Slice of stat (diHH + Fst)
	#first=1000
	#last=3500
	Ngenot=last-first
	input_shape = (Ngenot, 1); 
	
	print ('\tSLICE images :', SLICE);
	print ('\tRefefining the number of Statistics:', Ngenot);
	print ('\tNstat:', Ngenot);


### Initialization
my_num_filter=3;
my_kernel_size=10; 
nb_dense=128;
nb_dense1=256;
nb_dense2=512;
My_epochs=3;



if analyse == '100K': 
    #file_location=jobfolder ;
    output_location=jobfolder

    Nsim="100000" ; Ntraining=99700 ; Nvalid=100 ; Ncross=200; Nsim_crossval=Nsim
    #Nsim="100000" ; Ntraining=970 ; Nvalid=100 ; Ncross=200; Nsim_crossval=Nsim

else:
    ### BY DEFAULT : THE RETAINED MODEL
    file_location=jobfolder ; output_location=jobfolder

    Nsim="476800_CUSTOMTRAINING" ; Ntraining=9800 ; Nvalid=100 ; Ncross=200; Nsim_crossval="476800";


#### model name
#name_model= 'AliceNet_nodropout_' +  '_Filtr_' + str( my_num_filter ) + '_Kernel_' + str( my_kernel_size );
name_model= lineage  +  '_AliceNet_nodropout_' +  '_Filtr_' + str( my_num_filter ) + '_Kernel_' + str( my_kernel_size );
###name_model= name_model + '_dense_' + str( nb_dense ) + '_' + str( nb_dense1 ) + '_cnn'
###name_model= name_model + '_dense_' + str( nb_dense ) + '_' + str( nb_dense1 ) + '_epochs_' + str( My_epochs ) + '_cnn'
name_model= name_model + '_dense_' + str( nb_dense ) + '_' + str( nb_dense1 ) + '_' + str( nb_dense2 ) + '_epochs_' + str( My_epochs ) + '_cnn'


print(output_location)
print(pseudoempirical_only)
print(name_model)
print(Ntraining)
print(Ncross)


#files names
file_sim_image = file_location + '/' + lineage + '_My_simulated_img_for_training.txt'
file_sim_param = file_location + '/' + lineage + '_My_simulated_params_for_training.txt'
    
file_cross_image = file_location + '/' + lineage + '_My_simulated_img_for_cross_validations.txt'
file_cross_param =file_location + '/' + lineage + '_My_simulated_params_for_cross_validations.txt'


##### TO SPEED UP ??
if pseudoempirical_only == 'PSEUDOEMPIRICAL_ONLY': #NO TRAINING LOAD MODEL
    #TO avoid bugs load fake small simulated datasets 
    #file_sim_image = file_location + '/My_simulated_img_for_training_SHORT.txt'
    #file_sim_param = file_location + '/My_simulated_params_for_training_SHORT.txt'
    file_sim_image = file_location + '/FAKE_My_simulated_img_for_training.txt'
    file_sim_param = file_location + '/FAKE_My_simulated_params_for_training.txt'
    Nsim="10000" ; Ntraining=9700 ; Nvalid=100 ; Ncross=200; Nsim_crossval=Nsim

else:
	print(file_sim_image) 
	print(file_sim_param) 


##### load simulated 1D images
x_tot   = np.loadtxt(file_sim_image)


print("OK")

if SLICE == 'ALL':
	x_train = x_tot[0:Ntraining];
	x_test  = x_tot[Ntraining:(Ntraining+Nvalid)];
	#x_cross = x_tot[(Ntraining+Nvalid):(Ntraining+Nvalid+Ncross)];
	
	x_cross = np.loadtxt(file_cross_image)

else:
	x_train = x_tot[0:Ntraining,first:last];
	x_test  = x_tot[Ntraining:(Ntraining+Nvalid),first:last];
	#x_cross = x_tot[(Ntraining+Nvalid):(Ntraining+Nvalid+Ncross)];
	
	x_cross_0 = np.loadtxt(file_cross_image)
	x_cross = x_cross_0[:,first:last]




##### load simulated parameters (labels = selection coefficietn and age of selection)
y_tot   = np.loadtxt(file_sim_param)
y_cross = np.loadtxt(file_cross_param)

##### Convert labels into nb_classes classes (0, 1, ... , nb_classes, with 1 means 1/nb_classes )
nb_classes = 100; mydecimals=0
#label 0 (selection coefficient)
min_selcoeff=0.01; max_selcoeff=0.1 ; diff_selcoeff=(max_selcoeff - min_selcoeff)
y_tot[:,0]   = np.around( ( (  y_tot[:,0]-min_selcoeff)/diff_selcoeff )*nb_classes , decimals=mydecimals)
y_cross[:,0] = np.around( ( (y_cross[:,0]-min_selcoeff)/diff_selcoeff )*nb_classes , decimals=mydecimals)

#label 1 (onset of selection) CHANGE HERE IF MIN AND MAX PARAM CHANGE
min_onset=5300; max_onset=5699; diff_onset=(max_onset - min_onset)
y_tot[:,1]   = np.around(  ((    y_tot[:,1]-min_onset)/diff_onset)*nb_classes , decimals=mydecimals  )
y_cross[:,1] = np.around(  ((  y_cross[:,1]-min_onset)/diff_onset)*nb_classes , decimals=mydecimals  )

#label 2 (frequency at the onset of selection, ancestral frequ)
min_ancfreq=0.00001; max_ancfreq=0.2 ; diff_ancfreq=(max_ancfreq - min_ancfreq)
y_tot[:,2]   = np.around( ( (  y_tot[:,2]-min_ancfreq)/diff_ancfreq )*nb_classes , decimals=mydecimals)
y_cross[:,2] = np.around( ( (y_cross[:,2]-min_ancfreq)/diff_ancfreq )*nb_classes , decimals=mydecimals)

y_tot[y_tot==nb_classes]=(nb_classes-1)


##### create label variables
##### WARNING pour faire un copy dans les 2D arrays il faut faire :: y_train = y_tot[0:Ntraining].ncopy
y_train = y_tot[0:Ntraining]
y_test = y_tot[Ntraining:(Ntraining+Nvalid)]
#y_cross = y_tot[(Ntraining+Nvalid):(Ntraining+Nvalid+Ncross)]

sel_train   = y_train[:,0]
onset_train = y_train[:,1]
anc_train = y_train[:,2]

sel_test   = y_test[:,0]
onset_test = y_test[:,1]
anc_test = y_test[:,2]

sel_cross   = y_cross[:,0]
onset_cross = y_cross[:,1]
anc_cross = y_cross[:,2]



print(Ntraining)
print( np.shape(x_train) )
print( np.shape(y_train) )


##### Reshaping the array to 4-dims so that it can work with the Keras API
#x_tot = x_tot.reshape(x_tot.shape[0], Ngenot, 1)
x_train = x_train.reshape(x_train.shape[0], Ngenot, 1)
x_test = x_test.reshape(x_test.shape[0], Ngenot, 1)
x_cross = x_cross.reshape(x_cross.shape[0], Ngenot, 1)
# Making sure that the values are float so that we can get decimal points after division
#x_tot = x_tot.astype('float32')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_cross = x_cross.astype('float32')
##### Normalizing the RGB codes by dividing it to the max RGB value. UNECESSARY since genotype are 0,1 in simulations
#x_tot /= 255
#x_train /= 255
#x_test /= 255
#x_cross /= 255

print( np.shape(x_train) )
print( np.shape(y_train) )

#%%
##### Setting and training th model
##### Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, Dropout, Flatten, MaxPooling2D, MaxPooling1D


if pseudoempirical_only == 'PSEUDOEMPIRICAL_ONLY': #NO TRAINING LOAD MODEL
    model = tf.keras.models.load_model(output_location + '/saved_model/' + name_model )
    model.summary()

else:
    ##### 1st featrues: Input frequency trajectory in a first image
    input_freqTraj = tf.keras.Input(shape=(Ngenot,1), name="img_trajectory")
    x=tf.keras.layers.Conv1D(my_num_filter, kernel_size=my_kernel_size, input_shape=input_shape, name="first_conv")(input_freqTraj)
    x=tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    ##### AlexNet
    #x=tf.keras.layers.Conv1D(my_num_filter, kernel_size=10, input_shape=input_shape, name="second_conv")(input_freqTraj)
    x=tf.keras.layers.Conv1D(my_num_filter, kernel_size=10, input_shape=input_shape, name="second_conv")(x)
    x=tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x=tf.keras.layers.Conv1D(my_num_filter, kernel_size=5, input_shape=input_shape, name="third_conv")(x)
    x=tf.keras.layers.Conv1D(my_num_filter, kernel_size=5, input_shape=input_shape, name="fourth_conv")(x)
    x=tf.keras.layers.Conv1D(my_num_filter, kernel_size=5, input_shape=input_shape, name="fifth_conv")(x)
    x=tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    features=tf.keras.layers.Flatten()(x)
    
     
    #learning dense network
    x=tf.keras.layers.Dense(nb_dense, activation=tf.nn.relu)(features)
    x=tf.keras.layers.Dense(nb_dense1, activation=tf.nn.relu)(x)
    x=tf.keras.layers.Dense(nb_dense2, activation=tf.nn.relu)(x)
     
    ##### ouputs
    sel_pred=tf.keras.layers.Dense(nb_classes,name="selcoeff",activation=tf.nn.softmax)(x)
    onset_pred=tf.keras.layers.Dense(nb_classes,name="onsetsel",activation=tf.nn.softmax)(x)
    anc_pred=tf.keras.layers.Dense(nb_classes,name="ancfreq",activation=tf.nn.softmax)(x)
    
    ##### #build the model with a single feature
    model = tf.keras.Model(inputs=input_freqTraj,outputs=[sel_pred, onset_pred, anc_pred]); #A single ferature
    
    ##### Compiling the model
    model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy','sparse_categorical_crossentropy','sparse_categorical_crossentropy'],loss_weights=[1.0, 1.0, 1.0], metrics=['accuracy'])
    
    model.summary()

    #Fitting the model
    fit_metrics=model.fit( x_train, [sel_train, onset_train, anc_train] , epochs=My_epochs); #A 5 et 10 epoch on un s=0.3, a 20 epoch ca over fit (loss function) !
    


##### Prediction on cross validation simlations
prediction_cross=np.zeros((Ncross,18))
for i in range(Ncross):
    temp_pred=model.predict(x_cross[i].reshape(1, Ngenot, 1))
    ####temp_pred=model.predict(x_cross[i].reshape(1, Ngenot, 1), batch_size=1)
    
    temp_sel=temp_pred[0]
    true=y_cross[i,0]; predicted=temp_sel.argmax() ; norm_likelihood=temp_sel[0,predicted]
    #compute the weigthed average (bin*normlikelihood)
    arr = np.arange(0, nb_classes, dtype=float); arr = arr.reshape(1,nb_classes); #array dim(1,10) of the bin values
    weighted_mean=np.sum(temp_sel*arr)
    #fill the array
    prediction_cross[i,0]=true;
    prediction_cross[i,1]=predicted;
    prediction_cross[i,2]=norm_likelihood; #prediction using armax (with corresponding likelihood value)
    
    temp_onset=temp_pred[1]
    true=y_cross[i,1]; predicted=temp_onset.argmax() ; norm_likelihood=temp_onset[0,predicted]
    #compute the weigthed average (bin*normlikelihood)
    arr = np.arange(0, nb_classes, dtype=float); arr = arr.reshape(1,nb_classes); #array dim(1,10) of the bin values
    weighted_mean=np.sum(temp_onset*arr)
    #fill the array
    prediction_cross[i,3]=true;
    prediction_cross[i,4]=predicted;
    prediction_cross[i,5]=norm_likelihood; #prediction using armax (with corresponding likelihood value)
    
    temp_anc=temp_pred[2]
    true=y_cross[i,2]; predicted=temp_anc.argmax() ; norm_likelihood=temp_anc[0,predicted]
    #compute the weigthed average (bin*normlikelihood)
    arr = np.arange(0, nb_classes, dtype=float); arr = arr.reshape(1,nb_classes); #array dim(1,10) of the bin values
    weighted_mean=np.sum(temp_anc*arr)
    #fill the array
    prediction_cross[i,6]=true;
    prediction_cross[i,7]=predicted;
    prediction_cross[i,8]=norm_likelihood; #prediction using armax (with corresponding likelihood value)
    
  
    
    #to derive the posterior distibution  (source Imagene:: function  plot_scores, file on GIthub ImaGene/ImaGene.py)
    classes=np.arange(nb_classes)
    samples_distr = np.random.choice(classes, size = 100000, replace = True, p = temp_sel.reshape(100))
    if POINTESTIM == 'AVERAGE':
        prediction_cross[i,9]=np.average(samples_distr);         #posterior mean sel_coeff
    else:
        prediction_cross[i,9]=stats.mode(samples_distr)[0];         #posterior mean sel_coeff
    
    prediction_cross[i,10]=np.quantile(samples_distr,0.025);  #posterior ICmin sel_coeff
    prediction_cross[i,11]=np.quantile(samples_distr,0.975);  #posterior ICmax sel_coeff
    
    classes=np.arange(nb_classes)
    samples_distr = np.random.choice(classes, size = 100000, replace = True, p = temp_onset.reshape(100))
    if POINTESTIM == 'AVERAGE':
        prediction_cross[i,12]=np.average(samples_distr);         #posterior mean onset
    else:
        prediction_cross[i,12]=stats.mode(samples_distr)[0];         #posterior mean sel_coeff
    
    prediction_cross[i,13]=np.quantile(samples_distr,0.025); #posterior ICmin onset
    prediction_cross[i,14]=np.quantile(samples_distr,0.975); #posterior ICmax onset
    
    
    classes=np.arange(nb_classes)
    samples_distr = np.random.choice(classes, size = 100000, replace = True, p = temp_anc.reshape(100))
    if POINTESTIM == 'AVERAGE':
        prediction_cross[i,15]=np.average(samples_distr);         #posterior mean onset
    else:
        prediction_cross[i,15]=stats.mode(samples_distr)[0];         #posterior mean sel_coeff
    
    prediction_cross[i,16]=np.quantile(samples_distr,0.025); #posterior ICmin onset
    prediction_cross[i,17]=np.quantile(samples_distr,0.975); #posterior ICmax onset
    

    
    
#file_name=output_location + '/crossval/pred_crossvalidation_two_labels.txt'
file_name=output_location + '/crossval/pred_crossvalidation_' + name_model + '.txt'
temp_file = open(file_name, "w")
print(file_name + "saved path")
np.savetxt(file_name, prediction_cross, fmt='%.4f',delimiter='\t') #integer (note: fmt='%.2f' for floating, fmt='%.2e' for floating in decimal power)
temp_file.close()


#### estimate parameter on LCT data
real_snp = 'DO';
#real_snp = 'DONTDO';  #### estimate parameter on true data
if real_snp == 'DO':
    print("analyze LCT for validation")
    snp_file = output_location + '/My_empirical_image_for_a_variant.txt' 
    
    if SLICE == 'ALL':
        snp = np.loadtxt(snp_file);#WARNING current genotypes are missing
        snp = snp.astype('float32')
	    
    else:
        snp_0 = np.loadtxt(snp_file);#WARNING current genotypes are missing
        x_cross = x_cross_0[:,first:last]
        snp = snp_0[first:last]
        snp = snp.astype('float32')
    
    nreplicat=1
    prediction_snp=np.zeros((nreplicat,18))
    
    #first replicat fot the current model fit
    temp_pred = model.predict(snp.reshape(1, Ngenot, 1))
    temp_sel=temp_pred[0]
    true=-1; predicted=temp_sel.argmax() ; norm_likelihood=temp_sel[0,predicted]
    print(snp_name, " prediction (selection coefficient):", predicted, " norm_likelihood=", norm_likelihood )
    #fill the array
    prediction_snp[0,0]=true;
    prediction_snp[0,1]=predicted;
    prediction_snp[0,2]=norm_likelihood; #prediction using armax (with corresponding likelihood value)
    
    temp_onset=temp_pred[1]
    true=-1; predicted=temp_onset.argmax() ; norm_likelihood=temp_onset[0,predicted]
    print(snp_name, " prediction (onset of selection):", predicted, " norm_likelihood=", norm_likelihood )
    #fill the array
    prediction_snp[0,3]=true;
    prediction_snp[0,4]=predicted;
    prediction_snp[0,5]=norm_likelihood; #prediction using armax (with corresponding likelihood value)
    
    temp_anc=temp_pred[2]
    true=-1; predicted=temp_anc.argmax() ; norm_likelihood=temp_anc[0,predicted]
    print(snp_name, " prediction (frequency at onset):", predicted, " norm_likelihood=", norm_likelihood )
    #fill the array
    prediction_snp[0,6]=true;
    prediction_snp[0,7]=predicted;
    prediction_snp[0,8]=norm_likelihood; #prediction using armax (with corresponding likelihood value)
    
    
    #to derive the posterior distibution  (source Imagene:: function  plot_scores, file on GIthub ImaGene/ImaGene.py)
    classes=np.arange(nb_classes)
    samples_distr = np.random.choice(classes, size = 100000, replace = True, p = temp_sel.reshape(100))
    file_name=output_location + '/crossval/empirical_validation_posterior_selcoeff_' + snp_name + '_' + name_model + '.txt'
    temp_file = open(file_name, "w")
    np.savetxt(file_name, samples_distr, fmt='%.2f',delimiter='\t') #integer (note: fmt='%.2f' for floating, fmt='%.2e' for floating in decimal power)
    temp_file.close()
    
    if POINTESTIM == 'AVERAGE':
        prediction_snp[0,9]=np.average(samples_distr);         #posterior mean sel_coeff (twice)
    else:
        prediction_snp[0,9]=stats.mode(samples_distr)[0];         #posterior mean sel_coeff (twice)
    
    prediction_snp[0,10]=np.quantile(samples_distr,0.025);  #posterior ICmin sel_coeff
    prediction_snp[0,11]=np.quantile(samples_distr,0.975);  #posterior ICmax sel_coeff
    
    classes=np.arange(nb_classes)
    samples_distr = np.random.choice(classes, size = 100000, replace = True, p = temp_onset.reshape(100))
    file_name=output_location + '/crossval/empirical_validation_posterior_onsetsel_' + snp_name + '_' + name_model + '.txt'
    temp_file = open(file_name, "w")
    np.savetxt(file_name, samples_distr, fmt='%.2f',delimiter='\t') #integer (note: fmt='%.2f' for floating, fmt='%.2e' for floating in decimal power)
    temp_file.close()
    
    if POINTESTIM == 'AVERAGE':
        prediction_snp[0,12]=np.average(samples_distr);         #posterior mean onset  (twice)
    else:
         prediction_snp[0,12]=stats.mode(samples_distr)[0];         #posterior mean onset  (twice)
    
    prediction_snp[0,13]=np.quantile(samples_distr,0.025); #posterior ICmin onset
    prediction_snp[0,14]=np.quantile(samples_distr,0.975); #posterior ICmax onset
    
    classes=np.arange(nb_classes)
    samples_distr = np.random.choice(classes, size = 100000, replace = True, p = temp_anc.reshape(100))
    file_name=output_location + '/crossval/empirical_validation_posterior_ancfreq_' + snp_name + '_' + name_model + '.txt'
    temp_file = open(file_name, "w")
    np.savetxt(file_name, samples_distr, fmt='%.2f',delimiter='\t') #integer (note: fmt='%.2f' for floating, fmt='%.2e' for floating in decimal power)
    temp_file.close()
    
    if POINTESTIM == 'AVERAGE':
        prediction_snp[0,15]=np.average(samples_distr);         #posterior mean onset  (twice)
    else:
         prediction_snp[0,15]=stats.mode(samples_distr)[0];         #posterior mean onset  (twice)
    
    prediction_snp[0,16]=np.quantile(samples_distr,0.025); #posterior ICmin onset
    prediction_snp[0,17]=np.quantile(samples_distr,0.975); #posterior ICmax onset
    
    
    
    file_name=output_location + '/crossval/empirical_validation_pred_' + snp_name + '_' + name_model + '.txt'
    temp_file = open(file_name, "w")
    np.savetxt(file_name, prediction_snp, fmt='%.4f',delimiter='\t') #integer (note: fmt='%.2f' for floating, fmt='%.2e' for floating in decimal power)
    temp_file.close()
    
    
    
else:
    print("skip analyze real data")


if pseudoempirical_only == 'PSEUDOEMPIRICAL_ONLY': #NO TRAINING LOAD MODEL
    print("PSEUDOEMPIRICAL_ONLY\n")

else:
    #### sauvegarde du model et rechargement
    if os.path.exists(output_location + '/saved_model/' + name_model):
        print("saving model\n")
        
    else:
        print("creating folder and saving model\n")
        os.mkdir(output_location + '/saved_model/' + name_model )
    
    model.save(output_location + '/saved_model/' + name_model )





