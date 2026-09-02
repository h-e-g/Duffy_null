# -*- coding: utf-8 -*-
"""
Guillaume laval: 11/01/2024
A Deep learning alforithm to jointly estimate 
    - SELECTION COEFFICIENT
    - AGE of SELECTION
from aDNA genotypes
"""


#Import libraries
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from array import array

#############   LOADING SCRIPT PARAMETERS       ###############################
#
jobfolder=sys.argv[1];
Ngenot=int( sys.argv[2] );
parID=int( sys.argv[3] );
###############################################################################

#Initialize some variables
file_location=jobfolder
output_location=jobfolder
file_sim_image=''
file_sim_param=''
file_cross_image=''
file_cross_param=''
snp_file=''
name_model=''



print("\n")
print("##################################################################################")
print ('Entering DISCRIMINATOR.py');

#############   LOADING MAIN SETTINGS       ###################################
#
settings_main   ='SETTINGS_GAN-discriminator_MAIN.txt'
myfile = open(settings_main, "r")
header = myfile.readline();
##### Reading the number of epochs for training
myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
My_usage=temp[1];

myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
Ntraining=int( temp[1] );
myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
Ncross=int( temp[1] );
myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
Nsnp=int( temp[1] );

myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
My_file_sim_image=temp[1];
myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
My_file_sim_param=temp[1];
myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
My_file_cross_image=temp[1];
myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
My_file_cross_param=temp[1];
myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
My_snp_file=temp[1];
myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
My_name_model=temp[1];
myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; temp = myline.split("\t")
My_verbose=temp[1];
myfile.close()

usage=My_usage;
verbose=My_verbose;
###############################################################################


#############   LOADING SETTINGS FOR CNN    ###################################
#
settings_cnn   ='SETTINGS_DISCRIMINATOR.txt'
myfile = open(settings_cnn, "r")
##### Reading the number of epochs for training
myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; x = myline.split("\t")
My_epochs=int( x[1] );
print("\nNumber of epochs for training :",My_epochs)

print("\nReading the CNN architecture : ")
##### Reading the CNN architecture
iline=0;
myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', '') ; x = myline.split("\t")
#print(myline); print(x)

layer =[ x[0] ] ; param1=[ x[1] ] ; param2=[ x[2] ] ; name  =[ x[3] ]
print("\tLayer ",iline," ", layer[iline] ,";", param1[iline] ,";", param2[iline] ,";", name[iline], "")


iline=1; num_line=1
while myline:
	myline = myfile.readline(); myline = myline.replace('\n', ''); myline = myline.replace('\r', ''); x = myline.split("\t")
	#print(myline);	print(x)
	if myline == "":
		break	
	
	layer.append(x[0]); param1.append(x[1]); param2.append(x[2]); name.append(x[3]);
	print("\tLayer ",iline," ", layer[iline] ,";", param1[iline] ,";", param2[iline] ,";", name[iline], "")
	
	
	iline=iline+1 ; num_line=num_line+1

myfile.close()
print("\n\tNumber of layers in CNNs : ",num_line)


### BY DEFAULT : file names and the nme of the CNN architectureused must be specified by user 
file_sim_image   =file_location + "/" + My_file_sim_image
file_sim_param   =file_location + "/" + My_file_sim_param

file_cross_image =file_location + "/" + My_file_cross_image
file_cross_param =file_location + "/" + My_file_cross_param

snp_file         =file_location + "/" + My_snp_file

#loading the CNN architecture
name_model= My_name_model


###############################################################################


  

if verbose == 'VERBOSE':
	print ();
	print ('Entering', sys.argv[0] ,'(', (len(sys.argv)-1) , 'arguments ) stored in', os.getcwd() );
	print ('\tTraining data stored in', jobfolder);
	print ('\tNumber of X variable', Ngenot);
	
	#print(f);
	#print("num_line",num_line);
	#print("num_col",num_col);
	
	print('\tUsage ', usage, '( ', Ntraining , ' data loaded for training )');
	print('\tNN name', name_model);
	
	
	
	print ('\tfile_sim_image', file_sim_image);
	print ('\tfile_sim_param', file_sim_param);
	print ('\tfile_cross_image', file_cross_image);
	print ('\tfile_cross_param', file_cross_param);	
	
	print ('\tparID', parID);	
	
	
###############################################################################


#############   LOADING DATA    ####################################################
#files names
file_sim_image   =file_sim_image    
file_sim_param   =file_sim_param 
    
file_cross_image =file_cross_image
file_cross_param =file_cross_param

##### load simulated 1D images 
x_tot   = np.loadtxt(file_sim_image)
x_train = x_tot[0:Ntraining];
x_cross = np.loadtxt(file_cross_image)

##### load simulated parameters (labels = selection coefficietn and age of selection)
y_tot   = np.loadtxt(file_sim_param)
y_cross = np.loadtxt(file_cross_param)

##### Convert labels into nb_classes classes (0 no data, 1 data)
nb_classes = 2; 
##### create training and cross validations variables
y_train = y_tot[0:Ntraining]

#par1_train  = y_train[:,parID]
#par1_cross  = y_cross[:,parID]
par1_train  = y_train[:]
par1_cross  = y_cross[:]

#par2_train  = y_train[:,parID]
#par2_cross  = y_cross[:,parID]
par2_train  = y_train[:]
par2_cross  = y_cross[:]

##### Reshaping arrays
x_tot = x_tot.reshape(x_tot.shape[0], Ngenot, 1)
x_train = x_train.reshape(x_train.shape[0], Ngenot, 1)
x_cross = x_cross.reshape(x_cross.shape[0], Ngenot, 1)
##### Making sure that the values are float
x_tot = x_tot.astype('float32')
x_train = x_train.astype('float32')
x_cross = x_cross.astype('float32')

print( np.shape(x_train) )
print( np.shape(y_train) )
print( np.shape(par1_train) )
###############################################################################



#############   SETTING CNN    ################################################
#
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, Dropout, Flatten, MaxPooling2D, MaxPooling1D


if usage == 'PREDICTION': #Loading a trained model
	model = tf.keras.models.load_model(output_location + '/saved_model/' + name_model )
	print('\tUsage: ', usage, ' = the previously trained model called ', name_model ,' has been loaded');
	print('\t-----------------------------------');
	print('\t');
	model.summary()
	
else:	
	print('\tUsage: ', usage, '( ', Ntraining , ' eQTLs used for training )');
	print('\t-----------------------------------');
	print('\t');
	
	
	##### INPUT SHAPE FOR CNN
	#input_shape = (Ngenot, 1); 
	##### INPUT SHAPE FOR MLP
	input_shape = (Ngenot, ); 
	
	##### 
	input_data = tf.keras.Input(shape=input_shape, name="img_trajectory")
	
	
	##### CNN architecture read from settings files, e.g., 'SETTINGS_CNN_architecture_simplified.txt'
	for i in range(num_line):
		#if f[i,0] == 1:
		if layer[i] == "Conv1D":
			numfilter=int(param1[i])
			kernelsize=int(param2[i])
			if i == 0:
				#tmpname='conv' + str(i)
				x=tf.keras.layers.Conv1D( numfilter , kernel_size=kernelsize, input_shape=input_shape, name=name[i])(input_data)
			else:
				#tmpname='conv' + str(i)
				x=tf.keras.layers.Conv1D( numfilter , kernel_size=kernelsize, input_shape=input_shape, name=name[i])(x)
			
		elif layer[i] == "Pool1D":
			poolsize=int(param1[i])
			x=tf.keras.layers.MaxPooling1D(pool_size=poolsize)(x)
			
		elif layer[i] == "Flatten":
			x=tf.keras.layers.Flatten()(x)
			
		elif layer[i] == "Dense":
			nbdense=int(param1[i])
			#x=tf.keras.layers.Dense(nbdense, activation=tf.nn.relu)(x)
			if i == 0:
				x=tf.keras.layers.Dense(nbdense, activation=tf.nn.relu)(input_data)
			else:
				x=tf.keras.layers.Dense(nbdense, activation=tf.nn.relu)(x)
			
		else:
			print("Unknown layer definition")
	
	##### A single ouput layer (twice the same)
	par1_pred=tf.keras.layers.Dense(nb_classes,name="par1",activation=tf.nn.softmax)(x)
	par2_pred=tf.keras.layers.Dense(nb_classes,name="par2",activation=tf.nn.softmax)(x)
	
	##### Several ouput layers (will be implementing later)
	#par1_pred=tf.keras.layers.Dense(nb_classes,name="par1",activation=tf.nn.softmax)(x)
	#par2_pred=tf.keras.layers.Dense(nb_classes,name="par2",activation=tf.nn.softmax)(x)
	#.....
	
	
	
	##### Building the model	
	#WITH a single output layer
	#model = tf.keras.Model(inputs=input_data,outputs=[par1_pred]);
	#model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy'],loss_weights=[1.0], metrics=['accuracy'])
	model = tf.keras.Model(inputs=input_data,outputs=[par1_pred, par2_pred]); 
	model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy','sparse_categorical_crossentropy'],loss_weights=[1.0, 1.0], metrics=['accuracy'])
	
	
	#WITH several output layers (will be implementing later)
	#model = tf.keras.Model(inputs=input_data,outputs=[par1_pred, par2_pred]); 
	#model.compile(optimizer='adam',loss=['sparse_categorical_crossentropy','sparse_categorical_crossentropy'],loss_weights=[1.0, 1.0], metrics=['accuracy'])
	
	
	model.summary()
	
	#print("\n")
	#print ('TEMP EXIT FOR SAFETY');
	#print ('UNCOMMENT AND RUN AGAIN');
	#print("\n")
	#exit()


	##### Fitting the model
	#WITH a single output layer
	#fit_metrics=model.fit( x_train, par1_train , epochs=My_epochs);
	fit_metrics=model.fit( x_train, [par1_train, par2_train] , epochs=My_epochs); 
	
	
	#WITH several output layers (will be implementing later)
	#fit_metrics=model.fit( x_train, [par1_train, par2_train] , epochs=My_epochs); #A 5 et 10 epoch on un s=0.3, a 20 epoch ca over fit (loss function) !
	


#############   PREDICTIONS     ##########################################################
#
##### Predictions on CROSS VALIDATION DATA
prediction_cross=np.zeros((Ncross,6))
print("Simulated data in '",Ncross,"' simulated 1D images")
for i in range(Ncross):
#for i in range(200):
	temp_pred=model.predict(x_cross[i].reshape(1, Ngenot, 1))
	####temp_pred=model.predict(x_cross[i].reshape(1, Ngenot, 1), batch_size=1)
	
	temp_par1=temp_pred[0]
	#true=y_cross[i,parID]; predicted=temp_par1.argmax() ; norm_likelihood=temp_par1[0,predicted]  
	true=y_cross[i]; predicted=temp_par1.argmax() ; norm_likelihood=temp_par1[0,predicted]       
	#compute the weigthed average (bin*normlikelihood)    
	arr = np.arange(0, nb_classes, dtype=float); arr = arr.reshape(1,nb_classes); #array dim(1,10) of the bin values
	weighted_mean=np.sum(temp_par1*arr)
	#fill the array
		
	prediction_cross[i,0]=true;
	prediction_cross[i,1]=predicted;
	prediction_cross[i,2]=norm_likelihood; #prediction using armax (with corresponding likelihood value)
	prediction_cross[i,3]=x_cross[i,0]; 
	prediction_cross[i,4]=x_cross[i,1]; 
	prediction_cross[i,5]=x_cross[i,2]; 




file_name=output_location + '/crossval/crossvalidation_prediction_' + name_model + '.txt'
temp_file = open(file_name, "w")
np.savetxt(file_name, prediction_cross, fmt='%.4f',delimiter='\t') #integer (note: fmt='%.2f' for floating, fmt='%.2e' for floating in decimal power)
temp_file.close()


##### Predictions on REAL DATA (INACTIVATED)
#real_snp = 'DO';
real_snp = 'DONTDO';  #### estimate parameter on true data  
if real_snp == 'DO':
	print("Real data in '",snp_file,"'")
	snp = np.loadtxt(snp_file);#WARNING current genotypes are missing
	snp = snp.astype('float32')
	prediction_snp=np.zeros((Nsnp,3))
	for i in range(Nsnp):
		
		#first replicat fot the current model fit
		temp_pred = model.predict(snp[i].reshape(1, Ngenot, 1))
		temp_par1=temp_pred[0]
		true=-100; predicted=temp_par1.argmax() ; norm_likelihood=temp_par1[0,predicted]        
		#print(snp_name, " prediction (par1):", predicted, " norm_likelihood=", norm_likelihood )
		#fill the array
		prediction_snp[0,0]=true;
		prediction_snp[0,1]=predicted;
		prediction_snp[0,2]=norm_likelihood; #prediction using armax (with corresponding likelihood value)
		
		

	file_name=output_location + '/crossval/empirical_prediction_' + name_model + '.txt'
	temp_file = open(file_name, "w")
	np.savetxt(file_name, prediction_snp, fmt='%.4f',delimiter='\t')
	temp_file.close()
	
else:
	print("skip analyze real data")
###############################################################################



#############   SAVING THE TRAINED MODEL ######################################
#
if usage == 'PREDICTION': #prediction based on a loaded trained model
	print("\n")

else:
	#### save the trained model
	if os.path.exists(output_location + '/saved_model/' + name_model):
		print("saving model in:\t")
		print(output_location + '/saved_model/' + name_model + "")
		
	else:
		print("creating folder and saving model in:\t")
		print(output_location + '/saved_model/' + name_model + "")
		os.mkdir(output_location + '/saved_model/' + name_model )
	
	model.save(output_location + '/saved_model/' + name_model )
###############################################################################

print ('Ending MLP-eQTL.py');
print("##################################################################################\n")

