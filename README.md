# DUffy_null
Duffy-null evolution reveals a Late Palaeolithic rise of vivax malaria in West-Central Africa

A preliminary study can be found via biorxiv website.
"Deep learning analyses of DNA sequences resolve the retention of the Duffy-null resistance to Plasmodium vivax malaria in Africa" (Laval et al., BioRxiv 2025; https://www.biorxiv.org/content/10.64898/2025.12.25.695976v1.full)



##### Simulation files

1 - SLiM_Figure_3.txt and SLiM_Figure_S11.txt
Text files to set the SLiM simulations performed in this manuscript. 

-----------
initialize() {
	
	//####### POPULATION NOMENCLURE ######################################################
	//#generation_time	#Ngene	#age(ya)	#age(gene.ago)	#generation
	//25	5700	200000	800	4900
	//	175000	700	5000
	//	100000	400	5300
	//	75000	300	5400
	//	50000	200	5500
	//	25000	100	5600
	//####################################################################################
	
	//####### POPULATION NOMENCLURE ######################################################
	//lineages p1* are african lineages 
	//		p1  = West africa (AGR)										wAGR
	//		[ arbitrary get the name of the initial pop ] 
	//			p1   = West africa (AGR)									wAGR
	//			p15  = West africa (AGR) BANTU SPEAKERS						BAN
	//				p15  = WEST africa (AGR) BANTU SPEAKERS						wBAN
	//				p16  = EAST africa (AGR) BANTU SPEAKERS						eBAN
	//------------------------------------------------------------------------------------
	//		p10 = South africa (Bushmen)								BHG
	//------------------------------------------------------------------------------------
	//		p11 = unknown location (RHG)								RHG
	//			p11  = western RHG (arbitrary)								wRHG
	//			p111 = eastern RHG (arbitrary)								eRHG
	//------------------------------------------------------------------------------------
	//		p12 = East africa (East-agriculturalists)					eAGR
	//------------------------------------------------------------------------------------
	//lineages p2* are european lineages 								EUR
	//		p21 = CEU
	//		p22 = GBR ...
	//------------------------------------------------------------------------------------
	//lineages p3* are asian lineages  									ASI
	//		p31 = CHB
	//		p32 = JPT ...	
	//####################################################################################
	
-------------


2 - File for the generative adversarial network (GAN) 

Python script for the GAN
"CNN_Duffy_Null_s_T_p0.py"





3 - Python script for the convolutional neural network (CNN)

Python script for the CNN


Settings for convolution layers
	Ex. Conv1D	28	10	my1Dconvolution_2
	"Conv1D" calls for	keras.layers.Conv1D( 28 , kernel_size=10, input_shape=input_shape, name=name[i])

Settings for  Pooling layers
	Ex. Pool1D	2	NA	myMaxPool_2
	"Pool1D" calls for	keras.layers.MaxPooling1D(pool_size=poolsize)

Settings for  Flaten layers
	Ex. Flatten	NA	NA	myFlatten_1
	"Flatten" calls for	layers.Flatten()(x)

Settings for  Pooling layers
	Ex. Dense	128	NA	myDense_1
	"Dense" calls for	keras.layers.Dense(nbdense, activation=tf.nn.relu)(x)

