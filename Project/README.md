# The course project

THis project was done during the first months of the covid19 pandemic and aimed to provide a tool to help doctor identify diffrent types of tissues in the CT scans of
covid19 patients' lungs. 
Four classes of lung tissue were identified which represent the healthy tissue and three types of infected tissues which represents the following conditions: ground class opacification,consolidations and pleural effusions.
a dataset of 100 manually segmented images were used to train and test the proposed algorithms. 
Three algorithms were used.
1- using OTSU algorithm
2- using OTSU algorithm in additon to spatial colour segmentation.
3- by extracting features from the training images that include information about the image (histogram for example) in addition to imformation about each pixel (colour, class, position and others) 
and building a database with all these features. After that in the segmentation (runtime) process, the algorithms, for each pixel, finds the nearest pixel from the database that could match it and bases 
the decision class on this comparision. 

the dataset used are provided in addition to two programs, the first one performs the feature extraction and saves the database and the second one apply the segmenataion.
