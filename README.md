# HaarCNN

HaarCNN is a character detection pipeline combining cascade classifiers 
and convolutional neural networks. 

This repository consists of four parts:

* annotation - scripts used for annotation of characters and ground truth
bounding boxes for the evaluation purposes
  
* cnn - software architectures of convolutional neural networks used in the
project
  
* data_generation - scripts used for generating data or performing data augmentation

* evaluation - scripts used for evaluation of HaarCNN, HaarCNN with Selective
Search step, pure Selective Search approach and sliding window technique with
  Non-Maximum Suppression on pyramid of scaled images
  
Please note: Image datasets have been omitted according 
to terms and conditions regarding re-distribution of Dilbert 
comcis (see https://dilbert.com/terms).