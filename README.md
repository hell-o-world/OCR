OCR
===

Printed digits recognition using SVMs

A simple digits recognition program using Support Vector Machines.
Consists of 3 modules:
1)imgproc.py is used to pre-process the training data
2)imgprocTest.py is used to pre-process the testing data
--New test cases can be added to the grid(blank.jpg) and increasing the no. of testfiles paramter in imgprocTest.py

3)final.py is used for feature extraction and classification using SVMs
--The trained model is already saved in savedata/
--For training a new model with new features delete all the contents of savedata/ and run 'final.py' 
