# SVM for pulse shape discrimination in PyTorch (Work In Progress)
Support Vector Machine (SVM) for neutron-gamma pulse shape discrimination

## Brief Background and Introduction
Pulse Shape Discrimination is a problem where one tries to distinguish pulses based on a measured signal. In this case, we are trying to distinguish between neutron and gamma rays from a signal that contains both of them. If this were to be implemented into a machine learning problem, the optimal approach would be to use pure neutron rays and pure gamma rays as a training set. However, as far as my knowledge goes, it is not possible to obtain these pure signals. Therefore, we rely on a theoretical assumption that can be used to identify the neutron and gamma rays. This assumption has a threshold in which the higher the threshold, the more distinguishable the two rays are. Therefore we will use this assumption with a high threshold as our training set, and the same assumption with a lower threshold as our validation set.

Now we have a classification problem involving just 2 classes, neutron and gamma rays. An approach would be to use a support vector machine that can separate these 2 classes using a hyperplane. The hyperplane parameters are learned throughout the training process.

## Installation
Download the necessary neutron and gamma pulse data in https://www.dropbox.com/sh/sklqbrd7gvq6azz/AABCExrGTyESctHbs1eQO4m6a?dl=0.
Install pytorch and execute "python main.py" at the command line.
