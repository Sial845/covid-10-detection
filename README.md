# covid-19-detection
COVID-19 (coronavirus disease 2019) is an infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), a strain of coronavirus. The first cases were seen in Wuhan, China, in late December 2019 before spreading globally. The current outbreak was officially recognized as a pandemic by the World Health Organization (WHO) on 11 March 2020. Currently Reverse transcription polymerase chain reaction (RT-PCR) is used for diagnosis of the COVID-19. X-ray machines are widely available and provide images for diagnosis quickly so chest X-ray images can be very useful in early diagnosis of COVID-19.
# Dataset
Positive Cases : https://github.com/ieee8023/covid-chestxray-dataset
Normal Cases : https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

# Data scaling, normalization and augmentation
Based on data inspection, images are scaled to a size of 244 by 244, normalized to values (0,1) and augmented by simple zoom and rotation to enhance the generalization.

# Modeling:
Keras' Convolutional Neural Network model was implemented to classify whether a patient has pneumonia. A CNN is used to capture the spatial distributions in an image by applying the aforementioned filters.

Conv2D Layer is a set of learnable filters. Using the kernel filter, each filter transforms a part of the image to transform parts of the image. Conv2D also has two padding options:

Valid Padding - reduces convolved feature dimensionality
Same Padding - either increases or leaves the dimensionality alone.
Essentially, the first Conv2D layer captures low-level features such as the images' edges, colors, and gradient orientation.

Added Conv2D layers allow the model to learn high-level features such as identifying the ribs and lungs in the images.

Max Pooling Layers reduce the spatial size of the convolved features and returns the max value from the portion of the image covered by the kernel for three reasons:

Decrease the computation power to process the data by dimensionality reduction
Extract dominant features by training the model
Reduce noise
Flatten Layer converts all of the learned features from the previous convolutional layers to a format that can be used by the densely connected neural layer

Dense Layers are used to generate the final prediction. It takes in the number of output nodes and has an activation function which we will use the sigmoid activation function. The values for the sigmoid range between 0 and 1 that allows the model to perform a binary classification.

The first Conv2D is the input layer which takes in the images that have been converted to 64x64x3 floating point tensors.

ReLu (recified linear unit) is used as the activation function max(0,x) for the convolutional 2D layers. Essentially, until a threshold is hit, it doesn't activate!

# Model Evaluation:
After running the CNN with 20 epochs with batch sizes of 32, it appears both training and validation accuracy scores converge to higher accuracy percentages, meaning the model is not overfit. Furthermore, the loss score for both training and validation decrease overall as the number of epochs increase. The model is able to predict whether a patient has covid-19 with 99.1% accuracy, which is not bad, but can definitely be better.

