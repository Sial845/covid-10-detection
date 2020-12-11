# covid-19-detection
COVID-19 (coronavirus disease 2019) is an infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), a strain of coronavirus. The first cases were seen in Wuhan, China, in late December 2019 before spreading globally. The current outbreak was officially recognized as a pandemic by the World Health Organization (WHO) on 11 March 2020. Currently Reverse transcription polymerase chain reaction (RT-PCR) is used for diagnosis of the COVID-19. X-ray machines are widely available and provide images for diagnosis quickly so chest X-ray images can be very useful in early diagnosis of COVID-19.
# Dataset
Positive Cases : https://github.com/ieee8023/covid-chestxray-dataset
Normal Cases : https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

# Network Architecture
Model: "sequential_1"

Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 64, 64, 32)        896         
max_pooling2d_1 (MaxPooling2 (None, 31, 31, 32)        0                
conv2d_2 (Conv2D)            (None, 29, 29, 32)        9248     
max_pooling2d_2 (MaxPooling2 (None, 14, 14, 32)        0                  
conv2d_3 (Conv2D)            (None, 12, 12, 32)        9248              
flatten_1 (Flatten)          (None, 4608)              0         
dense_1 (Dense)              (None, 128)               0    
dense_2 (Dense)              (None, 1)                 0       
=================================================================
Model Type: Sequential
Total parameters: 609,473
Trainable parameters: 609,473
Non-trainable parameters: 0
