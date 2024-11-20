# SignSence- Indian Sign Language Recognition System
## **Welcome to Our Project..!** 

Our team has developed a real-time Indian Sign Language Recognition System using Python and Convolutional Neural Networks (CNN). The goal of this project is to recognize various signs made using hand gestures and convert them into text for easy communication, particularly aiding those with speech and hearing impairments. This system captures images, processes them, and uses a deep learning model to predict the sign being shown in real-time. The system was created using custom datasets collected through image capture, ensuring data quality and relevance to Indian Sign Language (ISL) signs.

## Objective

The objective of this project aims to analyze and recognize various alphabets and numbers from a dataset of Indian Sign Language images. The system is designed to work with a diverse dataset that includes images captured under different lighting conditions and various hand orientations and shapes.
                                                                                                                                                                                          
## Gestures

-All the gestures used in the dataset are in the shown below image with labels.

![Dataset of Alphabets](https://github.com/user-attachments/assets/87c48716-90d5-47f5-9925-c2b64cc6ffd3)

![Dataset of Numericals](https://github.com/user-attachments/assets/7c61ebfa-52bb-415a-a369-20d7a091a6f6)


## Pre-requisites

Before running this project, make sure you have following dependencies -

[Python](https://www.python.org/downloads/)

[Pip](https://pypi.org/project/pip/)

[OpenCV](https://pypi.org/project/opencv-python/)

[TensorFlow](https://www.tensorflow.org/install)

[Keras](https://pypi.org/project/keras/)

[NumPy](https://pypi.org/project/numpy/)

## Steps of execution 

1]Collecting Images: The first step involved capturing images of hand signs representing different letters or gestures in Indian Sign Language. Our team set up a simple camera interface using OpenCV to capture thousands of images for each sign. The dataset included multiple samples to ensure model robustness.

2]Data Splitting: Once the images were collected, they were organized and split into training and validation sets to ensure the model could learn effectively and be evaluated accurately on unseen data. This split was necessary to avoid overfitting and assess model performance on new data.                                                                                                                                           
3]Data Preprocessing: Images were preprocessed to improve model training. This included resizing images to a uniform size, normalizing pixel values, and, in some cases, converting images to grayscale. Data augmentation techniques, such as rotation and flipping, were applied to increase data diversity and generalization.                                                                                               

4]Model Building: A CNN model was constructed to classify the images into different sign classes. The model consists of several convolutional layers to capture spatial features of the hand gestures, followed by pooling layers to reduce dimensionality and dense layers for classification.                                                                                                                                      

5]Model Training: The CNN model was trained on the training set while monitored through the validation set. The model's accuracy improved over epochs as it learned the distinguishing features of each sign. Hyperparameter tuning was performed to optimize the model's performance.                                                                                                                                              
6]Real-Time Prediction: After training, the model was integrated with a real-time camera feed to recognize hand gestures in real-time. The system captures each frame, preprocesses it, and then feeds it to the trained model for prediction. The predicted sign is displayed instantly, providing immediate feedback.


Using a Convolutional Neural Network (CNN) for real-time gesture recognition involves several steps, from capturing the data to deploying the trained model in a live system. Here’s how you can integrate CNN into your project:
________________________________________________________________________________________________________________________

### 1. Data Collection and Preprocessing
   
•	Data Collection:

    o	Use your existing script to collect gesture images. Ensure the dataset is diverse, covering various angles, lighting conditions, and hand positions for each gesture.

    o	Label each gesture image appropriately 

•	Resize and Normalize:

    o	Resize all images to a fixed size to match the input dimensions of the CNN.

    o	Perform augmentations like flipping, rotation, or brightness adjustments to increase dataset variability and improve model robustness.

________________________________________________________________________________________________________________________

### 2. Model Design and Training
   
•	Design the CNN Architecture:

    o	Build a CNN model with layers such as:

       Convolutional layers for feature extraction.

       Pooling layers for dimensionality reduction.

       Fully connected layers for classification.

    o	Use activation functions like ReLU in intermediate layers and softmax in the output layer for multi-class classification.

•	Train the Model:

    o	Use the labeled dataset to train the CNN.

    o	Split the data into training, validation, and testing sets.

    o	Optimize the model using loss functions like categorical_crossentropy and optimizers like Adam or SGD.

•	Save the Model:

    o	Once trained, save the model to a file (e.g., .keras).

________________________________________________________________________________________________________________________

### 3. Integration into Real-Time System

•	Real-Time Webcam Feed:

    o	Use OpenCV to access the webcam feed in real-time, as demonstrated in your project.

•	Preprocess Input for CNN:

    o	For each frame:

       Extract the Region of Interest (ROI).

       Resize and normalize the ROI to match the CNN's input shape.

•	Load the Trained Model:

    o	Use a library like TensorFlow/Keras or PyTorch to load the saved CNN model.

•	Predict Gesture:

    o	Pass the preprocessed ROI to the CNN model.

    o	The model outputs probabilities for each class (gesture), and the class with the highest probability is selected as the predicted gesture.

•	Display Results:

    o	Overlay the predicted gesture on the live webcam feed using OpenCV's text-drawing functions.


## Result

<img src="https://github.com/user-attachments/assets/96a2804e-d562-4346-9001-6aefdf04f51b" width=200 height=400 />

![Screenshot 2024-11-20 185714](https://github.com/user-attachments/assets/447e7c78-3a39-495b-8167-7f631ecf0a7d)

![Screenshot 2024-11-20 190059](https://github.com/user-attachments/assets/5ebf0074-03cb-4051-8530-27d27c08c034)

![Screenshot 2024-11-20 190203](https://github.com/user-attachments/assets/a8dbafb8-78d3-4ce5-9ed7-276563518381)



## Group Members:

[@Anuradha Bansode](https://github.com/anyalisis12)

[@Preeti Dubile](https://github.com/preeti109)

[@Sayali Tachale](https://github.com/Sayali2408)



