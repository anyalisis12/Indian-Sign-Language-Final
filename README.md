SignSence- Indian Sign Language Recognition System

Our team has developed a real-time Indian Sign Language Recognition System using Python and Convolutional Neural Networks (CNN). The goal of this project is to recognize various signs made using hand gestures and convert them into text for easy communication, particularly aiding those with speech and hearing impairments. This system captures images, processes them, and uses a deep learning model to predict the sign being shown in real-time. The system was created using custom datasets collected through image capture, ensuring data quality and relevance to Indian Sign Language (ISL) signs.

                                                                                                                                                                                          
Gestures


All the gestures used in the dataset are in the shown below image with labels.


![Dataset of Alphabets](https://github.com/user-attachments/assets/87c48716-90d5-47f5-9925-c2b64cc6ffd3)

![Dataset of Numericals](https://github.com/user-attachments/assets/7c61ebfa-52bb-415a-a369-20d7a091a6f6)



Pre-requisites
Before running this project, make sure you have following dependencies -


[https://www.python.org/downloads/](https://www.python.org/downloads/)


[Pip](https://pypi.org/project/pip/)


[OpenCV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html)


[TensorFlow](https://www.tensorflow.org/install)

https://pypi.org/project/keras/

1]Collecting Images: The first step involved capturing images of hand signs representing different letters or gestures in Indian Sign Language. Our team set up a simple camera interface using OpenCV to capture thousands of images for each sign. The dataset included multiple samples to ensure model robustness.

2]Data Splitting: Once the images were collected, they were organized and split into training and validation sets to ensure the model could learn effectively and be evaluated accurately on unseen data. This split was necessary to avoid overfitting and assess model performance on new data.                                                                                                                                           
3]Data Preprocessing: Images were preprocessed to improve model training. This included resizing images to a uniform size, normalizing pixel values, and, in some cases, converting images to grayscale. Data augmentation techniques, such as rotation and flipping, were applied to increase data diversity and generalization.                                                                                               

4]Model Building: A CNN model was constructed to classify the images into different sign classes. The model consists of several convolutional layers to capture spatial features of the hand gestures, followed by pooling layers to reduce dimensionality and dense layers for classification.                                                                                                                                      

5]Model Training: The CNN model was trained on the training set while monitored through the validation set. The model's accuracy improved over epochs as it learned the distinguishing features of each sign. Hyperparameter tuning was performed to optimize the model's performance.                                                                                                                                              

6]Real-Time Prediction: After training, the model was integrated with a real-time camera feed to recognize hand gestures in real-time. The system captures each frame, preprocesses it, and then feeds it to the trained model for prediction. The predicted sign is displayed instantly, providing immediate feedback.

Group Members:

@github.com/anyalisis12


@github.com/preeti109


@github.com/Sayali2408

