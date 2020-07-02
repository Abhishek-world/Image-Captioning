**Image-Captioning**<br>
"**Image Captioning Project Using RNN and CNN**"<br>
**Abstract**<br>
"Generating textual description for a given photograph is a challenging artificial intelligence problem. The methodology should be developed in a such a way that it requires computer vision to understand the content of the image and language model from the field of natural language processing. It should turn the understanding of the image into the words in the right order. This project involves computer vision and natural language processing concepts to recognize the context of an image and describe them in a natural language like English. We will use specialized deep neural networks like CNN, RNN for feature extraction and LSTM to generate the description for the features extracted from CNN.\n"
**Index Terms—Image caption, CNN, LSTM, model development, training dataset.

**Motivation**
The reason for choosing this project is it’s a part of ongoing researches going around. Here, we examine a proof-of-concept by taking a medium dataset and perform our analysis, create the model and test. A wide range of applications are possible using this methodology.

**Objectives**
Extracting the features of an input image. Generating a caption from the extracted features. To make our image descriptor model, we will be merging the architectures. It is also called a CNN- RNN model. CNN (Convolution Neural Network) is used for extracting features from the image. LSTM (Long Short-Term Memory) will use the information from CNN to help generate a description of the image. Developing the model using the extracted features and the text related to it.

**Introduction**
Humans can give insight descriptions of the images or the scenes presented to them. Computer vision aims at incorporating this ability of humans to provide distinctive and captious description of the images and different scenes. Thus, image descriptor is a task of generating a subjective description of all the objects, their relationship with the environment around them present in the images, to effectively describe the scene which are present.

Image descriptor performs a task that the computer vision and natural language processing concepts to recognize the context of an image and describe them in a natural language like English. This requires both image analysis and neural networks. We will use specialized deep neural networks like CNN, RNN for feature extraction and LSTM to generate the description for the features extracted from CNN. Humans can give insight descriptions of the images or the scenes presented to them. Computer vision aims at incorporating this ability of humans to provide distinctive and captious description of the images and different scenes. To make our.
image descriptor model, we will be merging the architectures. It is also called a CNN-RNN model.
The image features will be extracted from Xception which is a CNN model trained on the ImageNet dataset and then we feed the features into the LSTM model which will be responsible for generating the image captions.

**Proposed Methodology**
a. **Dataset**
For this project we are using Flickr 8k dataset. It contains 8000 images along with the captions. In these 8000 images, we use 1000 images for development process and 1000 images as testing/ evaluating the model and the rest of 6000 images for training the model.

**Convolution Neural Networks**
These are specialised deep neural networks that process the data that consists shape like a 2D matrix. As we know it is easy to represent any image in matrix form, CNN is very useful in working with the images. CNN is usually used for image classification and identification.
Long Short-Term Memory/n

This is a type of RNN (Recurrent Neural Network) which is used for sequence prediction. Based on the previous text, we can predict the next word. LSTM carries out relevant information throughout processing of inputs and discards the non-relevant information.
**Algorithm**
For this project, we don’t have to create the model from the scratch, we can apply transfer learning. Transfer learning is process where we use the pre trained model which have already been trained on large datasets. Here we use VGG16 model which was trained on ImageNet dataset. We can directly import this dataset from keras.applications
**Model Preparation**
To make our image descriptor model, we will be merging the architectures. It is also called a CNN-RNN model. CNN (Convolution Neural Network) is used for extracting features from the image. LSTM (Long Short-Term Memory) will use the information from CNN to help generate a description of the image.In CNN – RNN Model, we have to get the three functions: 1) Feature Extraction 2) Sequence Processor 3) Decoder
**Results and Discussions**
By running the following command with the testing image path, we get the following output
This project is an attempt to prove that we can train computers to give captious descriptions for the different scenes just as human do. Here, we create a model and train it with the dataset and we test the model with various new images to check how accurately the computer is extracting the features and mapping with the text from the descriptions we prepare in the model
 **Conclusion**
In this project, we have covered Image Captioning, a multimodal task which constitutes deciphering the image and describing it in natural sentences. We have then explained the methodology to solve the task and given a walk-through of its implementation. The model which we saw above was just the tip of the iceberg. There has been a lot of research done on this topic.

**Dependecies
Keras
numpy 
Tensorflow
Pickle
pillow
cv2
**
