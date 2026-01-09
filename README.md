# CNN Image Classification using Flask

This project shows deploying a trained CNN model for image classification using a Flask API.
The model was trained in Google Colab and deployed locally.
The trained model is saved as an ".h5" file and used for inference.

# Model Explanation
In this project the input to the system is an image of a cat or a dog uploaded by the user through a Flask API. The image is resized to a fixed shape and normalized before being passed to a Convolutional Neural Network (CNN). The CNN processes the image through multiple convolution and max-pooling layers to extract hierarchical visual features such as edges, textures, and object shapes. These features are flattened and passed through fully connected layers with dropout to reduce overfitting. The output of the model is a probability score for each class Cat and Dog,generated using a sigmoid or softmax activation function, and the class with the highest probability is returned as the final prediction in JSON format along with the confidence score.



