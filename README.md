# live_ASL_classifier
This is a quick project that I did so that I could learn how to use mediapipe and fiddle around with how to train a neural network.
Some of the code is taken from a bunch of online tutorials and documentation.
One of the files processes JPG images, detects where hand keypoints are in each image, and saves each coordinate in a list.
The other file trains a neural network with the coordinate data along with the letter it represents. Then, you can use the model to recognise what letter you're holding up to the webcam, and it will show you the letter on the screen.
