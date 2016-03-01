
************************Digit Recognizer*****************************

About
	This project is a Digit Recognizer tested and trained on MNIST database 
		http://yann.lecun.com/exdb/mnist/

	1. It uses Multi Layer Forward Neural Network with Back Propagation with one hidden layer having 100 hidden perceptrons.
	2. It compares stats with 1NN classifier
	3. One variation in Pre Processing i.e. Noise -- is added and then trained which showed better results.
	4. One variation with objective function i.e. Weight decay -- reduced the accuracy.
	5. Activation function used is sigmoid function

Language chosen "OCTAVE"


Instructions-

	1. Run as extractv1 in octave. As there are 60000 images for training it will take 6-7 hours for converging.
		So it's better to first train with less number of images like 600 editing variable N.
	2. for noise addition run extractv2Noise
	3. for weight decay run extractweightDecay
	4. for knn run knnWrapper