# feathers_in_focus_AML
Classifying images of bird species. Feathers in Focus. Applied Machine Learning.

The dataset for this project is provided through the Kaggle competition:

https://www.kaggle.com/competitions/aml-2025-feathers-in-focus/data

##### Step 1: Determining the upper baseline
First, a pretrained model was selected via HuggingFace to determine the upper baseline.
Model: Emiel/cub-200-bird-classifier-swin · Hugging Face

Accuracy on Kaggle: 0.86850

Model characteristics:
•	The base architecture of the model is microsoft/swin-large-patch4-window12-384-in22k, and it was finetuned to make it suitable as a bird classifier.
•	Pre-trained on ImageNet-21k.
•	Data augmentation: images were not replaced but were duplicated and augmented. Augmentations applied were HorizontalFlips and Rotations (10 degrees) to align with the relatively homogeneous dataset.
•	Finetuning was done on some 50 different models including different VTs and CNNs.
•	All models were trained for 10 epochs with the best model, based on the evaluation accuracy, saved every epoch.
•	The finetuning data is a subset of the CUB-200-2011 dataset. The model was finetuned on 3,533 samples of the labeled dataset we were given, stratified on the label (7,066 including augmented images).
•	Note: Building the model, pretraining, and finetuning had already been done. We did not do this ourselves. We purely used this model as a baseline to determine what the highest possible accuracy is to work toward with our own model.

##### Step 2: Determining the lower baseline
Next, we started building our own model from scratch. This model was used to determine the lower baseline and to see how we could build a working model in the first place. Model: feathers_in_focus_AML/model-feathers-in-focus-code-elske.ipynb at elske-code · elhartw/feathers_in_focus_AML

Accuracy on Kaggle: 0.02000

Model characteristics:
•	The model was built based on the MNIST example provided in class. We used the basic structure and adapted it slightly to make it work for our classification problem.
•	For the model, only the 4,000 images were used as input, not the file with attributes. From these 4,000 examples, 200 bird species had to be classified.
•	The model was a Convolutional Neural Network (CNN) with 4 convolutional layers. With filters: self.conv1 = nn.Conv2d(3, 32, 3, 1), self.conv2 = nn.Conv2d(32, 64, 3, 1), self.conv3 = nn.Conv2d(64, 64, 3, 1), self.conv4 = nn.Conv2d(64, 64, 3, 1). (The first 3 is due to RGB.)
•	After each hidden layer, a ReLU activation function follows for non-linearity, max pooling, and finally a fully connected layer and an output function with a softmax function.
•	Hyperparameters: Epochs = 14, learning rate: 1.0.

Notable observations:
•	The model works and is able to make predictions for the 200 bird species. The model architecture is highly explainable, and training is fast (even possible without a GPU). However, the accuracy is very low, only 2% of the predictions are correct, so this needs to be improved.

##### Step 3: Building our own model
Problems with our first model: Classifying 200 bird species from 4,000 images is an extremely complex task. Instead of, for example, a binary classification problem, we are dealing with 200 classes here. And since there are only 4,000 example images to train on, this amounts to an average of 20 images per bird species, which means there is very little data. In short: the problem is too complex for the model and there is too little data available. This means we are dealing with underfitting.

Possible solutions for underfitting are: choosing a more complex model, adding extra features, and training the model longer. We tried to implement these solutions as best as possible, and in doing so, we attempted to optimize various hyperparameters, such as the learning rate, batch size, epochs, and weight decay (L2 regularization). Model: feathers_in_focus_AML/CNN-incl-models.resnet34(weights=None).ipynb at main · elhartw/feathers_in_focus_AML

Accuracy on Kaggle: 0.32475

Model characteristics:
•	Building the model with only the 4,000 images as input is insufficient. Therefore, a file with various attributes was added, containing descriptions of bird characteristics that help the model with its predictions. A combined loss function is used for both the images and the attributes.
•	Additionally, data augmentation was added in various ways, namely RandomResizedCrop, RandomHorizontalFlip, RandomRotation, and ColorJitter (contrast).
•	The model was made much more complex than the simple 4 layers of the previous approach by applying the ResNet34 architecture with 34 layers (21 million parameters).
•	The model was trained much longer (epochs=70), giving it ample time to learn and reach convergence.
•	A learning rate scheduler is used, which means the learning rate gradually decreases following a cosine curve.
•	Other parameters that were adjusted: the optimizer was changed from Adadelta to AdamW, ImageNet normalization was added (standardizing pixel values for more stable training), best model saving was implemented (only the best model is saved after each epoch, instead of the last model), and weight decay was added (L2 regularization, penalizing large weights).

##### Conclusion: 
Classifying 200 bird species from only 4,000 images is a highly complex task. Our initial from-scratch model achieved just 2% accuracy due to underfitting. By implementing various optimizations (adding attribute-based learning, data augmentation, switching to ResNet34 architecture, training for 70 epochs with cosine learning rate scheduling, and using AdamW optimizer with weight decay) we improved accuracy to 35%. This still falls short of the 87% achieved by the pretrained baseline, highlighting the value of transfer learning from large-scale datasets like ImageNet-21k. However, building from scratch offers advantages: full transparency in model architecture, complete control over training data, and no dependency on external pretrained weights. These are relevant considerations for privacy and fairness in sensitive applications.



### Explanation of CNN development and optimization

##### CNN 1 (baseline model based on MNIST example) 

Started with a basic CNN model, based on the MNIST example. Model contains 4 convolutional layers: 3 channels (RGB), 32 channels, 64, 64.

##### CNN 2 (including attributes) 

Transitioned from a standard CNN to a multi-task CNN, with 2 output heads: 1 for the 200 classes and 1 for the attributes (provided file). Epochs reduced to 3 due to training duration, but this resulted in all predictions receiving the same label.

##### CNN 3 (5 layers and optimization) 

Further developed and optimized the multi-task CNN. Model architecture changed from Net to DeeperNet. Increased from 4 convolutional layers to 5 convolutional layers. More filters per layer (64 to 512). Added BatchNorm to normalize output per layer. Added padding to prevent loss of image edges. Changed pooling to MaxPool and Adaptive AvgPool. Changed gradient descent optimizer from Adadelta to Adam. Lambda learning rate from 1.0 to 0.001. Loss changed from class_loss to include attribute loss as well. Epochs increased to 6 to balance training duration while still achieving results.

##### CNN 4 (including data augmentation) 
Problem with CNN 3 output is that some classes are predicted very frequently while many classes are predicted rarely or not at all. Added data augmentation, synthetically expanding the image dataset with modified versions of existing images to provide more training data. Because the model trains very slowly, reduced image resize to 160, increased learning rate to 0.003, and increased batch size.

##### CNN 5 (extended training, 20 epochs) 
Expectation is that the model needs more epochs to train. Increased epochs to 20. Learning rate set to 0.002 and attribute loss weight to 0.3. 
