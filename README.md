# feathers_in_focus_AML
Classifying images of bird species. Feathers in Focus. Applied Machine Learning.

The dataset for this project is provided through the Kaggle competition:

https://www.kaggle.com/competitions/aml-2025-feathers-in-focus/data

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
