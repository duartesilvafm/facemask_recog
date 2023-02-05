# facemask_recog

In this repo, a comparison is made between simple CNN stacked architectures containing MaxPooling layers and pre-trained version of the InceptionV3 Model (https://keras.io/api/applications/inceptionv3/)

The models metrics are then evaluated to see what would be the difference between simple architectures and complex architectures for a multi-class classification problem, where the aim is to classify correctly black and white 64x64 images of faces:

*wearing a facemask 
*not wearing a facemask
*wearing a facemask partially covered
*not a face

The dataset was obtained from kaggle from the following link: https://www.kaggle.com/datasets/jamesnogra/face-mask-usage 

InceptionV3 is trained with RGB images, so to fit black and white images, the images had to be converted to 3 channels and an input size of 75x75, therefore the image shape will be 75x75x3 when fitting to the InceptionV3 model (that is why a new ImageDataGenerator class is created for training the Inception Model). 

For the other models, the images where kept in their standard format - (64x64x1).
