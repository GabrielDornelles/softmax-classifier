# softmax-classifier
A numpy vectorized and non-vectorized full implementation of Softmax loss function and backpropagation on cifar-10.

Learned and implemented through Karpathy/Stanford cs231n course.

Notes:  https://cs231n.github.io/

Recorded classes: https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC
## Dataset being used for all this tests is cifar-10 
- The CIFAR-10 dataset consists of 60000 32x32 colour images (RGB) in 10 classes, with 6000 images per class. There are 50000 training images  and 10000 test images for the following:

![image](https://user-images.githubusercontent.com/56324869/133947770-f2a3bed9-8fab-4fcd-91f0-0f59beb26061.png)

# Single Layer Neural Net (softmax classifier for cifar-10)
- composed by a simple 3073x10 matrix that transforms an image (32x32x3 matrix) vector space into a 10x1 vector space where each element in the row vector is the probabilty for each class.
- 1x3073 * 3073x10 = 1x10 probabilities vector

## Learned weights
![image](https://user-images.githubusercontent.com/56324869/133947509-86e5d1f9-43fe-4bd3-b60a-81bfdc16a939.png)

Here each row from the 3073x10 matrix is decomposed by its last element (bias) and re-arranged into a 32x32x3 image (3072 elements). We can see that the learned weights for a simple net like, with only one neuron for each class, each row is trying to simple trying to generalize all the images into a single one, and then, if the pixels in the image inputed are similar on these positions, it activates and gives a higher score for the class.

The best accuracy for this model on cifar-10 after 1000 epoches using SGD and hyperparameter search was 39.7%.


# Two Layer Neural Net (100 neurons)

## 100 Neurons learned weights visualized
![image](https://user-images.githubusercontent.com/56324869/133947680-84341d47-2b9b-4801-b140-5e2f824bb608.png)

In this model, the 32x32x3 image matrix is transformed into a 3073x100 vector space (X@W1, image vector multiplied by first matrix), instead of 3073x10 (again, 3072 pixels + 1 bias element), and then, the 3073x100 matrix takes Relu function ( max(0,x) ) and is multiplied by a 100x10 matrix, resulting then in the 10 classes vector.

Here we can see a few shapes, some looks like different cars, some looks like differents horses, and in general, since its 10x more powerful compared to the 10 neurons net, it can find different features for each class and achieve a higher accuracy, which for this model is close to 50% (depends on hyperparameter search, something between 49 and 52%).

The process is:

- 1x3073(input) * 3073x100(w1) = 1x100(hidden layer)
- relu(1x100) * 100x10(w2) = 1x10 probabilties vector
