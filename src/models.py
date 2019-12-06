import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Please read the free response questions before starting to code.

class Digit_Classifier(nn.Module):
    """
    This is the class that creates a neural network for classifying handwritten digits
    from the MNIST dataset.
	
	Network architecture:
	- Input layer
	- First hidden layer: fully connected layer of size 128 nodes
	- Second hidden layer: fully connected layer of size 64 nodes
	- Output layer: a linear layer with one node per class (in this case 10)

	Activation function: ReLU for both hidden layers

    """
    def __init__(self):
        super(Digit_Classifier, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)


    def forward(self, input):
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        output = F.relu(self.fc3(input))
        return output


class Dog_Classifier_FC(nn.Module):
    """
    This is the class that creates a fully connected neural network for classifying dog breeds
    from the DogSet dataset.
    
    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """

    def __init__(self):
        super(Dog_Classifier_FC, self).__init__()
        self.fc1 = nn.Linear(12288, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, input):
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        output = F.relu(self.fc3(input))
        return output


class Dog_Classifier_Conv(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying dog breeds
    from the DogSet dataset.
    
    Network architecture:
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    Inputs: 
    kernel_size: list of length 2 containing kernel sizes for the two convolutional layers
                 e.g., kernel_size = [(3,3), (3,3)]
    stride: list of length 2 containing strides for the two convolutional layers
            e.g., stride = [(1,1), (1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Dog_Classifier_Conv, self).__init__()

        print(kernel_size, stride)
        k1 = kernel_size[0]
        k2 = kernel_size[1]

        s1 = stride[0]
        s2 = stride[1]


        self.conv1 = nn.Conv2d(3, 16,  k1, (2,2))
        self.conv2 = nn.Conv2d(16, 32,  k2, (2,2))
        self.fc1 = nn.Linear(32 * 13* 13, 10)

    def forward(self, input):
        input = input.permute(0, 3, 1, 2)
        input = F.relu(self.conv1(input))
        input = F.relu(self.conv2(input))
        input = input.view(-1, 32*13*13)
        output = self.fc1(input)
        return output

class Synth_Classifier(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying 
    synthesized images.
    
    Network architecture:
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 2)

    Activation function: ReLU for both hidden layers

    Inputs: 
    kernel_size: list of length 3 containing kernel sizes for the three convolutional layers
                 e.g., kernel_size = [(5,5), (3,3),(3,3)]
    stride: list of length 3 containing strides for the three convolutional layers
            e.g., stride = [(1,1), (1,1),(1,1)]

    """

    def __init__(self, kernel_size, stride):
        super(Synth_Classifier, self).__init__()

        #print(kernel_size, stride)
        self.conv1 = nn.Conv2d(1, 2, kernel_size[0], stride[0])
        self.conv2 = nn.Conv2d(2, 4, kernel_size[1], stride[1])
        self.conv3 = nn.Conv2d(4, 8, kernel_size[2], stride[2])
        self.fc1 = nn.Linear(8 , 2)


    def forward(self, input):
        input = input.permute(0, 3, 1, 2)
        output = F.max_pool2d(F.relu(self.conv1(input)), kernel_size = 2)
        output = F.max_pool2d(F.relu(self.conv2(output)), kernel_size = 2)
        output = F.max_pool2d(F.relu(self.conv3(output)), kernel_size = 2)
        output = output.view(-1, 8)
        output = F.relu(self.fc1(output))
        print(output.size())
        return output















