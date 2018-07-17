# Import Statements
import matplotlib.pyplot as plt
import pandas as pd

import torch
import json

from PIL import Image
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Load Data Function
def LoadData(data_dir, batch_size) :
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # Define transforms for the training and validation/test sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    
    # Load the train, valid, and test datasets and apply tranformations to each
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Define a loader for each dataset
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    
    return train_data, trainloader, validloader


# Define the Model
def CreateModel(model_arch, hidden_units, drp_out=0.3) :
    
    model = models.__dict__[model_arch](pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        
    input_size = int(model.classifier.in_features)
    hidden_layer = hidden_units
    output_size = 102
    dropout = drp_out
    classifier_dropout = drp_out
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_layer)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(p=dropout)),
        ('fc2', nn.Linear(hidden_layer, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    
    return model, input_size, dropout


# Train the Network
def TrainNetwork(model, trainloader, train_data, validloader, criterion, optimizer, epochs, device):
    
    print_every = 20
    steps = 0

    model.to(device)

    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs = inputs.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        output = model.forward(inputs)
                        valid_loss += criterion(output, labels).item()

                        ps = torch.exp(output)
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()
                print("Epoch: {}/{}".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/len(train_data)), 
                      "Validation Loss: {:.3f}".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
    
# Save a Checkpoint
def SaveCheckpoint(model, model_arch, classifier_input_size, dropout, hidden_units, class_to_idx, save_dir):
    
    # Define Features for Checkpoint
    
    model.to('cpu')
    
    checkpoint = {
        'input_size': classifier_input_size,
        'hidden_layer': hidden_units,
        'output_size': 102,
        'dropout': dropout,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict(),
        'arch': model_arch
    }

    # Save the Checkpoint to Specified Directory
    torch.save(checkpoint, './' + save_dir + 'checkpoint.pth')


# Loading The Checkpoint
def LoadCheckpoint(checkpoint_path):
    
    #Get Checkpoint from Specified Path
    checkpoint = torch.load(checkpoint_path)
    
    # Create a Model from the Information Provided in the Checkpoint
    new_model = models.__dict__[checkpoint['arch']](pretrained=True)
    
    new_classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer'])),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(p=checkpoint['dropout'])),
                          ('fc2', nn.Linear(checkpoint['hidden_layer'], checkpoint['output_size'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    new_model.classifier = new_classifier

    new_model.load_state_dict(checkpoint['state_dict'])
    
    new_model.class_to_idx = checkpoint['class_to_idx']
    
    return new_model

# Process Image to be Used for Pytorch Model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    im = Image.open(image)
    
    im.resize((256, 256))
    im = im.crop((16, 16, 240, 240))
    
    np_image = np.array(im)
    np_image = np_image / 255
    
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    
    im = (np_image - means) / stds
    
    tranposed_im = im.transpose(2, 0, 1)
    
    return torch.from_numpy(tranposed_im)

# Predict the Class of an Image
def Predict(image_path, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    img = process_image(image_path).unsqueeze(0).float()
    
    model, img = model.to(device), img.to(device)
    
    model.eval()
    
    output = torch.exp(model.forward(img))
    output = output.topk(topk)
    
    probs = output[0].data.cpu().numpy()[0]
    classes = output[1].data.cpu().numpy()[0]
    
    idx_to_class = {key: value for value, key in model.class_to_idx.items()}
    
    classes = [idx_to_class[classes[i]] for i in range(classes.size)]
    
    return probs, classes

def ViewPredictionResults(probs, classes):
    dataframe = pd.DataFrame({
        'classes': pd.Series(data = classes),
        'probabilities': pd.Series(data = probs, dtype='float64')
    })
    
    print(dataframe)
