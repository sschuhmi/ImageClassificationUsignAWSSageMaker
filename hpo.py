#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
#import torchvision.models as models
#import torchvision.transforms as transforms
from torchvision import datasets, models, transforms

from torchvision.datasets import ImageFolder
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# SMDebug Libraries:
import smdebug.pytorch as smd
from smdebug import modes
from smdebug.pytorch import get_hook

import os
import logging
import sys

import argparse
import os
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    '''
    This function takes a model, a testing data loader and the defined criterion.
    It determines the test accuray/loss of the models
    '''

    logger.info("Testing started.")

    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad(): # disable gradient calculation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) # Get output
            test_loss += criterion(output, target).item()  # Sum up the batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the the max log-probability index
            correct += pred.eq(target.view_as(pred)).sum().item() # Update number of correct predictions

    test_loss /= len(test_loader.dataset)
    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)))

def train(model, train_loader, criterion, optimizer, epochs, device):
    '''
    This function takes a model, the train data loader as well as the selected criterion & optimizer.
    It returns the trained model.
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Training started.")
    model.train()
    #epochs = args.epoch
    for e in range(epochs):
        print(f"Starting epoch {e + 1}/{epochs}")
        running_loss=0
        correct=0
        seen=0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(data)             #No need to reshape data since CNNs take image inputs
            loss = criterion(pred, target)
            running_loss+=loss.item()
            loss.backward()
            optimizer.step()
            
            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            seen += target.size(0)

            if batch_idx % 10 == 0:
                #print(f"Epoch {e + 1}, batch {batch_idx}, loss={loss.item():.4f}")
                #print(f"Running loss {running_loss/seen:.6f}, Accuracy {100 * correct/seen:.2f}%")
                acc = 100 * correct/seen
                print(f"epoch={e+1} batch={batch_idx} loss={loss.item():.4f} accuracy={acc:.2f}%")
                
        #print(f"Epoch {e + 1}/{epochs} completed...")
        
    print(f"Training completed.")
    return model
    
def net(classes):
    '''
    This function initializes the model.
    It uses a pretrained ResNet50 model with IMAGENET1K_V1 weights.
    '''
    logger.info("Model creation for fine-tuning started.")
    model = models.resnet50(pretrained=True)

    # Disable gradient calculation to keep the weights of hidden layers unchanged.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # The number of input features of the model:
    num_features = model.fc.in_features

    model.fc = nn.Sequential(nn.Linear(num_features, classes))
    
    print("Completed model creation using ResNet50 with pretrained weights...")

    logger.info("Model creation completed.")

    return model

def create_data_loaders(bucket, batchsize):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''

    logger.info("Data loader creation started")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    train_uri = f"s3://{bucket}/dogImages/train"
    test_uri = f"s3://{bucket}/dogImages/test"
    val_uri = f"s3://{bucket}/dogImages/val"

    train_dir = os.environ["SM_CHANNEL_TRAIN"]
    test_dir = os.environ["SM_CHANNEL_TEST"]

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize["train"],
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batchsize["test"],
        shuffle=False
    )

    logger.info("Data loader creation completed")
    
    return train_loader, test_loader

def main(args):

    print("Starting main method of HPO and trying to initialize the model...")
    bucket = "uda-mleng-project3"

    # Determine device to be used:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The number of different dog classes in the used dataset:    
    dog_classes = 133

    '''
    Initialize model by calling the net function
    '''
    model = net(dog_classes)

    # Put to device...
    model.to(device)  
    #...and register to hook:
    #hook = get_hook()
    #hook.register_module(model)

    print(f"Model initialized and moved to device {device}")
    
    '''
    Creating loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss() # Use cross-entropy loss criterion
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr) # Use ADAM optimizer

    print("Defined loss criterion and the optimizer... Now creating the data loaders!")
    
    '''
    Call the train function to start training the model.
    Training data is taken from an S3 bucket.
    '''
    train_loader, test_loader = create_data_loaders(bucket, {"train":args.batch_size, "test":args.test_batch_size})

    print("Successfully created the data loaders for train & test data! Now training the model...")

    logger.info("Training the model")
    toc = time.perf_counter()
    
    model=train(model, train_loader, loss_criterion, optimizer, args.epochs, device)

    tic = time.perf_counter()
    logger.info(f"Training time: {tic - toc:0.2f}sec")

    print("Successfully trained the model! Now testing the model and evaluating against the loss criterion...")
    
    '''
    Testing the model to determine its accuracy
    '''
    test(model, test_loader, loss_criterion, device)
    
    '''
    Saving the trained model
    '''
    
    model_dir = os.environ["SM_MODEL_DIR"]
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    logger.info("Model weights saved.")
    print("Saved model.")
    
if __name__=='__main__':
    '''
    Specification of all the hyperparameters used to train our model.
    '''
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="Input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=128,
        metavar="N",
        help="Input batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--s3bucket",
        type=str,
        default=None,
        metavar="N",
        help="Name of the S3 Bucket with dataset data (default: None)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="The number of epochs to train (default: 5)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
    )
    
    
    args=parser.parse_args()
    
    main(args)