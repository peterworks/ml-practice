#%%
import torch
import matplotlib as mpl
import importlib

#%%
from matplotlib import pyplot as plt
from classes.Network import Network as Net
from torch.utils.data import DataLoader as DataLoader
from torchvision import datasets, transforms
from torch import nn, optim

#%%
guess_mode = True
continue_training = False
training = not guess_mode
training_validation = False

#%%
network = Net(1, 2, 2)
network = network.cuda()

#%%
transform_in = transforms.Compose([
    transforms.Resize((72, 128)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

transform_out = transforms.Compose([
    transforms.Resize((72, 128)),
    transforms.ToTensor()
])

#%%
if(not guess_mode):
    dataset_in = datasets.ImageFolder('./rooms', transform = transform_in )
    data_loader_in = DataLoader(dataset_in, batch_size = 1, shuffle = False)

    dataset_out = datasets.ImageFolder('./rooms', transform = transform_out)
    data_loader_out = DataLoader(dataset_out, batch_size = 1, shuffle = False)

    # criterion = nn.HingeEmbeddingLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=1e-6, weight_decay=0.023)

#%%
dataset_guess = datasets.ImageFolder('./guesses', transform = transform_in)
data_loader_guess = DataLoader(dataset_guess, batch_size = 1, shuffle = False)

dataset_validation = datasets.ImageFolder('./guesses', transform = transform_out)
data_loader_validation = DataLoader(dataset_validation, batch_size = 1, shuffle = False)

#%%
images_guess = []
images_validation = []

#%%
if(not guess_mode):
    images_in = []
    images_out = []

    for img in data_loader_in:
        image, label = img
        images_in.append(image.cuda())

    for img in data_loader_out:
        image, label = img
        images_out.append(image.cuda())

#%%
for img in data_loader_guess:
    image, label = img
    images_guess.append(image.cuda())

for img in data_loader_validation:
    image, label = img
    images_validation.append(image.cuda())

#%%
if(not guess_mode):
    epochs = 100
    lossesList = []
else:
    epochs = 1

#%%
if(guess_mode or ( continue_training and (not guess_mode) )):
    network.load_state_dict(torch.load('./model2.pth'))
    network.eval()

#%%
if(not guess_mode):
    validationAvgOld = 99999

#%%
for epoch in range(epochs):
    guessList = []
    if(not guess_mode):
        for imageIndex, image in enumerate(images_in):
            guess = network(image, training)
            optimizer.zero_grad()
            loss = criterion(guess.view(-1), images_out[imageIndex].view(-1))
            lossesList.append(loss.item())
            loss.backward()
            optimizer.step()
        validationLossesList = []
        for imageIndex, image in enumerate(images_guess):
            guess = network(image, training)
            loss = criterion(guess.view(-1), images_validation[imageIndex].view(-1))
            validationLossesList.append(loss.item())
            guessList.append(guess)
        if(training_validation):
            validationAvg = sum(validationLossesList) / len(validationLossesList)
            if(validationAvgOld):
                if(validationAvg > validationAvgOld):
                    print("Validation condition met.")
                    break
            validationAvgOld = validationAvg
    else:
        for imageIndex, image in enumerate(images_guess):
            guess = network(image, training)
            guessList.append(guess)

#%%
if(not guess_mode):
    torch.save(network.state_dict(), './model2.pth')

#%%
fig = plt.figure(figsize=(36, 64))
grid = plt.GridSpec(30, 2, figure=fig, wspace=0, hspace=0 )
for imageIndex, image in enumerate(guessList):
    plt.subplot(grid[imageIndex, 0])
    plt.imshow(transforms.ToPILImage()(image[0].cpu().detach()))
for imageIndex, image in enumerate(images_validation):
    plt.subplot(grid[imageIndex, 1])
    plt.imshow(transforms.ToPILImage()(image[0].cpu().detach()))

#%%
if(not guess_mode):
    summing = 0
    currentIndex = 0
    averagesArray = []
    avg = 0
    count = 0

    for index, loss in enumerate(lossesList):
        for i in range(1000):
            if(index - i >= 0):
                count = count + 1
                newIndex = index - i
                avg = avg + lossesList[newIndex]
        averagesArray.append(avg / count)
        avg = 0
        count = 0
    
    plt.plot(lossesList, color='g')
    plt.plot(averagesArray, color='r')
    plt.show()