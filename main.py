import sys
import torch
import numpy as np
from model.AMCModel import AMCModel
from data.dataset.OFDMDataset import OFDMDataset, get_dataloaders
from torch.optim import Adam
import torch.nn as nn

def main():

    LR = 0.0003
    BATCH_SIZE = 1
    SHUFFLE = True
    EPOCHS = 1

    dataset = OFDMDataset()
    trainloader, testloader = get_dataloaders(dataset=dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    model = AMCModel()
    model = model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=LR)


    losses, predictions = [], []
    n_total_steps = len(trainloader)
    for epoch in range(EPOCHS):

        model = model.train()
        for i, (x, y) in enumerate(trainloader):
            optimizer.zero_grad()
            scores = model(x)
            loss = criterion(scores, torch.squeeze(y))
            loss.backward()
            optimizer.step()

            predictions.append(torch.argmax(scores).item())
            losses.append(loss.item())

            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{n_total_steps}], Loss: {np.mean(losses):.4f}')
                losses.clear()

        model = model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for x, y in testloader: 
                scores = model.forward(x)
                predicted = torch.argmax(scores).item()
                labels = torch.argmax(torch.squeeze(y)).item()
                n_samples += y.size(0)
                n_correct += predicted == labels

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the test set: {acc} %')

if __name__ == "__main__":
    sys.exit(main())
