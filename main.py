import sys
import torch
from model.ofdm_amc_net import OFDM_AMC_Model
from data.dataset.ofdm_amc_dataset import OFDM_AMC_Dataset, getDataLoaders
from torch.optim import Adam
import torch.nn as nn

def main():

    LR = 0.001
    BATCH_SIZE = 16
    SHUFFLE = True
    EPOCHS = 5

    dataset = OFDM_AMC_Dataset()
    trainloader, testloader = getDataLoaders(dataset=dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    model = OFDM_AMC_Model().double()
    criterion = nn.BCELoss()
    optimizer = Adam(params=model.parameters(), lr=LR)

    n_total_steps = len(trainloader)
    for epoch in range(EPOCHS):
        for i, (x, y) in enumerate(trainloader):
            scores = model.forward(x)
            loss = criterion(scores,torch.squeeze(y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    
    param_cnt = 0
    for param in model.parameters():
        param_cnt += torch.prod(torch.tensor(param.shape))

    print(f'Number of parameters: {param_cnt:,}')
    torch.save(model.state_dict(), "/Users/gtosun/Documents/vsc_workspace/ofdm-amc/min_model.pth")

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for x, y in testloader:
            scores = model.forward(x.double())
            _, predicted = torch.max(scores.data, 1)
            _, labels = torch.max(torch.squeeze(y), 1)
            n_samples += y.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test set: {acc} %')

if __name__ == "__main__":
    sys.exit(main())
