import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import nni.retiarii.strategy as strategy
import nni
from torchvision import transforms
from torch.utils.data import DataLoader
import dataset
import math
import os
import random
from pathlib import Path

import nni
import torch
import torch.nn.functional as F
# remember to import nni.retiarii.nn.pytorch as nn, instead of torch.nn as nn
import nni.retiarii.nn.pytorch as nn
import nni.retiarii.strategy as strategy
from nni.retiarii import model_wrapper
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment, debug_mutated_model
from torch.utils.data import DataLoader
from torchvision import transforms


def psnr(img1, img2):
    mse = torch.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


#optimized_params = nni.get_next_parameter()
#params.update(optimized_params)
#print(params)


@model_wrapper
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.LayerChoice([
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.Conv2d(32, 16, 1, 1, 0),
            nn.Conv2d(32, 16, 5, 1, 2)
            #DepthwiseSeparableConv(32, 64)
        ])
        # LayerChoice is used to select a layer between Conv2d and DwConv.
        self.conv3 = nn.LayerChoice([
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Conv2d(16, 3, 1, 1, 0)
            #DepthwiseSeparableConv(32, 64)
        ])
        # ValueChoice is used to select a dropout rate.
        # ValueChoice can be used as parameter of modules wrapped in `nni.retiarii.nn.pytorch`
        # or customized modules wrapped with `@basic_unit`.
        #self.dropout1 = nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75]))  # choose dropout rate from 0.25, 0.5 and 0.75
        #self.dropout2 = nn.Dropout(0.5)
        #feature = nn.ValueChoice([64, 128, 256])
        #self.fc1 = nn.Linear(9216, feature)
        #self.fc2 = nn.Linear(feature, 10)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        output = F.relu(self.conv2(x))
        output = F.relu(self.conv3(output))
        #x = torch.flatten(self.dropout1(x), 1)
        #x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
        #output = F.log_softmax(x, dim=1)
        return output + inputs


#model_space = ModelSpace()
#search_strategy = strategy.Random(dedup=True)



content_folder1 = '/home/wenwen/dataset/dehaze/4k_test_input/'
information_folder1 = '/home/wenwen/dataset/dehaze/4k_test_gt/'



def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.MSELoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float().to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device).float()
            output = model(data)
            correct += psnr(output, target)
            #pred = output.argmax(dim=1, keepdim=True)
            #correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    #accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), correct/len(test_loader.dataset)))

    return correct/len(test_loader.dataset)


def evaluate_model(model_cls):
    # "model_cls" is a class, need to instantiate
    model = model_cls()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #transf = transforms.Compose([transforms.ToTensor()])
    train_loader = dataset.style_loader(content_folder1, information_folder1, 512, 1)
    test_loader = dataset.style_loader(content_folder1, information_folder1, 512, 1)

    for epoch in range(3):
        # train the model for one epoch
        train_epoch(model, device, train_loader, optimizer, epoch)
        # test the model for one epoch
        accuracy = test_epoch(model, device, test_loader)
        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(accuracy)

    # report final test result
    nni.report_final_result(accuracy)

#evaluate_model(ModelSpace)


if __name__ == '__main__':
    base_model = Net()

    search_strategy = strategy.Random()
    model_evaluator = FunctionalEvaluator(evaluate_model)

    exp = RetiariiExperiment(base_model, model_evaluator, [], search_strategy)

    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'minist_search'
    exp_config.trial_concurrency = 2
    exp_config.max_trial_number = 20
    exp_config.training_service.use_active_gpu = False
    export_formatter = 'dict'

    # uncomment this for graph-based execution engine
    # exp_config.execution_engine = 'base'
    # export_formatter = 'code'

    exp.run(exp_config, 8080)
    print('Final model:')
    for model_code in exp.export_top_models(formatter=export_formatter):
        print(model_code)






