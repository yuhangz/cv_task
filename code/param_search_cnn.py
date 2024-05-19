# 下面是 python 得内建库
# 导入functools库中的reduce函数，该函数将一个二元操作符应用于序列的项，将其规约到单个值  
from functools import reduce  
  
# 导入operator库，它提供了一系列内置的运算符函数  
import operator  
  
# 导入copy库，提供了浅拷贝和深拷贝对象的方法  
import copy  
  
# 导入time库，提供了与时间相关的各种函数  
import time

# 导入shutil库，提供了高级文件操作功能，如复制、删除目录和文件  
import shutil

# 基本的库，用于操作文件系统和系统路径
import os

# 导入PyTorch库，PyTorch是一个用于深度学习的开源库  
import torch  
  
# 导入PyTorch中的神经网络模块，提供了构建神经网络所需的各种层  
import torch.nn as nn  
  
# 导入PyTorch中的优化器模块，提供了各种优化算法如SGD, Adam等  
import torch.optim as optim  
  
# 导入PyTorch中的功能函数模块，提供了一系列激活函数、损失函数等  
import torch.nn.functional as F  
  
# 从PyTorch的utils模块中导入SummaryWriter，用于TensorBoard的日志记录  
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter  
  
# 从torchvision库中导入datasets和transforms模块  
# torchvision是PyTorch的一个扩展库，提供了计算机视觉相关的数据集、模型和图像转换工具  
from torchvision import datasets, transforms  
  
# 导入yaml库，用于读取和写入YAML格式的文件  
import yaml  
  
# 导入matplotlib的pyplot模块，用于绘制图表和可视化数据  
import matplotlib.pyplot as plt  
  
# 导入numpy库，一个强大的科学计算库，提供了多维数组对象、数学函数等  
import numpy as np

from sklearn.metrics import accuracy_score

# 自定义 type
from type import HyperParameter, OptimizerType, LrScheduler, ActivationFunction, Regularization, search_space



def make_model(param: HyperParameter) -> nn.Sequential:
    activation = getattr(nn, param.activation_function)()
    use_batch_norm = param.use_batch_normalization
    dropout_rate = param.dropout_rate
    
    layers = []
    linear_input = copy.deepcopy(input_size)

    for i in range(len(conv_layers)):
        if i == 0:
            layers.append(nn.Conv2d(input_size[0], conv_layers[i], filter_size[i], stride[i], padding[i]))
        else:
            layers.append(nn.Conv2d(conv_layers[i-1], conv_layers[i], filter_size[i], stride[i], padding[i]))
        layers.append(activation)
        if pool_type == 'MaxPool':
            layers.append(nn.MaxPool2d(pool_size[i], pool_stride[i]))
        elif pool_type == 'AvgPool':
            layers.append(nn.AvgPool2d(pool_size[i], pool_stride[i]))
        layers.append(nn.BatchNorm2d(conv_layers[i])) if use_batch_norm else nn.Identity(),
        layers.append(nn.Dropout(dropout_rate)) if dropout_rate else nn.Identity(),
        linear_input[0] = conv_layers[i]
        linear_input[1] = (linear_input[1] - filter_size[i] + 2*padding[i]) // stride[i] + 1
        linear_input[2] = (linear_input[2] - filter_size[i] + 2*padding[i]) // stride[i] + 1
        linear_input[1] = (linear_input[1] - pool_size[i] ) // pool_stride[i] + 1
        linear_input[2] = (linear_input[2] - pool_size[i] ) // pool_stride[i] + 1      
        
    product = reduce(operator.mul, linear_input, 1)
    layers.append(nn.Flatten(start_dim=1))
    for i in range(len(hidden_layers)):
        if i == 0:
            layers.append(nn.Linear(product, hidden_layers[i]))
        else:
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        layers.append(activation)
        layers.append(nn.BatchNorm1d(hidden_layers[i])) if use_batch_norm else nn.Identity(),
        layers.append(nn.Dropout(dropout_rate)) if dropout_rate else nn.Identity(),

    layers.append(nn.Linear(hidden_layers[-1], output_size))
    model = nn.Sequential(*layers)
    return model

def get_cross_valiation_score_by_fold_id(
    train_loader: torch.utils.data.DataLoader,
    param: HyperParameter,
    k_fold_id: int,
    k_fold: int
) -> tuple[float, float]:
    model = make_model(param)
    optimizer_type = param.optimizer_type
    lr_scheduler = param.lr_scheduler
    regularization = param.regularization
    
    optimizer = getattr(optim, optimizer_type)(model.parameters(), lr=learning_rate)

    if lr_scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    # define the loss function
    if regularization == 'L1':
        criterion = nn.CrossEntropyLoss() # or whatever loss function you are using
        def loss_function(output, target):
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            return criterion(output, target) + l1_lambda * l1_norm
    elif regularization == 'L2':
        criterion = nn.CrossEntropyLoss()  # or whatever loss function you are using
        def loss_function(output, target):
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            return criterion(output, target) + l2_lambda * l2_norm
    else:
        loss_function = nn.CrossEntropyLoss()  # or whatever loss function you are using

    log_dir = f"hidden_layers_{'_'.join(map(str, hidden_layers))}_activation_{param.activation_function}"

    # create the SummaryWriter
    writer = SummaryWriter(f'runs/CNN/{log_dir}')
    shutil.copy('config/config_mlp.yaml', f'runs/CNN/{log_dir}/config.yaml')

    model = model.cuda()
    model.train()
    scores = []
    losses = []
    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            indice = torch.arange(len(data))
            # k fold
            train_indice = indice % k_fold != k_fold_id
            val_indice = indice % k_fold == k_fold_id
            train_data = data[train_indice]
            train_target = target[train_indice]
            val_data = data[val_indice]
            val_target = target[val_indice]
            
            data, target = train_data.cuda(), train_target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = loss_function(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running trainngi loss
            train_loss += loss.item() * data.size(0)
            
            # val
            with torch.no_grad():
                val_data = val_data.cuda()
                output: torch.Tensor = model(val_data).cpu()
                pred = output.argmax(1)
                score = accuracy_score(pred.flatten(), pred.flatten())
                loss = criterion(output, val_target)

                losses.append(loss.item())
                scores.append(score)
            
        scheduler.step()    
        # print training statistics 
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.dataset)
        writer.add_scalar('Training Loss', train_loss, epoch)
        # print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        #     epoch+1, 
        #     train_loss
        # ))
        # close the SummaryWriter
        writer.close()
        
        mean_score = torch.mean(scores)
        mean_loss = torch.mean(losses)
        return mean_score, mean_loss

def get_cross_valiation_score(
        train_loader,
        param,
        k_fold=4
) -> int:
    scores = []
    losses = []
    for k_fold_id in range(k_fold):
        mean_score, mean_loss = get_cross_valiation_score_by_fold_id(train_loader, param, k_fold_id, k_fold)
        scores.append(mean_score)
        losses.append(mean_loss)
    return torch.mean(scores), torch.mean(losses)

if __name__ == '__main__':
    with open('config/config_cnn.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 模型参数
    conv_layers = config['model']['conv_layers']
    filter_size = config['model']['filter_size']
    stride = config['model']['stride']
    padding = config['model']['padding']
    pool_type = config['model']['pool_type']
    pool_size = config['model']['pool_size']
    pool_stride = config['model']['pool_stride']
    hidden_layers = config['model']['hidden_layers']
    input_size = config['model']['input_size']
    output_size = config['model']['output_size']

    learning_rate = config['optimizer']['learning_rate']
    step_size = config['optimizer']['step_size']
    gamma = config['optimizer']['gamma']
    l1_lambda = config['model']['l1_lambda']
    l2_lambda = config['model']['l2_lambda']
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']  # suggest training between 20-50 epochs

    num_workers = 12

    # 转换到 tensor 的 pipe
    transform = transforms.ToTensor()

    # 3. 导入数据，根据 train 参数是否为 True 分为训练集和测试集
    train_data = datasets.EMNIST(root='./data',split='balanced', train=True,
                                    download=True, transform=transform)
    test_data = datasets.EMNIST(root='./data',split='balanced', train=False,
                                    download=True, transform=transform)

    # 4.a Print-out the number of training/testing samples in the dataset
    print("the number of train data samples: ", len(train_data))
    print("the number of test data samples: ", len(test_data))

    # 创建训练集的数据加载器
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        num_workers=num_workers)
    
    param = HyperParameter()
    
    # 初始化搜索参数
    param.activation_function = ActivationFunction.ELU
    param.dropout_rate = 0.1
    param.optimizer_type = OptimizerType.Adam
    param.regularization = Regularization.L1
    param.lr_scheduler = LrScheduler.StepLR
    param.use_batch_normalization = True
    
    for param_name in search_space:
        option_results = {k: None for k in search_space[param_name]}
        for option in option_results:
            setattr(param, option)
            score, loss = get_cross_valiation_score(train_loader, param, k_fold=4)
            option_results[option] = (score, loss)
        
        print('test {}, result {}'.format(param_name, option_results))
        best_option = sorted(option_results, key=lambda x : option_results[x][0], reverse=True)[0]
        # 设置为最好的
        setattr(param, best_option)
        print('select {} as best option in {}'.format(best_option, param_name))