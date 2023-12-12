import torch
import torch.nn.functional as F
import timm
import argparse

from torch.autograd import Variable

from utils import *

react_threshold = None
#MSP
def calculate_msp(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = torch.softmax(outputs, dim=1)
            outputs = outputs.max(dim=1).values
            predictions.append(outputs)
    predictions = torch.cat(predictions).to(device)
    return predictions

#MLS
def calculate_mls(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.max(dim=1).values
            predictions.append(outputs)
    predictions = torch.cat(predictions).to(device)
    return predictions


#CNC
def calculate_cnc(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            conf = torch.softmax(outputs, dim=-1)[:,-1]
            predictions.append(-conf)
    predictions = torch.cat(predictions).to(device)
    return predictions

#ODIN
"""
ODIN method
original code is on https://github.com/facebookresearch/odin
"""
def calculate_odin(model, loader, device):
    model.eval()
    predictions = []
    for batch_idx, (inputs,_) in enumerate(loader):
        inputs = inputs.to(device)
        inputs = Variable(inputs, requires_grad = True)
        outputs = model(inputs)
        
        #label
        labels = outputs.data.max(1)[1]
        labels = Variable(labels)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2
        gradient[:, 0] = gradient[:, 0]
        gradient[:, 1] = gradient[:, 1]
        gradient[:, 2] = gradient[:, 2]
        temp_inputs = torch.add(inputs.data, -0.0004* gradient)
        temp_inputs = Variable(temp_inputs)

        with torch.no_grad():
            outputs = model(temp_inputs)
            outputs = torch.softmax(outputs/1000.0, dim=1)
            outputs = outputs.max(1)[0]
        # outputs = torch.norm(x, dim=1, keepdim=True)
        predictions.append(outputs)
    predictions = torch.cat(predictions).to(device)
    return predictions


#Norm
def calculate_norm(model, loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            x = inputs.to(device)
            out = model(x)            
            # norm = torch.max(out, dim=1).values
            
            norm = torch.norm(F.relu(out), p=2, dim=1)

            predictions.append(norm)
    predictions = torch.cat(predictions).to(device)
    return predictions

"""
Energy score
source code from 'https://github.com/deeplearning-wisc/gradnorm_ood/blob/master/test_ood.py'
"""
def calculate_energy(model, loader, device):
    model.eval()
    predictions = []
  
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)

            energy = torch.logsumexp(outputs.data, dim=1)
            predictions.append(energy)
    predictions = torch.cat(predictions).to(device)
    return predictions    

"""
GradNorm score
source code from 'https://github.com/deeplearning-wisc/gradnorm_ood/blob/master/test_ood.py'
"""
def calculate_gradnorm(model, loader, device):
    model.eval()
    predictions = []

    logsoftmax = torch.nn.LogSoftmax(dim=-1).to(device)
    for batch_idx, (inputs, t) in enumerate(loader):
        # for image in inputs:
        #     image = image.unsqueeze(0).to(device)
            image = inputs.to(device)

            model.zero_grad()
            outputs = model(image)
            targets = torch.ones((image.shape[0], outputs.size(1))).to(device)
            
            loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))
            loss.backward()

            layer_grad = model.fc.weight.grad.data
            layer_grad_norm = torch.sum(torch.abs(layer_grad), dim=0).view(-1,1)
            # print(layer_grad.shape, layer_grad_norm.shape)

            predictions.extend(layer_grad_norm)

    predictions = torch.cat(predictions).to(device)
    return predictions    


"""
ReAct + Energy
source code from "https://github.com/deeplearning-wisc/react"
"""
def calculate_react(model, loader, thr, device):
    threshold = thr#1.0#1.5 #2.4
    model.eval()
    predictions = []        
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            inputs = inputs.to(device)
            
            x = model.conv1(inputs)
            x = model.bn1(x)
            x = F.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            x = model.avgpool(x)
            features = torch.flatten(x, 1)
            # features = model.forward_features(inputs)
            # features = model.global_pool(features) if type(model).__name__ == 'ResNet' else features         
            features = torch.clip(features, max=threshold)
            outputs = model.fc(features)
            energy = torch.logsumexp(outputs, dim=1)
            predictions.append(energy)
    predictions = torch.cat(predictions).to(device)
    return predictions  


"""
Isomax: minimum distance score
source code from "https://github.com/dlmacedo/entropic-out-of-distribution-detection"
"""
def calculate_mdscore(model, loader, device):
    model.eval()
    predictions = []        
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            score, _ = outputs.max(dim=1)
            predictions.append(score)
    predictions = torch.cat(predictions).to(device)
    return predictions  



def OOD_results(preds_id, model, loader, device, method, file):  
    #image_norm(loader)  
    if 'msp' in method:
        preds_ood = calculate_msp(model, loader, device).cpu()
    if 'odin' in method:
        preds_ood = calculate_odin(model, loader, device).cpu()
    if 'norm' in method:
        preds_ood = calculate_norm(model, loader, device).cpu()
    if 'energy' in method:
        preds_ood = calculate_energy(model, loader, device).cpu()
    if 'gradnm' in method:
        preds_ood = calculate_gradnorm(model, loader, device).cpu()
    if 'react' in method:
        preds_ood = calculate_react(model, loader, device).cpu()
    if 'md' in method:
        preds_ood = calculate_mdscore(model, loader, device).cpu()
    if 'mls' in method:
        preds_ood = calculate_mls(model, loader, device).cpu()
    if 'cnc' in method:
        preds_ood = calculate_cnc(model, loader, device).cpu()
    if 'dice' in method:
        preds_ood = calculate_msp(model, loader, device).cpu()
    print(torch.mean(preds_ood), torch.mean(preds_id))
    fpr, auroc = show_performance(preds_id, preds_ood, method, file=file)
    return fpr, auroc