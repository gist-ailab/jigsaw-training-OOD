import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import argparse

from utils import *
from evaluation import *
import data 

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'vit_tiny_patch16_224', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--method' ,'-m', default = 'msp', type=str)

    args = parser.parse_args()

    config = read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    dataset_path = config['id_dataset']
    batch_size = config['batch_size']
    save_path = config['save_path'] + args.save_path
    
    num_classes = int(config['num_classes']) 
    if 'cnc' in args.method:
        num_classes += 1

    if 'cifar' in args.data:
        _, valid_loader = get_cifar(args.data, dataset_path, batch_size)
    elif 'aircraft' == args.data:
        datasplit = data.get_class_splits('aircraft')
        num_classes = len(datasplit[0])
        dataset = data.get_datasets('aircraft', image_size=384, split_train_val=False, train_classes=datasplit[0], open_set_classes=datasplit[1])
        
    elif 'scars' == args.data:
        datasplit = data.get_class_splits('scars')
        dataset = data.get_datasets('scars', image_size=384, split_train_val=False, train_classes=datasplit[0], open_set_classes=datasplit[1])
        num_classes = len(datasplit[0])
            
        train_data = dataset['train']
        test_data = dataset['test_known']
        osr_data = dataset['test_unknown']
        print(num_classes)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True, num_workers = 4)
        valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
        osr_loader = torch.utils.data.DataLoader(osr_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)

        
    if 'vit_tiny_patch16_224' == args.net:
        model = timm.create_model(args.net, pretrained=False, num_classes=num_classes)
    model = timm.create_model(args.net, pretrained=True, num_classes=num_classes)

    state_dict = (torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict'])    

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if args.method == 'msp':
        calculate_score = calculate_msp
    if args.method == 'odin':
        calculate_score = calculate_odin
    if args.method == 'norm':
        calculate_score = calculate_norm
    if args.method == 'allnm':
        calculate_score = calculate_allnm
    if args.method == 'cosine':
        calculate_score = calculate_cosine
    if args.method == 'energy':
        calculate_score = calculate_energy
    if args.method == 'gradnm':
        calculate_score = calculate_gradnorm
    if args.method == 'react':
        calculate_score = calculate_react
    if args.method == 'md':
        calculate_score = calculate_mdscore
    if args.method == 'godn':
        calculate_score = calculate_godin
    if args.method == 'mls':
        calculate_score = calculate_mls
    if args.method == 'cnc':
        calculate_score = calculate_cnc
    if args.method == 'dice':
        calculate_score = calculate_msp
    f = open(save_path+'/{}_result.txt'.format(args.method), 'w')
    valid_accuracy = validation_accuracy(model, valid_loader, device)

    print('In-distribution accuracy: ', valid_accuracy)
        
    f.write('Accuracy for ValidationSet: {}\n'.format(str(valid_accuracy)))
    #MSP
    #image_norm(valid_loader)  
    preds_in = calculate_score(model, valid_loader, device).cpu()
    if 'cifar' in args.data:
        OOD_results(preds_in, model, get_svhn('/SSDb/Workspaces/yeonguk.yu/data/svhn', batch_size), device, args.method+'-SVHN', f)
        OOD_results(preds_in, model, get_textures('/SSDb/Workspaces/yeonguk.yu/data/textures/images'), device, args.method+'-TEXTURES', f)
        OOD_results(preds_in, model, get_lsun('/SSDb/Workspaces/yeonguk.yu/data/LSUN'), device, args.method+'-LSUN', f)
        OOD_results(preds_in, model, get_lsun('/SSDb/Workspaces/yeonguk.yu/data/LSUN_resize'), device, args.method+'-LSUN-resize', f)
        OOD_results(preds_in, model, get_lsun('/SSDb/Workspaces/yeonguk.yu/data/iSUN'), device, args.method+'-iSUN', f)
        # OOD_results(preds_in, model, get_tiny_as_ood('/SSDe/yyg/tiny-imagenet-200', crop=True), device, args.method+'-tinyimagenet-crop', f)
        # OOD_results(preds_in, model, get_tiny_as_ood('/SSDe/yyg/tiny-imagenet-200', crop=False), device, args.method+'-tinyimagenet-resize', f)
        OOD_results(preds_in, model, get_places('/SSDb/Workspaces/yeonguk.yu/data/places'), device, args.method+'-Places365', f)
        cifar = 'cifar100' if args.data == 'cifar10' else 'cifar10'
        OOD_results(preds_in, model, get_cifar(cifar, '/SSDe/yyg/data/{}'.format(cifar), batch_size)[1], device, args.method+'-{}'.format(cifar), f)
    else:  
        OOD_results(preds_in, model, get_ood_folder('/SSDe/yyg/data/OOD_for_ImageNet/iNaturalist', batch_size, True), device, args.method+'-iNaturalist', f)
        OOD_results(preds_in, model, get_ood_folder('/SSDe/yyg/data/OOD_for_ImageNet/SUN', batch_size, True), device, args.method+'-SUN', f)
        OOD_results(preds_in, model, get_ood_folder('/SSDe/yyg/data/OOD_for_ImageNet/Places', batch_size, True), device, args.method+'-PLACES', f)
        OOD_results(preds_in, model, get_ood_folder('/SSDe/yyg/data/OOD_for_ImageNet/dtd/images', batch_size, True), device, args.method+'-Textures', f)
        OOD_results(preds_in, model, osr_loader, device, args.method+'-OSR', f)

  



    f.close()


if __name__ =='__main__':
    eval()