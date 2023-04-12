import json
import torch.utils.data as data
import numpy as np
import torch
import os
import random
import glob

from PIL import Image
from scipy import io
from torchvision import transforms
from torchvision import datasets as dset
import torchvision

# mean = [0.4914, 0.4822, 0.4465]
# std = [0.2023, 0.1994, 0.2010]

mean=[0.485, 0.456, 0.406]
std= [0.229, 0.224, 0.225]

EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = 'wnids.txt'
VAL_ANNOTATION_FILE = 'val_annotations.txt'




corrupt_name = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                    'glass_blur', 'snow', 'fog', 'zoom_blur',
                     'contrast', 'elastic_transform', 'brightness',
                    'speckle_noise', 'gaussian_blur', 'saturate', 'frost']

from imagecorruptions import corrupt


class jigsaw_train_dataset(data.Dataset):
    def __init__(self, dataset, transform, is_tinyimagenet = True):
        self.dataset = dataset
        self.transform = transform

        self.is_imagenet = is_tinyimagenet
        if self.is_imagenet:
            self.open = Image.open
        else:
            self.open = Image.fromarray
        print(type(self.dataset))
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x, y = self.dataset[index]    

        # img = self.dataset.data[index]
        if self.is_imagenet:
            img_ = self.dataset.image_paths[index]
        else:
            img_ = self.dataset.data[index]
        img_ = self.open(img_)
        img_ = img_.convert('RGB')

        img = self.transform(img_)
        x_ = torch.zeros_like(img)
        jigsaw_num = 3#np.random.choice([3])

        s = int(float(x.size(1)) / jigsaw_num)

        tiles_order = random.sample(range(jigsaw_num**2), jigsaw_num**2)
        for o, tile in enumerate(tiles_order):
            i = int(o/jigsaw_num)
            j = int(o%jigsaw_num)
            
            ti = int(tile/jigsaw_num)
            tj = int(tile%jigsaw_num)
            # print(i, j, ti, tj)
            x_[:, i*s:(i+1)*s, j*s:(j+1)*s] = img[:, ti*s:(ti+1)*s, tj*s:(tj+1)*s] 
        return x, x_, y

def get_cifar_jigsaw(dataset, folder, batch_size, size = 224):
    train_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == 'cifar10':
        train_data = dset.CIFAR10(folder, train=True, transform=train_transform_cifar, download=True)
        test_data = dset.CIFAR10(folder, train=False, transform=test_transform_cifar, download=True)

    else:
        train_data = dset.CIFAR100(folder, train=True, transform=train_transform_cifar, download=True)
        test_data = dset.CIFAR100(folder, train=False, transform=test_transform_cifar, download=True)


    transform_pre = transforms.Compose([
        transforms.Resize([size,size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    jigsaw = jigsaw_train_dataset(train_data, train_transform_cifar, is_tinyimagenet=False)

    train_loader = torch.utils.data.DataLoader(jigsaw, batch_size, shuffle=True, pin_memory=True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    
    return train_loader, valid_loader

def get_cifar_cnc(dataset, folder, batch_size, size=224):
    transform_pre = transforms.Compose([
        transforms.Resize([size,size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_train = transforms.Compose([
        transforms.Resize([size,size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.Resize([size,size]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataset == 'cifar10':
        train_data = dset.CIFAR10(folder, train=True, transform=transform_train, download=True)
        train_data_corr = dset.CIFAR10(folder, train=True, transform=transform_pre, download=True)
        test_data = dset.CIFAR10(folder, train=False, transform=transform_test, download=True)
    else:
        train_data = dset.CIFAR100(folder, train=True, transform=transform_train, download=True)
        train_data_corr = dset.CIFAR100(folder, train=True, transform=transform_pre, download=True)
        test_data = dset.CIFAR100(folder, train=False, transform=transform_test, download=True)
    from cnc import aug_dict

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True, num_workers = 4)
    cnc_loader = torch.utils.data.DataLoader(train_data_corr, batch_size, shuffle=True, pin_memory=True, num_workers = 8, collate_fn=aug_dict['cnc'])
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    
    return train_loader, cnc_loader, valid_loader


class TinyImageNet(data.Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """
    def __init__(self, root='', split='train', transform=None, target_transform=None, in_memory=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing

        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        if self.split == 'test':
            return img
        else:
            # file_name = file_path.split('/')[-1]
            return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_image(self, path):
        img = Image.open(path)
        if (img.mode == 'L'):
            img = img.convert('RGB')
        return self.transform(img) if self.transform else img

def get_tinyimagenet(folder, batch_size, size = 64):
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=10) if size == 64 else transforms.Resize([size, size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([
        transforms.Resize([size, size]), 
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)])

    tiny_train = TinyImageNet(folder, split='train', transform = train_transform, in_memory=False)
    tiny_val = TinyImageNet(folder, split='val', transform = test_transform, in_memory=False)

    train_loader = torch.utils.data.DataLoader(tiny_train, batch_size, shuffle=True, pin_memory=True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(tiny_val, batch_size, shuffle=False, pin_memory=True, num_workers = 4)

    return train_loader, valid_loader
    
def get_tinyimagenet_cnc(folder, batch_size, size = 64, plot=False):
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=10) if size == 64 else transforms.Resize([size, size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)])

    transform_pre = transforms.Compose([
        transforms.RandomCrop(64, padding=10) if size == 64 else transforms.Resize([size, size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize([size, size]), 
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)])

    from cnc import aug_dict

    if plot == True:
        train_data_corr = torchvision.datasets.ImageFolder(folder, transform_pre) 
        return torch.utils.data.DataLoader(train_data_corr, 4, shuffle=True, pin_memory=True, num_workers = 4, collate_fn=aug_dict['cnc'])

    train_data = TinyImageNet(folder, split='train', transform = train_transform, in_memory=False)
    train_data_corr = TinyImageNet(folder, split='train', transform = transform_pre, in_memory=False)
    test_data = TinyImageNet(folder, split='val', transform = test_transform, in_memory=False)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True, num_workers = 4)
    cnc_loader = torch.utils.data.DataLoader(train_data_corr, batch_size, shuffle=True, pin_memory=True, num_workers = 4, collate_fn=aug_dict['cnc'])
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    
    return train_loader, cnc_loader, valid_loader

def get_tinyimagenet_jigsaw(folder, batch_size, size = 64, plot=False):
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=10) if size == 64 else transforms.Resize([size, size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)])

    transform_pre = transforms.Compose([
        transforms.RandomCrop(64, padding=10) if size == 64 else transforms.Resize([size, size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize([size, size]), 
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)])

    if plot:
        trainset = torchvision.datasets.ImageFolder(folder, train_transform) 
        jigsaw = jigsaw_train_dataset(trainset, transform_pre, is_tinyimagenet=True)
        loader = torch.utils.data.DataLoader(jigsaw, 4, shuffle=True, pin_memory=True, num_workers = 4)
        return loader
    
    train_data = TinyImageNet(folder, split='train', transform = train_transform, in_memory=False)
    test_data = TinyImageNet(folder, split='val', transform = test_transform, in_memory=False)
    jigsaw = jigsaw_train_dataset(train_data, transform_pre, is_tinyimagenet=True)

    train_loader = torch.utils.data.DataLoader(jigsaw, batch_size, shuffle=True, pin_memory=True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    
    return train_loader, valid_loader

def get_cifar_test(dataset, folder, batch_size, test=False):
    test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform_cifar_blur = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ = not test
    
    if dataset == 'cifar10':
        train_data = dset.CIFAR10(folder, train=train_, transform=test_transform_cifar, download=True)
        test_data = dset.CIFAR10(folder, train=train_, transform=test_transform_cifar_blur, download=True)

    else:
        train_data = dset.CIFAR100(folder, train=train_, transform=test_transform_cifar, download=True)
        test_data = dset.CIFAR100(folder, train=train_, transform=test_transform_cifar_blur, download=True)
    jigsaw = jigsaw_dataset(test_data)
    # print(jigsaw[0])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(jigsaw, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    
    return train_loader, valid_loader

# test_transforms = test_transform_cifar
def get_tiny_as_ood(path, batch_size=100, crop = True):
    transform_crop = transforms.Compose([
        transforms.CenterCrop([32,32]),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_resize = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ood_data = torchvision.datasets.ImageFolder(os.path.join(path, 'test'), transform_crop if crop else transform_resize)        
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader

def read_conf(json_path):
    """
    read json and return the configure as dictionary.
    """
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config
    
def get_cifar(dataset, folder, batch_size, eval=False, size=224):
    train_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor(), transforms.Normalize(mean, std)])
    if eval==True:
        train_transform_cifar_ = test_transform_cifar
    else:
        train_transform_cifar_ = train_transform_cifar
    if dataset == 'cifar10':
        train_data = dset.CIFAR10(folder, train=True, transform=train_transform_cifar_, download=True)
        test_data = dset.CIFAR10(folder, train=False, transform=test_transform_cifar, download=True)
        num_classes = 10
    else:
        train_data = dset.CIFAR100(folder, train=True, transform=train_transform_cifar_, download=True)
        test_data = dset.CIFAR100(folder, train=False, transform=test_transform_cifar, download=True)
        num_classes = 100
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    
    return train_loader, valid_loader

def get_train_svhn(folder, batch_size):
    train_data = dset.SVHN(folder, split='train', transform=test_transform_cifar, download=True)    
    test_data = dset.SVHN(folder, split='test', transform=test_transform_cifar, download=True)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)     
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    return train_loader, valid_loader
    
def get_outlier(path, batch_size):
    class temp(torch.utils.data.Dataset):
        def __init__(self, path, transform=None):
            self.data = np.load(path)
            self.transform = transform

        def __getitem__(self, index):
            data = self.data[index]
            data = self.transform(data)
            return data

        def __len__(self):
            return len(self.data)
    
    test_data = temp(path, transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(), transforms.ToTensor()]))
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, pin_memory=True, num_workers = 4)    
    return valid_loader

def get_svhn(folder, batch_size, size=224):
    test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_data = dset.SVHN(folder, split='test', transform=test_transform_cifar, download=True)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, pin_memory=False, num_workers = 4)    
    return valid_loader

def get_svhn_test(folder, batch_size):
    test_transform_cifar = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor()])
    test_transform_cifar_blur = transforms.Compose([transforms.Resize([4,4]), transforms.Resize([32,32]), transforms.ToTensor()])

    test_data = dset.SVHN(folder, split='test', transform=test_transform_cifar, download=True)
    test_data_blur = dset.SVHN(folder, split='test', transform=test_transform_cifar_blur, download=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    blur_loader = torch.utils.data.DataLoader(test_data_blur, batch_size, shuffle=False, pin_memory=False, num_workers = 4)    

    
    return test_loader, blur_loader

def get_textures(path, size=224):
    test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor(), transforms.Normalize(mean, std)])
    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=128, shuffle=False, pin_memory=False)
    return ood_loader

def get_ood_blur(path):
    test_transform_cifar = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor()])
    test_transform_cifar_blur = transforms.Compose([transforms.Resize([4,4]), transforms.Resize([32,32]), transforms.ToTensor()])

    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_data_blur = torchvision.datasets.ImageFolder(path, test_transform_cifar_blur)

    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    ood_loader_blur = torch.utils.data.DataLoader(ood_data_blur, batch_size=100, shuffle=False, pin_memory=False)
    return ood_loader, ood_loader_blur

def get_lsun(path, size=224):
    test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor(), transforms.Normalize(mean, std)])
    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=True, pin_memory=False)
    return ood_loader    

def get_places_blur(path):
    test_transform_cifar = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor()])
    test_transform_cifar_blur = transforms.Compose([transforms.Resize([4,4]), transforms.Resize([32,32]), transforms.ToTensor()])

    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_data_blur = torchvision.datasets.ImageFolder(path, test_transform_cifar_blur)

    random.seed(0)
    ood_data.samples = random.sample(ood_data.samples, 10000)
    ood_data_blur.samples = random.sample(ood_data.samples, 10000)

    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    ood_loader_blur = torch.utils.data.DataLoader(ood_data_blur, batch_size=100, shuffle=False, pin_memory=False)
    return ood_loader, ood_loader_blur

def get_places(path, size=224):
    test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor(), transforms.Normalize(mean, std)])

    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)

    random.seed(0)
    ood_data.samples = random.sample(ood_data.samples, 10000)

    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=True, pin_memory=False)
    return ood_loader

def get_mnist(path, batch_size=100, transform_imagenet = False):
    ood_data = dset.MNIST(path, train=False, transform=test_transform_224_gray if transform_imagenet else test_transform_gray, download=True)
    print(ood_data[0][0].shape)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    return ood_loader 

def get_knist(path):
    ood_data = dset.KMNIST(path, train=False, transform=test_transform_gray, download=True)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader 

def get_fnist(path):
    ood_data = dset.FashionMNIST(path, train=False, transform=test_transform_gray, download=True)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader 

# def get_food101(path):
#     ood_data = dset.Food101(path, split='test', transform=test_transform_cifar, download=True)
#     ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
#     return ood_loader 

def get_stl10(path):
    ood_data = dset.STL10(path, split='test', transform=test_transform_cifar, download=True)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader 


def get_folder(path):
    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=False)
    return ood_loader 

def get_blob():
    # /////////////// Blob ///////////////
    ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(10000, 32, 32, 3)))
    for i in range(10000):
        ood_data[i] = gblur(ood_data[i], sigma=1.5, channel_axis=False)
        ood_data[i][ood_data[i] < 0.75] = 0.0

    dummy_targets = torch.ones(10000)
    ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=True, pin_memory=False)
    return ood_loader

def get_gaussian():
    dummy_targets = torch.ones(50000)
    gaussian = np.random.randint(0, 255, [50000, 3, 32, 32], dtype=int)
    gaussian = gaussian/255.0
    # print(gaussian)
    ood_data = torch.from_numpy(np.float32(gaussian))
    # ood_data = torch.from_numpy(np.float32(np.clip(
    #     np.ones([50000, 3, 32, 32]), 0, 1))
    # )
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size = 128, shuffle=True)
    return ood_loader

def get_rademacher():
    dummy_targets = torch.ones(10000)
    ood_data = torch.from_numpy(np.random.binomial(
        n=1, p=0.5, size=(10000, 3, 32, 32)).astype(np.float32)) * 2 - 1
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=True)
    return ood_loader


def get_domainnet(path, split, subset, batch_size, eval=False):
    train_trans = train_transforms
    test_trans = test_transforms
    if eval:
        train_trans = test_transforms
    trainset = DomainNetClass(path, split, subset, train=True, transform=train_trans)
    testset = DomainNetClass(path, split, subset, train=False, transform=test_trans)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_ood_folder(path, batch_size = 32, sample_cut = False):
    oodset = torchvision.datasets.ImageFolder(path, test_transforms)
    if sample_cut:
        oodset.samples = oodset.samples[:3000]
    ood_loader = torch.utils.data.DataLoader(oodset, batch_size, shuffle = True, pin_memory = False, num_workers = 4)
    return ood_loader
    
if __name__ == '__main__':
    pass