from utils.utils import *
# from ntl_utils.getdata import Cus_Dataset
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import os
import cv2


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class Cus_Dataset(data.Dataset):
    def __init__(self, mode = None, \
                            dataset = None, begin_ind = 0, size = 0,\
                            is_img_path = False, is_img_path_aug = [None, None, None, None],
                            spec_dataTransform = None, config=None):
        # Init
        self.mode = mode
        self.list_img = []    
        self.list_label = []
        self.data_size = size
        self.is_img_path = is_img_path
        self.is_img_path_aug = is_img_path_aug
        
        # Transform
        if spec_dataTransform is None: 
            dataTransform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            self.transform = dataTransform
            # for data-free KD
            self.nonpil_transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else: 
            self.transform = spec_dataTransform
        # self.transform11 = dataTransform_ori

        # Load Data
        if 'authorized_training' in self.mode: 
            # Get img and label
            path_list = dataset[0][begin_ind: begin_ind+size]
            label_list1 = dataset[1][begin_ind: begin_ind+size]
            for i in range(size):
                self.list_img.append(path_list[i])
                self.list_label.append(label_list1[i])

            # Shuffle
            ind = np.arange(self.data_size)
            ind = np.random.permutation(ind)
            self.list_img = np.asarray(self.list_img)
            self.list_img = self.list_img[ind]
            self.list_label = np.asarray(self.list_label)
            self.list_label = self.list_label[ind]
        elif self.mode == 'val':
            # Get img and label
            path_list = dataset[0][begin_ind: begin_ind+size]
            if self.is_img_path:
                for file_path in path_list:
                    img = np.array(rgb_loader(file_path))
                    self.list_img.append(img)
            else:
                for file_path in path_list:
                    self.list_img.append(file_path)

            self.list_label = dataset[1][begin_ind: begin_ind+size]
            
    def __getitem__(self, item):
        # get the item
        img = self.list_img[item]
        label = self.list_label[item]
        
        # img preprocess: path(optional) -> img -> tensor -> numpy -> transform
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = np.array(img)
        img = self.transform(img)

        
        # label img preprocess: int(optional) -> one-hot
        if isinstance(label, (int, np.int32, np.int64)):
            label = np.eye(345)[label]  # set to 345 for domain_net dataset
            label = np.array(label)
        
        # unsqueez the label for the case of val
        if self.mode == 'authorized_training_src':
            return img, torch.LongTensor(label)
        elif self.mode == 'val' or self.mode == 'style':
            return img, torch.LongTensor(label).unsqueeze(0)
       

    def __len__(self):
        return self.data_size


def split():
    image_size = 64
    # image_size = 32
    # sample_num = 5000
    data_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

    domain_dict = {
            #    'mt': get_mnist_data,
            #    'us': get_usps_data,
            #    'sn': get_svhn_data,
            #    'mm': get_mnist_m_data,
            #    'sd': get_syn_data,
               'cifar': get_cifar_data,
               'stl': get_stl_data,
            #    'visda_t': get_visda_data_src,
            #    'visda_v': get_visda_data_tgt,
            #    'home_art': get_home_art, 
            #    'home_cli': get_home_cli, 
            #    'home_pd': get_home_pd, 
            #    'home_rw': get_home_rw,
            #    'vlcs_v': get_vlcs_V,
            #    'vlcs_c': get_vlcs_C,
            #    'vlcs_l': get_vlcs_L,
            #    'vlcs_s': get_vlcs_S,
            #    'pacs_p': get_pacs_P,
            #    'pacs_a': get_pacs_A,
            #    'pacs_c': get_pacs_C,
            #    'pacs_s': get_pacs_S,
            #    'ti_l38': get_ti_l38,
            #    'ti_l43': get_ti_l43,
            #    'ti_l46': get_ti_l46,
            #    'ti_l100': get_ti_l100,
            #    'domain_net_cli': get_domain_net_cli,
            #    'domain_net_info': get_domain_net_info,
            #    'domain_net_paint': get_domain_net_paint,
            #    'domain_net_qd': get_domain_net_qd,
            #    'domain_net_real': get_domain_net_real,
            #    'domain_net_sketch': get_domain_net_sketch,
            #    'cmt10': get_mnist_color_10,
            #    'cmt20': get_mnist_color_20,
            #    'cmt90': get_mnist_color_90
            #    'imagenette': get_imagenette_data,
            #    'rmt0': get_mnist_rotate_0,
            #    'rmt15': get_mnist_rotate_15,
            #    'rmt30': get_mnist_rotate_30,
            #    'rmt45': get_mnist_rotate_45,
            #    'rmt60': get_mnist_rotate_60,
            #    'rmt75': get_mnist_rotate_75
               }

    dataset_names = list(domain_dict.keys())
    dataset_funcs = list(domain_dict.values())

    dataset_split_seed = 2021
    setup_seed(dataset_split_seed)

    if not os.path.exists('./data_presplit'):
        os.makedirs('./data_presplit')

    for name, dataset_funcs in zip(dataset_names, dataset_funcs):
        data = dataset_funcs()
        
        # Split data into train, val, test with 8:1:1
        train_sample_num = int(0.8 * data[2])
        val_sample_num = int(0.1 * data[2])
        test_sample_num = int(0.1 * data[2])
        
        # Limit the number of samples except for domain_net
        if train_sample_num > 8000 and 'domain_net' not in name:
            train_sample_num = 8000
            val_sample_num = 1000
            test_sample_num = 1000
        
        datafile_train = Cus_Dataset(mode='authorized_training_src',
                            # source domain
                            dataset=data,
                            begin_ind=val_sample_num + test_sample_num,
                            size=train_sample_num,
                            # others
                            is_img_path_aug=[False, False, False, False],
                            spec_dataTransform=data_transforms)
        datafile_val = Cus_Dataset(mode='val',
                            # source domain
                            dataset=data,
                            begin_ind=0,
                            size=val_sample_num,
                            # others
                            is_img_path_aug=[False, False, False, False],
                            spec_dataTransform=data_transforms)
        datafile_test = Cus_Dataset(mode='val',
                            # source domain
                            dataset=data,
                            begin_ind=val_sample_num,
                            size=test_sample_num,
                            # others
                            is_img_path_aug=[False, False, False, False],
                            spec_dataTransform=data_transforms)
        

        torch.save({'train': datafile_train, 'val': datafile_val, 'test': datafile_test}, 
                   f'data_presplit/{name}_{image_size}.pth')
        
        pass


if __name__ == '__main__':

    split()
    exit()

    # load:
    loaded = torch.load('data_presplit/mt_64.pth')
    loaded['train']
    loaded['test']
    pass
    


