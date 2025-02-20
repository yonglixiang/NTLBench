from collections import defaultdict
from torchvision.datasets import MNIST
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import argparse
import tarfile
import shutil
import gdown
import uuid
import json
import os
import urllib
# import kaggle

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset


# utils
def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path

def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith((".tar.gz", ".tgz")):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)

# VLCS
def download_vlcs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "VLCS")

    download_and_extract("https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8",
                         os.path.join(data_dir, "VLCS.tar.gz"))

# MNIST: you can download when get, no need to download seperately
def download_mnist(data_dir):
    # Original URL: http://yann.lecun.com/exdb/mnist/
    full_path = stage_path(data_dir, "MNIST")
    MNIST(full_path, download=True)

# PACS
def download_pacs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "PACS")

    download_and_extract("https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
                         os.path.join(data_dir, "PACS.zip"))

    os.rename(os.path.join(data_dir, "kfold"),
              full_path)

# Office-Home
def download_office_home(data_dir):
    # Original URL: http://hemanthdv.org/OfficeHome-Dataset/
    full_path = stage_path(data_dir, "office_home")

    download_and_extract("https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC",
                         os.path.join(data_dir, "office_home.zip"))

    os.rename(os.path.join(data_dir, "OfficeHomeDataset_10072016"),
              full_path)

# DomainNET
def download_domain_net(data_dir):
    # Original URL: http://ai.bu.edu/M3SDA/
    full_path = stage_path(data_dir, "domain_net")

    urls = [
        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
    ]

    for url in urls:
        download_and_extract(url, os.path.join(full_path, url.split("/")[-1]))

# TerraIncognita
def download_terra_incognita(data_dir):
    # Original URL: https://beerys.github.io/CaltechCameraTraps/
    # New URL: http://lila.science/datasets/caltech-camera-traps

    full_path = stage_path(data_dir, "terra_incognita")

    download_and_extract(
        "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz",
        os.path.join(full_path, "terra_incognita_images.tar.gz"))


    download_and_extract(
        "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz",
        os.path.join(full_path, "eccv_18_annotations.tar.gz"))


    include_locations = ["38", "46", "100", "43"]

    include_categories = [
        "bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit",
        "raccoon", "squirrel"
    ]

    images_folder = os.path.join(full_path, "eccv_18_all_images_sm/")
    annotations_folder = os.path.join(full_path,"eccv_18_annotation_files/")
    cis_test_annotations_file = os.path.join(full_path, "eccv_18_annotation_files/cis_test_annotations.json")
    cis_val_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/cis_val_annotations.json")
    train_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/train_annotations.json")
    trans_test_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/trans_test_annotations.json")
    trans_val_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/trans_val_annotations.json")
    annotations_file_list = [cis_test_annotations_file, cis_val_annotations_file, train_annotations_file, trans_test_annotations_file, trans_val_annotations_file]
    destination_folder = full_path

    stats = {}
    data = defaultdict(list)

    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)

    for annotations_file in annotations_file_list:
        annots = {}
        with open(annotations_file, "r") as f:
            annots = json.load(f)
            for k, v in annots.items():
                data[k].extend(v)



    category_dict = {}
    for item in data['categories']:
        category_dict[item['id']] = item['name']

    for image in data['images']:
        image_location = str(image['location'])

        if image_location not in include_locations:
            continue

        loc_folder = os.path.join(destination_folder,
                                  'location_' + str(image_location) + '/')

        if not os.path.exists(loc_folder):
            os.mkdir(loc_folder)

        image_id = image['id']
        image_fname = image['file_name']

        for annotation in data['annotations']:
            if annotation['image_id'] == image_id:
                if image_location not in stats:
                    stats[image_location] = {}

                category = category_dict[annotation['category_id']]

                if category not in include_categories:
                    continue

                if category not in stats[image_location]:
                    stats[image_location][category] = 0
                else:
                    stats[image_location][category] += 1

                loc_cat_folder = os.path.join(loc_folder, category + '/')

                if not os.path.exists(loc_cat_folder):
                    os.mkdir(loc_cat_folder)

                dst_path = os.path.join(loc_cat_folder, image_fname)
                src_path = os.path.join(images_folder, image_fname)

                shutil.copyfile(src_path, dst_path)

    shutil.rmtree(images_folder)
    shutil.rmtree(annotations_folder)

# SVIRO
def download_sviro(data_dir):
    # Original URL: https://sviro.kl.dfki.de
    full_path = stage_path(data_dir, "sviro")

    download_and_extract("https://sviro.kl.dfki.de/?wpdmdl=1731",
                         os.path.join(data_dir, "sviro_grayscale_rectangle_classification.zip"))

    os.rename(os.path.join(data_dir, "SVIRO_DOMAINBED"),
              full_path)

# SPAWRIOUS
def download_spawrious(data_dir, remove=True):
    dst = os.path.join(data_dir, "spawrious.tar.gz")
    urllib.request.urlretrieve('https://www.dropbox.com/s/e40j553480h3f3s/spawrious224.tar.gz?dl=1', dst)
    tar = tarfile.open(dst, "r:gz")
    tar.extractall(os.path.dirname(dst))
    tar.close()
    if remove:
        os.remove(dst)

# synthetic_digits
def download_synthetic_digits(data_dir, remove=True):
    # Initialize the Kaggle API
    kaggle.api.authenticate()

    # Download the dataset
    kaggle.api.dataset_download_files("prasunroy/synthetic-digits", path=data_dir, unzip=True)

    # Remove the directory
    if os.path.exists(os.path.join(data_dir, 'data/synthetic_digits')):
        import shutil
        shutil.rmtree(os.path.join(data_dir, 'data'))

    # Remove the zip file
    if os.path.exists(os.path.join('synthetic-digits.zip')):
        os.remove(os.path.join('synthetic-digits.zip'))

# CIFAR10 
def download_cifar(data_dir, remove=True):
    download_and_extract("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                         os.path.join(data_dir, "cifar-10-python.tar.gz"))
    
# STL10
def download_stl(data_dir, remove=True):
    download_and_extract("http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
                         os.path.join(data_dir, "stl10_binary.tar.gz"))
    
# VisDA Train and Validation
def download_visda(data_dir, remove=True):
    full_path = stage_path(data_dir, "VisDA")

    download_and_extract("http://csr.bu.edu/ftp/visda17/clf/train.tar",
                         os.path.join(data_dir, "VisDA", "train.tar"))
    os.remove(os.path.join(full_path, 'train', 'image_list.txt'))
    download_and_extract("http://csr.bu.edu/ftp/visda17/clf/validation.tar",
                         os.path.join(data_dir, "VisDA", "validation.tar"))
    os.remove(os.path.join(full_path, 'validation', 'image_list.txt'))

# Imagenette
def download_imagenette(data_dir, remove=True):
    download_and_extract("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
                         os.path.join(data_dir, "imagenette2.tgz"))
    
# mnist-m
def download_mnist_m(data_dir, remove=True):
    download_and_extract("https://drive.google.com/uc?export=download&id=0B_tExHiYS-0veklUZHFYT19KYjg",
                         os.path.join(data_dir, "mnist_m.tar.gz"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    # Domainbed task
    # download_mnist(args.data_dir)  # download when get, no need to download seperately
    # download_office_home(args.data_dir)
    # download_domain_net(args.data_dir)
    # download_vlcs(args.data_dir)
    # download_pacs(args.data_dir)
    # download_terra_incognita(args.data_dir)
    # download_spawrious(args.data_dir)
    # download_sviro(args.data_dir)
    # Camelyon17Dataset(root_dir=args.data_dir, download=True)
    # FMoWDataset(root_dir=args.data_dir, download=True)
    
    # NTL task: digits(mm+sn), cifar, stl, visda
    # download_mnist_m(args.data_dir)
    # download_synthetic_digits(args.data_dir)
    download_cifar(args.data_dir)
    download_stl(args.data_dir)
    # download_visda(args.data_dir)
    
    # SOPHON task: apply imagenette as source domain
    # download_imagenette(args.data_dir)

