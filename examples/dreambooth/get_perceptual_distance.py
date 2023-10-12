import cv2
import os
import glob
import torch
import torchvision
import argparse

import numpy as np

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from PIL import Image
from tqdm import tqdm
from pprint import pprint
from piq import FID, GS, IS, KID, MSID, psnr

# Need to run `pip install piq` for this
input_transforms = T.Compose(
    [
        T.Resize((1052, 1914)),
        # T.Normalize(
        #     mean = [0.485, 0.456, 0.406],
        #     std = [0.229, 0.224, 0.225]),
        T.ToTensor(),
    ]
)

def get_transform(size=(1052, 1914)):
    return T.Compose(
        [
            T.Resize(size),
            T.ToTensor(),
        ]
    )

def get_filelist(img_dir, hr_str = "/*", img_list=None, src_dataset=None, is_label=False):
    if src_dataset == "cityscapes" and is_label:
        exts = ["_gtFine_color.png"]
    else:
        exts = [".jpg", ".png"]
    for ext in exts:
        files = glob.glob(img_dir + hr_str + ext)
        if len(files) > 0:
            break
    # print(files)
    # print()
    if img_list:
        # print(img_list)
        if src_dataset == "cityscapes":
            files = [x for x in files if (x.split('/')[-1].replace("_gtFine_color.png", "_leftImg8bit.png") in img_list)]
        else:
            files = [x for x in files if (x.split('/')[-1] in img_list)]
    return files

def get_filelist_helper(img_dir, img_list=None, src_dataset=None, is_label=False):
    files = get_filelist(img_dir, "/*", img_list=img_list, src_dataset=src_dataset, is_label=is_label)
    if len(files) == 0:
        files = get_filelist(img_dir, "/*/*", img_list,src_dataset=src_dataset, is_label=is_label)  

    return files         

class ImgDataset(Dataset):
    def __init__(self, img_dir, transforms=None, img_list=None):
        exts = [".jpg", ".png"]
        files = get_filelist_helper(img_dir, img_list=img_list)
        # files = get_filelist(img_dir, "/*", img_list)
        # if len(files) == 0:
        #     files = get_filelist(img_dir, "/*/*", img_list)            
        self.imgs = files
        print("# Samples: ", len(self.imgs))
        self.transforms = transforms
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        # print(img.shape)
        return {"images": img}

def create_dataloader(img_dir, transforms=input_transforms, img_list=None):
    ImgDset = ImgDataset(img_dir, transforms,img_list=img_list)
    ImgDloader = DataLoader(ImgDset, batch_size=256, shuffle=False)
    return ImgDloader

def compute_FID(img_dir1, img_dir2, transforms=input_transforms):
    print("Creating Dataloaders...")
    dload1 = create_dataloader(img_dir1, transforms)
    dload2 = create_dataloader(img_dir2, transforms)
    print("Initializing Metrics...")
    fid = FID()
    print("Computing Features...")
    feat1 = fid.compute_feats(dload1)
    feat2 = fid.compute_feats(dload2)
    print("Computing Metric...")
    fid: torch.Tensor = fid(feat1, feat2)
    print(fid)

# Get Edge Image
# def get_edge_image(image):
#     image = np.array(image)
#     low_threshold = 100
#     high_threshold = 200
#     image = cv2.Canny(image, low_threshold, high_threshold)
#     # image = T.ToTensor()(image)
#     return image

def get_edge_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Laplacian(image, cv2.CV_64F, ksize=5)
    return image

def l1_diff(arr1, arr2):
    return np.mean(np.abs(arr1 - arr2))

# def psnr(arr1, arr2):
#     EPS = 1e-8
#     max1 = arr1.max()
#     max2 = arr2.max()
#     maxv = max(max1, max2)
#     mse = np.mean((arr1 - arr2) ** 2)
#     score = -10 * np.log10(mse + EPS)
#     return score

def psnr(arr1, arr2):
    EPS = 1e-8
    max1 = arr1.max()
    max2 = arr2.max()
    maxv = max(max1, max2)
    arr1 /= maxv
    arr2 /= maxv
    mse = np.mean((arr1 - arr2) ** 2)
    score = -10 * np.log10(mse + EPS)
    return score

def compute_condition_fidelity(src_dir, trans_dir, src_lbldir, size=(1052, 1914), src_list=None, src_dataset=None):
    img_transform = T.Compose(
        [
            T.Resize(size),
        ]
    )
    t_diff, s_diff = [], []

    # grabs image lists (if len is 0, check 1 mroe directory in)
    src_imgs = get_filelist_helper(src_dir,img_list=src_list)
       
    src_lbls = get_filelist_helper(src_lbldir, img_list=src_list, src_dataset=src_dataset, is_label=True)
    trans_imgs = get_filelist_helper(trans_dir,img_list=src_list)
    
    print("Computing fidelity...")
    for i in tqdm(range(len(src_imgs))):
        src_img = Image.open(src_imgs[i])
        src_imgid = src_imgs[i].split("/")[-1].split(".")[0]
        if "_leftImg8bit" in src_imgid:
            src_imgid = src_imgid.replace("_leftImg8bit", "")
        tr_img = Image.open([x for x in trans_imgs if src_imgid in x][0])
        lbl_img = Image.open([x for x in src_lbls if src_imgid in x and "_labelTrainIds" not in x][0]).convert("RGB")
        src_img = img_transform(src_img)
        tr_img = img_transform(tr_img)
        lbl_img = img_transform(lbl_img)
        src_img = get_edge_image(src_img)
        tr_img = get_edge_image(tr_img)
        lbl_img = get_edge_image(lbl_img)
        slbl_diff = psnr(src_img, lbl_img)
        trlbl_diff = psnr(tr_img, lbl_img)
        # print(slbl_diff)
        # print(trlbl_diff)
        # slbl_diff = psnr(src_img[None, None, :, :], lbl_img[None, None, :, :], data_range=1., reduction='none')
        # trlbl_diff = psnr(tr_img[None, None, :, :], lbl_img[None, None, :, :], data_range=1., reduction='none')
        s_diff.append(slbl_diff)
        t_diff.append(trlbl_diff)
    
    scores = [1 if t_i >= s_diff[i] else 0 for i,t_i in enumerate(t_diff)]
    
    print("Original-Label Edge Fidelity: ", np.mean(s_diff))
    print("Translated-Label Edge Fidelity: ", np.mean(t_diff))
    print("TOTAL SCORE (based on PSNR if trlbl_diff >= slbl_diff): ", np.sum(scores))
    print("AVG SCORE (based on PSNR if trlbl_diff >= slbl_diff): ", np.mean(scores))
    

def compute_translation_improvements(src_dir, tgt_dir, trans_dir, transforms=input_transforms, size=(1052, 1914), target_list=None, src_list=None):
    print("Creating Dataloaders...")
    src_dload = create_dataloader(src_dir, transforms,src_list)
    tgt_dload = create_dataloader(tgt_dir, transforms, target_list)
    trans_dload = create_dataloader(trans_dir, transforms, src_list)
    print("Initializing metrics...")
    fid = FID()
    kid = KID()
    print("Computing features...")
    src_feat = fid.compute_feats(src_dload, device="cuda:0")
    tgt_feat = fid.compute_feats(tgt_dload, device="cuda:0")
    trans_feat = fid.compute_feats(trans_dload, device="cuda:0")
    if torch.cuda.is_available():
        src_feat = src_feat.cuda()
        tgt_feat = tgt_feat.cuda()
        trans_feat = trans_feat.cuda()
    print("Computing metrics..")
    # FID
    src_tgt_fid: torch.Tensor = fid(src_feat, tgt_feat)
    trans_tgt_fid: torch.Tensor = fid(trans_feat, tgt_feat)
    # # KID
    # src_tgt_kid: torch.Tensor = kid(src_feat, tgt_feat)
    # trans_tgt_kid: torch.Tensor = kid(trans_feat, tgt_feat)
    mets = {
        "Src-Tgt FID": src_tgt_fid.item(),
        "Trans-Tgt FID": trans_tgt_fid.item(),
        # "Src-Tgt KID": src_tgt_kid.item(),
        # "Trans-Tgt KID": trans_tgt_kid.item(),
    }
    pprint(mets)

def create_source_symlink(imglist_txt, imgdir, symdir, dset="gtav"):
    if not os.path.exists(symdir):
        os.makedirs(symdir)
    imglist = [x.strip("\n") for x in open(imglist_txt, "r").readlines()]
    for img in imglist:
        source = os.path.join(imgdir, img)
        if dset != "gtav":
            destination = os.path.join(symdir, img.split("/")[-1])
        else:
            destination = os.path.join(symdir, img)
        os.symlink(source, destination)
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perceptual Distance Metrics")
    parser.add_argument(
        "--target_dir",
        required=True,
        type=str,
        default=None,
        help="dir of target img",
    )
    parser.add_argument(
        "--target_img_list",
        required=False,
        type=str,
        default=None,
        help="target img list",
    )
    parser.add_argument(
        "--trans_dir",
        required=True,
        type=str,
        default=None,
        help="dir of translated source img in target style",
    )
    parser.add_argument(
        "--src_dir",
        required=True,
        type=str,
        default=None,
        help="dir of source img",
    )
    parser.add_argument(
        "--src_img_list",
        required=False,
        type=str,
        default=None,
        help="src img list",
    )
    parser.add_argument(
        "--src_label_dir",
        required=True,
        type=str,
        default=None,
        help="dir of source labelsdata",
    )
    parser.add_argument(
        "--src_dataset",
        required=False,
        type=str,
        default=None,
        help="src dataset name",
    )
    parser.add_argument(
        "--size",
        required=False,
        nargs='+',
        type=int,
        default=None,
        help="size of src img",
    )
    args, _ = parser.parse_known_args()

    # # Create GTAV Source Symlinks
    # IMGDIR = "/srv/share4/datasets/GTA5DA/images"
    # SYMDIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/generation_img_splits/source_dirs/gtav_gen_source"
    # IMGLIST = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/generation_img_splits/gta_img_split_700.txt"
    # create_source_symlink(IMGLIST, IMGDIR, SYMDIR)
    
    # # Create Cityscapes Source Symlinks
    # IMGDIR = "/srv/share4/datasets/cityscapesDA/leftImg8bit/train"
    # SYMDIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/generation_img_splits/source_dirs/cityscapes_gen_source"
    # IMGLIST = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/generation_img_splits/cityscapes_img_split_700.txt"
    # create_source_symlink(IMGLIST, IMGDIR, SYMDIR, "cityscapes")
    
    # Sim2Real (GTAV to Cityscapes)
    TGT_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/sd_datasets/sd_v3_dataset_city_n500_rand1234"
    TRANS_DIR = "/srv/share4/vvijaykumar6/diffusion-da/datasets/testing_v6_rml_sd_v15_pp_city_n500_rand1234/checkpoint_2k/gta_cs_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
    SRC_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/generation_img_splits/source_dirs/gtav_gen_source"
    SRC_LBL_DIR = "/srv/share4/datasets/GTA5DA/labels"

    # compute_translation_improvements(SRC_DIR, TGT_DIR, TRANS_DIR)
    # compute_condition_fidelity(SRC_DIR, TRANS_DIR, SRC_LBL_DIR)

    # grab from args
    TGT_DIR = args.target_dir
    TRANS_DIR = args.trans_dir
    SRC_DIR = args.src_dir
    SRC_LBL_DIR = args.src_label_dir


    with open(args.target_img_list, 'r') as file:
        target_img_list = [line.strip().split('/')[-1] for line in file.readlines()]
    
    with open(args.src_img_list, 'r') as file2:
        src_img_list = [line.strip().split('/')[-1] for line in file2.readlines()]

    if args.size:
        size = tuple(args.size)
    else:
        size = (1052, 1914)
    
    print("Target dir", TGT_DIR)
    print("Target img list", args.target_img_list)
    print("Translated src dir", TRANS_DIR)
    print("Source dir", SRC_DIR)
    print("source img list", args.src_img_list)
    print("Soruce label dir", SRC_LBL_DIR)
    print("Img Size", size)
    
    compute_translation_improvements(SRC_DIR, TGT_DIR, TRANS_DIR, target_list=target_img_list, src_list=src_img_list, size=size)
    compute_condition_fidelity(SRC_DIR, TRANS_DIR, SRC_LBL_DIR, src_list=src_img_list, src_dataset=args.src_dataset, size=size)
    
    
    # # Real2Real (Cityscapes to ACDC-Night)
    # TGT_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/sd_datasets/sd_v3_dataset_acdc_tr_night_n400_rand1234"
    # TRANS_DIR = "/srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_night_n400_res_512_crop_512_rand1234/checkpoint_2k/cityscapes_acdc_tr_night_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
    # SRC_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/generation_img_splits/source_dirs/cityscapes_gen_source"
    # compute_translation_improvements(SRC_DIR, TGT_DIR, TRANS_DIR, size=(1024, 2048))
    
    # # Real2Real (Cityscapes to ACDC-Snow)
    # TGT_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/sd_datasets/sd_v3_dataset_acdc_tr_snow_n400_rand1234"
    # TRANS_DIR = "/srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_snow_n400_res_512_crop_512_rand1234/checkpoint_2k/cityscapes_acdc_tr_snow_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
    # SRC_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/generation_img_splits/source_dirs/cityscapes_gen_source"
    # compute_translation_improvements(SRC_DIR, TGT_DIR, TRANS_DIR, size=(1024, 2048))
    
    # # Real2Real (Cityscapes to ACDC-Fog)
    # TGT_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/sd_datasets/sd_v3_dataset_acdc_tr_fog_n400_rand1234"
    # TRANS_DIR = "/srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_fog_n400_res_512_crop_512_rand1234/checkpoint_2k/cityscapes_acdc_tr_fog_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
    # SRC_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/generation_img_splits/source_dirs/cityscapes_gen_source"
    # compute_translation_improvements(SRC_DIR, TGT_DIR, TRANS_DIR, size=(1024, 2048))
    
    # # Real2Real (Cityscapes to ACDC-Rain)
    # TGT_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/sd_datasets/sd_v3_dataset_acdc_tr_rain_n400_rand1234"
    # TRANS_DIR = "/srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_rain_n400_res_512_crop_512_rand1234/checkpoint_2k/cityscapes_acdc_tr_rain_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
    # SRC_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/generation_img_splits/source_dirs/cityscapes_gen_source"
    # compute_translation_improvements(SRC_DIR, TGT_DIR, TRANS_DIR, size=(1024, 2048))
    
    # # Real2Real (Cityscapes to ACDC-All)
    # TGT_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/sd_datasets/sd_v3_dataset_acdc_tr_all_n500_rand1234"
    # TRANS_DIR = "/srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_all_n500_res_512_crop_512_rand1234/checkpoint_2k/cityscapes_acdc_tr_all_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
    # SRC_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/generation_img_splits/source_dirs/cityscapes_gen_source"
    # compute_translation_improvements(SRC_DIR, TGT_DIR, TRANS_DIR, size=(1024, 2048))
    