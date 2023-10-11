import cv2
import os
import glob
import torch
import torchvision

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

def get_filelist(img_dir, hr_str = "/*"):
    exts = [".jpg", ".png"]
    for ext in exts:
        files = glob.glob(img_dir + hr_str + ext)
        if len(files) > 0:
            break
    return files

class ImgDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        exts = [".jpg", ".png"]
        files = get_filelist(img_dir, "/*")
        if len(files) == 0:
            files = get_filelist(img_dir, "/*/*")            
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

def create_dataloader(img_dir, transforms=input_transforms):
    ImgDset = ImgDataset(img_dir, transforms)
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
def get_edge_image(image):
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    # image = T.ToTensor()(image)
    return image

def l1_diff(arr1, arr2):
    return np.mean(np.abs(arr1 - arr2))

def psnr(arr1, arr2):
    EPS = 1e-8
    max1 = arr1.max()
    max2 = arr2.max()
    maxv = max(max1, max2)
    mse = np.mean((arr1 - arr2) ** 2)
    score = -10 * np.log10(mse + EPS)
    return score

def compute_condition_fidelity(src_dir, trans_dir, src_lbldir, size=(1052, 1914)):
    img_transform = T.Compose(
        [
            T.Resize(size),
        ]
    )
    t_diff, s_diff = [], []
    src_imgs = get_filelist(src_dir)
    src_lbls = get_filelist(src_lbldir)
    trans_imgs = get_filelist(trans_dir)
    print("Computing fidelity...")
    for i in tqdm(range(len(src_imgs))):
        src_img = Image.open(src_imgs[i])
        src_imgid = src_imgs[i].split("/")[-1].split(".")[0]
        if "_leftImg8bit" in src_imgid:
            src_imgid = src_imgid.replace("_leftImg8bit", "")
        tr_img = Image.open([x for x in trans_imgs if src_imgid in x][0])
        lbl_img = Image.open([x for x in src_lbls if src_imgid in x and "_labelTrainIds" not in x][0])
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
    
    print("Original-Label Edge Fidelity: ", np.mean(s_diff))
    print("Translated-Label Edge Fidelity: ", np.mean(t_diff))
    

def compute_translation_improvements(src_dir, tgt_dir, trans_dir, transforms=input_transforms, size=(1052, 1914)):
    print("Creating Dataloaders...")
    src_dload = create_dataloader(src_dir, transforms)
    tgt_dload = create_dataloader(tgt_dir, transforms)
    trans_dload = create_dataloader(trans_dir, transforms)
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
    compute_condition_fidelity(SRC_DIR, TRANS_DIR, SRC_LBL_DIR)
    
    
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
    