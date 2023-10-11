import os
import glob
import torch
import torchvision

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from PIL import Image
from tqdm import tqdm
from pprint import pprint
from piq import FID, GS, IS, KID, MSID

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

class ImgDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        exts = [".jpg", ".png"]
        for ext in exts:
            files = glob.glob(img_dir + "/*" + ext)
            if len(files) > 0:
                break
        self.imgs = files
        print("#Samples: ", len(self.imgs))
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
    
    
    # IMG_DIR1 = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/sd_datasets/sd_v3_dataset_city_n500_rand1234"
    # # IMG_DIR2 = "/srv/share4/vvijaykumar6/diffusion-da/datasets/testing_v6_rml_sd_v15_pp_city_n500_rand1234/checkpoint_2k/gta_cs_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
    # IMG_DIR2 = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/generation_img_splits/source_dirs/gtav_gen_source"
    # compute_FID(IMG_DIR1, IMG_DIR2)
    
    # Sim2Real
    TGT_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/sd_datasets/sd_v3_dataset_city_n500_rand1234"
    TRANS_DIR = "/srv/share4/vvijaykumar6/diffusion-da/datasets/testing_v6_rml_sd_v15_pp_city_n500_rand1234/checkpoint_2k/gta_cs_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
    SRC_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/generation_img_splits/source_dirs/gtav_gen_source"
    compute_translation_improvements(SRC_DIR, TGT_DIR, TRANS_DIR)
    
    # # Real2Real
    # TGT_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/sd_datasets/sd_v3_dataset_acdc_tr_night_n400_rand1234"
    # TRANS_DIR = "/srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_night_n400_res_512_crop_512_rand1234/checkpoint_2k/cityscapes_acdc_tr_night_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
    # SRC_DIR = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/generation_img_splits/source_dirs/cityscapes_gen_source"
    # compute_translation_improvements(SRC_DIR, TGT_DIR, TRANS_DIR, (1024, 2048))
    