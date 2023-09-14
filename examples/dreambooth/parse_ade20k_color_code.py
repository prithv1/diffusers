import os
import re
import sys
import csv
import json
import copy

import numpy as np
import pandas as pd

from tqdm import tqdm
from pprint import pprint

# ADE20K Map from here - https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8/edit#gid=0
 
# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#     Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#     Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#     Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#     Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#     Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#     Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#     Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#     Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#     Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#     Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#     Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#     Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#     Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#     Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#     Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#     Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#     Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#     Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#     Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
# ]

CITYSCAPES_NAME2COLOR = {
    'road'         : (128, 64,128),
    'sidewalk'     : (244, 35,232),
    'building'     : ( 70, 70, 70),
    'wall'         : (102,102,156),
    'fence'        : (190,153,153),
    'pole'         : (153,153,153),
    'traffic light': (250,170, 30), # map to stoplight
    'traffic sign' : (220,220,  0), # map to signal
    'vegetation'   : (107,142, 35),
    'terrain'      : (152,251,152), # map to land
    'sky'          : ( 70,130,180),
    'person'       : (220, 20, 60),
    'rider'        : (255,  0,  0), # map to person
    'car'          : (  0,  0,142),
    'truck'        : (  0,  0, 70), 
    'bus'          : (  0, 60,100),
    'train'        : (  0, 80,100), # map to rail
    'motorcycle'   : (  0,  0,230), # map to motorbike
    'bicycle'      : (119, 11, 32),
}


def load_csv():
    ret_data = []
    df = pd.read_csv("ade20k_color_coding.csv")
    data = list(df.T.to_dict().values())
    for i in tqdm(range(len(data))):
        d = data[i]
        if ";" in d["Name"]:
            names = d["Name"].split(";")
            for name in names:
                temp_d = copy.deepcopy(d)
                temp_d["Name"] = name
                ret_data.append(temp_d)
        else:
            ret_data.append(d)
    n_df = pd.DataFrame(ret_data)
    return n_df

def make_dataset_specific_json(dset="cityscapes"):
    ade_df = load_csv()
    if dset == "cityscapes":
        # custom cityscapes name map
        change_keys = {
            "traffic light": "stoplight",
            "traffic sign": "signal",
            "terrain": "land",
            "rider": "individual",
            "train": "coach",
            "motorcycle": "motorbike",
            "vegetation": "flora",
        }
        use_map = copy.deepcopy(CITYSCAPES_NAME2COLOR)
        for k in change_keys.keys():
            use_map[change_keys[k]] = use_map[k]
            del use_map[k]
        rev_map = {v:k for k,v in use_map.items()}
        # pprint(rev_map)
        translate_map = {}
        for city_color in rev_map.keys():
            city_cls = rev_map[city_color]
            # print(city_cls)
            # print(city_color)
            # Get color from ade
            sub_dct = list(ade_df[ade_df["Name"] == city_cls].T.to_dict().values())[0]
            ade_color = sub_dct["Color_Code (R,G,B)"]
            ade_color = tuple([int(x) for x in re.findall(r'\d+', ade_color)])
            print(city_cls, city_color, ade_color)
            translate_map[str(city_color)] = ade_color
        print("*"*20)
        pprint(translate_map)
        with open("city2ade_translate_map.json", "w") as f:
            json.dump(translate_map, f)
    else:
        print("Not supported yet")

    # import pdb; pdb.set_trace()
    
if __name__ == "__main__":
    # load_csv()
    make_dataset_specific_json()