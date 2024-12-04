# Inpaint masked area
import os, sys
from glob import glob
from shutil import copyfile
import yaml
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import pillow_avif
from functools import reduce
from operator import add
import torch.nn as nn
import torch
from torch.utils.data._utils.collate import default_collate

import easyocr
from easyocr.detection import get_detector, get_textbox
import cv2

#from trocr.src.main import TrocrPredictor
from utils import *

from omegaconf import OmegaConf
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from saicinpainting.training.data.datasets import make_default_val_dataset, get_transforms
from saicinpainting.evaluation.data import InpaintingDataset as InpaintingEvaluationDataset #
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Note: English is compatible with all languages. Languages that share most of character (e.g. latin script) with each other are compatible.
# this needs to run only once to load the model into memory
reader_EF = easyocr.Reader(['en', 'fr'])
#reader_CN = easyocr.Reader(['cn_sim', 'cn_tra'])
#reader_JP = easyocr.Reader(['ja', 'en'])
#reader_KR = easyocr.Reader(['ko', 'en'])
#reader_DE = easyocr.Reader(['en', 'de']) # German
#reader_IT = easyocr.Reader(['en', 'it']) # Italian
#reader_ES = easyocr.Reader(['en', 'es']) # Spanish
#reader_PT = easyocr.Reader(['en', 'pt']) # Portuguese

def text_removal(input_dir, output_dir, config_path="configs/prediction/default.yaml", **kw_args):
    ## processing start
    image_filename_list = glob(input_dir+'/**/**/*.*')
    images_path = [os.path.join(input_dir, file_path) for file_path in image_filename_list]
    print(f"Total number of images: {len(images_path)}")

    ## setting up model params
    with open(config_path, 'r') as f:
        predict_config = OmegaConf.create(yaml.safe_load(f))
    model_path = os.path.join(os.getcwd(), 'big-lama')
    predict_config.model.path = model_path
    predict_config.indir = os.path.dirname(images_path[0])
    predict_config.outdir = output_dir + "/Inpainted"
    predict_config.refiner.gpu_ids = '0'

    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path,
                                       'models',
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

    ## img processing units
    print(predict_config.dataset) # {'kind': 'default', 'img_suffix': '.png', 'pad_out_to_modulo': 8}
    transform = get_transforms(transform_variant=predict_config.dataset.kind, out_size=512)
    #mask_generator = get_mask_generator(kind=None, kwargs=None) # None train_config.model.mask_schedule
    # print(mask_generator) ## saicinpainting.training.data.masks.MixedMaskGenerator object

    for img_path, img_name in tqdm(zip(images_path, image_filename_list)):
        iteration = 0
        if img_name == img_path:
            img_name = os.path.basename(img_name)
        if 'avif' in img_name:
            img = Image.open(img_path)
            img_path = img_path.replace('avif', 'png')
            img.save(img_path)
            img_name = img_name.replace('avif', 'png')
            del img
        print(img_path, img_name)
        # text detection
        det_results = reader_EF.readtext(img_path, **kw_args)
        img = cv2.imread(img_path)
        img_ = img.copy()
        row, col, channel = img.shape
        img_mask = np.zeros((row, col), dtype=np.uint8)
        radius = 0

        for (bbox, text, prob) in det_results:
            startPoint, endPoint = np.array(bbox[0], dtype=np.int64), np.array(bbox[2], dtype=np.int64)
            points = reduce(add, bbox)
            xy1 = (max(int(points[0]) - radius, 0), max(int(points[1]) - radius, 0))
            xy2 = (min(int(points[2]) + radius, col-1), max(int(points[3]) - radius, 0))
            xy3 = (min(int(points[4]) + radius, col-1), min(int(points[5]) + radius, row-1))
            xy4 = (max(int(points[6]) - radius, 0), min(int(points[7]) + radius, row-1))
            points = np.array([xy1, xy2, xy3, xy4])

            # draw mask on img
            img_mask = drawMask(points, img_mask)

        #img_mask = np.stack((img_mask,)*3, axis=-1)
        #mask = (img_mask[:,:,0]==1)*(img_mask[:,:,1]==0)*(img_mask[:,:,2]==0)

        #plt.axis("on")
        #plt.imshow(cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB))
        #plt.title(f"Mask of {img_name.split('.')[0]}")
        #plt.show()

        mask = cv2.bitwise_not(img_mask) == 0 # reversed mask
        plt.imsave(os.path.join(output_dir, 'data_for_prediction', f"{os.path.dirname(img_path).split('/')[-1]}_{img_name.split('.')[1]}_mask.png"), mask, cmap='gray')
        copyfile(img_path, os.path.join(output_dir, 'data_for_prediction', os.path.dirname(img_path).split('/')[-1]+'_'+img_name))

        # masked image
        img_inpaint = np.array((1-mask.reshape(mask.shape[0], mask.shape[1], -1))*img[:,:,:3]).astype(np.uint8)
        #plt.axis("on")
        #plt.imshow(cv2.cvtColor(img_inpaint, cv2.COLOR_BGR2RGB))
        #plt.title(f"Masked {img_name.split('.')[0]}")
        #plt.show()
        plt.imsave(os.path.join(output_dir, 'Masked', img_name), cv2.cvtColor(img_inpaint, cv2.COLOR_BGR2RGB))

        # construct in model input format dataset = InpaintingEvaluationDataset(indir, **kwargs)
        predict_config.dataset.img_suffix = img_path.split('.')[-1] # TODO

        img = np.array(Image.open(img_path).convert('RGB')) # default trans
        img = np.transpose(img, (2, 0, 1)) # mode = 'RGB'
        img = img.astype('float32') / 255
        img_mask = img_mask.astype('float32') / 255  # mode = 'L'
        img_batch = [{'image': img, 'mask': img_mask[None, ...], 'unpad_to_size': (row, col)}]

        img_batch = default_collate([img_batch[0]])
        img_batch = move_to_device(img_batch, device)

        cur_res = refine_predict(img_batch, model, **predict_config.refiner)
        cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, 'Inpainted', os.path.dirname(img_path).split('/')[-1]+'_'+img_name), cur_res)

        res_tile = hconcat_resize_min([img_, img_inpaint, cur_res])
        plt.imsave(os.path.join(output_dir, 'tiled', os.path.dirname(img_path).split('/')[-1]+'_'+img_name), cv2.cvtColor(res_tile, cv2.COLOR_BGR2RGB))
        


def text_style_recog_removal(input_dir, output_dir, max_radius=10, lama_config="/content/LAMA/configs/prediction/default.yaml", **kw_args):
    ## processing start
    image_filename_list = glob(input_dir+'/**/**/*.*')[:20]
    images_path = [os.path.join(input_dir, file_path) for file_path in image_filename_list]
    print(f"Total number of images: {len(images_path)}")

    ## setting up LAMA params
    with open(lama_config, 'r') as f:
        predict_config = OmegaConf.create(yaml.safe_load(f))
    model_path = os.path.join(os.getcwd(), 'big-lama')
    predict_config.model.path = model_path
    predict_config.indir = os.path.dirname(images_path[0])
    predict_config.outdir = output_dir + "/Inpainted"
    predict_config.refiner.gpu_ids = '0'

    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path,
                                       'models',
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config,
                                checkpoint_path,
                                strict=False,
                                map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

    ## img processing units
    print(predict_config.dataset) # {'kind': 'default', 'img_suffix': '.png', 'pad_out_to_modulo': 8

    for img_path, img_name in tqdm(zip(images_path, image_filename_list)):
        #if 'SKINCEUTICALS-48' not in img_name:
        #    continue
        if img_name == img_path:
            img_name = os.path.basename(img_name)
        if 'avif' in img_name:
            img = Image.open(img_path)
            img_path = img_path.replace('avif', 'png')
            img.save(img_path)
            img_name = img_name.replace('avif', 'png')
            del img
        print(img_path, img_name)
        # text detection
        det_results = reader_EF.readtext(img_path, **kw_args)
        #rec_results = reader_EF.recognize(img_path, det_results)

        img = cv2.imread(img_path)
        img_ = img.copy()
        row, col, channel = img.shape
        img_mask = np.zeros((row, col), dtype=np.uint8)

        for (bbox, text, prob) in det_results:
        #for results in det_results:
        #    bbox, text = results
            points = reduce(add, bbox)

            # text color style
            colors, is_bg_color, edge_wid = style_detect(img, points)
            print(img_name, text, colors)


            if colors is None:
                radius = 0
            elif is_bg_color:
                radius = min(max_radius, edge_wid)
            else:
                radius = 5
            xy1 = (max(int(points[0]) - radius, 0), max(int(points[1]) - radius, 0))
            xy2 = (min(int(points[2]) + radius, col-1), max(int(points[3]) - radius, 0))
            xy3 = (min(int(points[4]) + radius, col-1), min(int(points[5]) + radius, row-1))
            xy4 = (max(int(points[6]) - radius, 0), min(int(points[7]) + radius, row-1))
            points = np.array([xy1, xy2, xy3, xy4])
            # draw mask on img
            img_mask = drawMask(points, img_mask)

        #img_mask = np.stack((img_mask,)*3, axis=-1)
        #mask = (img_mask[:,:,0]==1)*(img_mask[:,:,1]==0)*(img_mask[:,:,2]==0)

        #plt.axis("on")
        #plt.imshow(cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB))
        #plt.title(f"Mask of {img_name.split('.')[0]}")
        #plt.show()

        mask = cv2.bitwise_not(img_mask) == 0 # reversed mask
        plt.imsave(os.path.join(output_dir, 'data_for_prediction', f"{os.path.dirname(img_path).split('/')[-1]}_{img_name.split('.')[1]}_mask.png"), mask, cmap='gray')
        copyfile(img_path, os.path.join(output_dir, 'data_for_prediction', os.path.dirname(img_path).split('/')[-1]+'_'+img_name))

        # masked image
        img_inpaint = np.array((1-mask.reshape(mask.shape[0], mask.shape[1], -1))*img[:,:,:3]).astype(np.uint8)
        #plt.axis("on")
        #plt.imshow(cv2.cvtColor(img_inpaint, cv2.COLOR_BGR2RGB))
        #plt.title(f"Masked {img_name.split('.')[0]}")
        #plt.show()
        plt.imsave(os.path.join(output_dir, 'Masked', img_name), cv2.cvtColor(img_inpaint, cv2.COLOR_BGR2RGB))

        # construct in model input format dataset = InpaintingEvaluationDataset(indir, **kwargs)
        predict_config.dataset.img_suffix = img_path.split('.')[-1] # TODO

        img = np.array(Image.open(img_path).convert('RGB')) # default trans
        img = np.transpose(img, (2, 0, 1)) # mode = 'RGB'
        img = img.astype('float32') / 255
        img_mask = img_mask.astype('float32') / 255  # mode = 'L'
        img_batch = [{'image': img, 'mask': img_mask[None, ...], 'unpad_to_size': (row, col)}]

        img_batch = default_collate([img_batch[0]])
        img_batch = move_to_device(img_batch, device)

        cur_res = refine_predict(img_batch, model, **predict_config.refiner)
        cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, 'Inpainted', os.path.dirname(img_path).split('/')[-1]+'_'+img_name), cur_res)

        res_tile = hconcat_resize_min([img_, img_inpaint, cur_res])
        plt.imsave(os.path.join(output_dir, 'tiled', os.path.dirname(img_path).split('/')[-1]+'_'+img_name), cv2.cvtColor(res_tile, cv2.COLOR_BGR2RGB))
        
