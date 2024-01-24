import argparse
import logging
import os

import numpy as np
import torch
import cv2
from PIL import Image
from skimage.color import label2rgb
import matplotlib.pyplot as plt

from math import ceil
from openslide import OpenSlide
from models import UNet, NuClick_NN
from config import DefaultConfig
from utils.process import post_processing, gen_instance_map
from utils.misc import get_coords_from_csv, get_clickmap_boundingbox, get_output_filename, get_my_images_points
from utils.guiding_signals import get_patches_and_signals

# ratio = 2
svs_path = '**/IHC'
def load_image(fn):
    patch_str = '{}/{}'.format(svs_path, fn)
    slide_path, h, w, level, ps = patch_str.split('&')
    h = int(h); w = int(w)
    level = int(level); ps = int(ps)
    
    with OpenSlide(slide_path) as slide:
        im = slide.read_region((w, h), level, (ps, ps)).convert('RGB')
        
    return im

def predict_img(net,
                full_img,
                device,
                points_csv,
                scale_factor=1,
                out_threshold=0.5,
                bs=128):
    net.eval()

    
    cx, cy = get_coords_from_csv(points_csv)
    imgWidth = full_img.width
    imgHeight = full_img.height

    clickMap, boundingBoxes = get_clickmap_boundingbox(cx, cy, imgHeight, imgWidth)

    image = np.asarray(full_img)[:, :, :3]
    image = np.moveaxis(image, 2, 0)

    patchs, nucPoints, otherPoints = get_patches_and_signals(image, clickMap, boundingBoxes, cx, cy, imgHeight, imgWidth)
    patchs = patchs / 255

    input = np.concatenate((patchs, nucPoints, otherPoints), axis=1, dtype=np.float32)
    input = torch.from_numpy(input)
    input = input.to(device=device, dtype=torch.float32)
    
    n_batches = ceil(len(input) / bs)
    preds = []
    
    with torch.no_grad():
        for i in range(n_batches):
            s = i * bs
            e = (i + 1) * bs if i != n_batches-1 else len(input)
            batch = input[s:e]
            output = net(batch) #(no.patchs, 5, 128, 128)
            output = torch.sigmoid(output)
            output = torch.squeeze(output, 1)   #(no.patchs, 128, 128)
            pred = output.cpu().numpy()
            preds += [pred]
    
    preds = np.concatenate(preds, axis=0)
    masks = post_processing(preds, thresh=out_threshold, minSize=10, minHole=30, doReconstruction=True, nucPoints=nucPoints)
    
    #Generate instanceMap
    instanceMap = gen_instance_map(masks, boundingBoxes, imgHeight, imgWidth)
    return instanceMap


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')

    parser.add_argument('--model', '-m', metavar='NAME', default='unet', help='Name of the model')
    parser.add_argument('--pretrained_weights', '-w', metavar='PATH', default='./ckpt/NuClick_UNet_40xAll.pth',
                        help='Path to the pretrained weights')
    
    parser.add_argument('--image', '-i', metavar='PATH', nargs='+', help='Path to the input images')
    parser.add_argument('-imgdir', metavar='PATH', default=None, help='Path to the directory containing input images')

    parser.add_argument('--points', '-p', metavar='PATH', nargs='+', help='Path to the CSV files containing points')
    parser.add_argument('-pntdir', metavar='PATH', default='../../Step2_CellDetection/dataset/infer_df_pred',
                             help='Path to the directory containing the CSV files')
    
    parser.add_argument('--output', '-o', metavar='PATH', default='../../Step3_CellSegmentation/dataset/preds', help='Directory where the instance maps will be saved into')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=DefaultConfig.mask_thresh,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=DefaultConfig.img_scale,
                        help='Scale factor for the input images')
    parser.add_argument('--gpu', '-g', metavar='GPU', default=None, help='ID of GPUs to use (based on `nvidia-smi`)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    # setting gpus
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(args.gpu)
        print(type(args.gpu))
    
    # getting images and points
    images_points = get_my_images_points(args)

    if (args.model.lower() == 'nuclick'):
        net = NuClick_NN(n_channels=5, n_classes=1)
    elif (args.model.lower() == 'unet'):
        net = UNet(n_channels=5, n_classes=1)
    else:
        raise ValueError('Invalid model type. Acceptable networks are UNet or NuClick')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.pretrained_weights, map_location=device))

    logging.info('Model loaded!')

    for i, image_point in enumerate(images_points):
        try:
            imagePath = image_point[0]
            pointsPath = image_point[1]
            logging.info(f'\nPredicting image {imagePath} ...')
            img = load_image(imagePath)# .resize((512, 512))

            instanceMap = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               points_csv=pointsPath,
                               device=device)

            if not args.no_save:
                #Save instance map
                out_filename = get_output_filename(imagePath, args.output)

                cv2.imwrite(out_filename, instanceMap)
                logging.info(f'Instance map saved as {out_filename}')

            if args.viz:
                #Visualise instance map
                logging.info(f'Visualizing results for image {imagePath}, close to continue...')
                instanceMap_RGB = label2rgb(instanceMap, image=np.asarray(img)[:, :, :3], alpha=0.3, bg_label=0, bg_color=(0, 0, 0), image_alpha=1,kind='overlay')
                plt.figure(), plt.imshow(instanceMap_RGB)
                plt.show()
        except:
            print('Error!, {}'.format(imagePath))
