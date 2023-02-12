import os
import cv2
import pydicom
import numpy as np
import argparse
import pandas as pd
import yaml
import torch
from torchvision import transforms
from tqdm import tqdm
from image_classification_model import ImageClassificationModel


def expand_greyscale_image_channels(grey_image_arr):
    grey_image_arr = np.expand_dims(grey_image_arr, -1)
    grey_image_arr_3_channel = grey_image_arr.repeat(3, axis=-1)
    return grey_image_arr_3_channel


def find_white_windows(row):

    map_white_windows = {}
    start_value = row[0]
    len_window = 1
    window_info = [row[0], len_window]
    count_window = 0
    map_white_windows[count_window] = window_info

    for i in range(1, len(row)):
        if row[i] == start_value + 1:
            len_window += 1
            map_white_windows[count_window][1] = len_window
            start_value += 1
        else:
            count_window += 1
            len_window = 1
            window_info = [row[i], len_window]
            map_white_windows[count_window] = window_info
            start_value = row[i]

    return map_white_windows


def remove_vertical_line(img, delta_h=50, delta_wind=80, delta_px=10):

    # Find the black and withe image with thershold
    retval, thresh_gray = cv2.threshold(img, thresh=55, maxval=255, type=cv2.THRESH_BINARY)
    h = thresh_gray.shape[0]
    w = thresh_gray.shape[1]

    # Define the orizontal line with the pixel values of the image. One at the beginning, one in the middle, one at the end
    p_0 = delta_h
    p_1 = int( h /2)
    p_2 = h - delta_h
    row_0 = thresh_gray[p_0, :]
    row_1 = thresh_gray[p_1, :]
    row_2 = thresh_gray[p_2, :]

    # Find the indices of the rows where the pixels are white
    white_row_0 = np.where(row_0 == 255)[0]
    white_row_1 = np.where(row_1 == 255)[0]
    white_row_2 = np.where(row_2 == 255)[0]

    # If no vertical line detected return original image
    if white_row_0.shape[0] == 0 or white_row_1.shape[0] == 0 or white_row_2.shape[0] == 0:
        # print("No vertical line")
        return img

    # For each row determine the windows with consecutive white pixels.
    # For each window set the startiung point and the lenght
    list_rows = [find_white_windows(white_row_0), find_white_windows(white_row_1), find_white_windows(white_row_2)]

    # Eliminate from the three lines all those windows whose width is greater than delta_wind=80
    for row in list_rows:
        for i in list(row.keys()):
            if row[i][1] > delta_wind:
                del row[i]

    # Search for the three windows, one for each row, which have the same starting point
    # included in a delta_px interval and the same width included in a delta_px interval
    start_px = 0
    len_wind = 0
    end_px = 0
    for k, win_k in list_rows[0].items():
        for j, win_j in list_rows[1].items():
            for i, win_i in list_rows[2].items():
                if win_k[0 ] -delta_px < win_j[0] <win_k[0] + delta_px and \
                        win_k[0] - delta_px < win_i[0] < win_k[0] + delta_px and \
                        win_k[1] - delta_px < win_j[1] < win_k[1] + delta_px and \
                        win_k[1] - delta_px < win_i[1] < win_k[1] + delta_px:
                    start_px = win_k[0] - delta_px
                    len_wind = win_k[1] + 2 * delta_px
                    end_px = start_px + len_wind

    # replace the pixel value to 0 to each rows of the found common window
    img[:, start_px:end_px] = 0
    img = img.astype(np.uint8)

    return img



def preprocess_image(dicom_path, size, is_pad):
    # take dicom image
    dicom = pydicom.dcmread(dicom_path)
    img = dicom.pixel_array

    # normalize dicom image between 0 and 1
    img = (img - img.min()) / (img.max() - img.min())

    # revert if MONOCHROME1
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img

    # normalize image between 0 and 255
    img = img * 255
    img = img.astype(np.uint8)

    # search for withe pixels of the written
    indices = np.where(img == 255)

    # replace withe pixels with black
    img[indices[0], indices[1]] = 0

    # remove vertical line
    img = remove_vertical_line(img)

    # find the black and white image with thershold
    retval, thresh_gray = cv2.threshold(img, thresh=60, maxval=255, type=cv2.THRESH_BINARY)

    # find the bownding box of the withe region
    points = np.argwhere(thresh_gray == 255)
    points = np.fliplr(points)  # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points)  # create a rectangle around those points

    # crop the image
    img_crop = img[y:y + h, x:x + w]

    if is_pad:

        h_sx = img_crop[0, 0]
        h_dx = img_crop[0, -1]
        l_sx = img_crop[-1, 0]
        l_dx = img_crop[-1, -1]

        # find new_h and new_w for squared the image
        if h > w:
            new_h = new_w = h
        elif h < w:
            new_h = new_w = w
        else:
            new_h = new_w = w

        # create square black image
        final_img = np.zeros((new_h, new_w))
        final_img = final_img.astype(np.uint8)

        # insert the crop image into the square black image depending on the orientation of the crop image
        if h_dx == 0 and l_dx == 0:
            final_img[:h, :w] = img_crop
        elif l_dx == 0 and l_sx == 0:
            final_img[:h, :w] = img_crop
        elif l_sx == 0 and h_sx == 0:
            final_img[:h, new_w - w:] = img_crop
        elif h_sx == 0 and h_dx == 0:
            final_img[new_h - h:, :w] = img_crop

    # resize the final square image
    final_img = cv2.resize(img_crop, (size, size))

    return final_img


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.
    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        print("cfg: ", cfg)
        print("")
    return cfg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config_file', default="", type=str, help='Path of the config file to use')
    parser.add_argument('--is_pad', default="N", type=str, help='Y to pad images - N to no pad images')
    opt = parser.parse_args()

    # 1 - load config file
    path_config_file = opt.path_config_file
    print('path_config_file: ', path_config_file)
    cfg = load_config(path_config_file)
    print("cfg: ", cfg)

    # 2 - set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # 3 - load model and best checkpoint in eval mode
    print("Load the model")
    model = ImageClassificationModel(cfg)

    path_best_checkpoint = os.path.join(cfg["model"]["saving_dir_experiments"], cfg["model"]["saving_dir_model"])
    print("Load the best checkpoint: ", path_best_checkpoint)
    checkpoint = torch.load(path_best_checkpoint, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    model = model.to(device)

    # 4 set is_pad parameter
    is_pad = opt.is_pad
    if is_pad == "Y":
        is_pad = True
    else:
        is_pad = False

    # 5 - load test csv
    test_csv_path = "/kaggle/input/rsna-breast-cancer-detection/test.csv"
    print("test_csv_path: {}".format(test_csv_path))
    test_csv = pd.read_csv(test_csv_path)

    # 6 - set dataset test path
    test_set_path = "/kaggle/input/rsna-breast-cancer-detection/test_images"
    print("test_set_path: {}".format(test_set_path))

    # 7 - start inference on test set
    print("start inference on test set")
    prediction_ids = []
    preds = []

    with torch.no_grad():
        for _, row in tqdm(test_csv.iterrows()):
            prediction_ids.append(row.prediction_id)
            patient_id = str(row.patient_id)
            image_id = str(row.image_id)
            img_path = os.path.join(test_set_path, patient_id, image_id + '.dcm')
            image = preprocess_image(img_path, cfg["data"]["size"])
            image = expand_greyscale_image_channels(image)
            image = transforms.ToTensor()(image)
            image = image[None].to(device)
            outputs = model(image)
            pred_proba = model.post_act_sigmoid(outputs)
            pred_proba = pred_proba.squeeze().cpu().numpy().item()
            preds.append(pred_proba)

    # 8 - create and save submission csv
    print("create and save submission csv")
    submission = pd.DataFrame(data={'prediction_id': prediction_ids, 'cancer': preds}).groupby(
        'prediction_id').max().reset_index()

    submission.to_csv('/kaggle/working/submission.csv', index=False)
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("end inference")

