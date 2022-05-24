#!/usr/bin/env python3

"""
Rename files in two folders at a same time (image and label directories)

!!!!!!!!!!!
! CAUTION: THE CODE WILL REPLACE CURRENT FILE NAMES !
!!!!!!!!!!

"""

import os

import sys


def main():

    # img path
    train_path_img = '/home/lucy/heartmet/heart_met_ws/src/HEART_MET_ODFBM/object_detection/dataset_bkup_24_may/images/train'
    test_path_img = '/home/lucy/heartmet/heart_met_ws/src/HEART_MET_ODFBM/object_detection/dataset_bkup_24_may/images/test'
    val_path_img = '/home/lucy/heartmet/heart_met_ws/src/HEART_MET_ODFBM/object_detection/dataset_bkup_24_may/images/val'

    # label path
    train_path_label = '/home/lucy/heartmet/heart_met_ws/src/HEART_MET_ODFBM/object_detection/dataset_bkup_24_may/labels/train'
    test_path_label = '/home/lucy/heartmet/heart_met_ws/src/HEART_MET_ODFBM/object_detection/dataset_bkup_24_may/labels/test'
    val_path_label = '/home/lucy/heartmet/heart_met_ws/src/HEART_MET_ODFBM/object_detection/dataset_bkup_24_may/labels/val'

    # get the files and sort
    files_img = sorted(os.listdir(val_path_img))
    files_label = sorted(os.listdir(val_path_label))

    # new file names
    img_name = "image_"
    label_name = "label_"

    # check if both directories have same number of files
    if len(files_img) == (len(files_label)):

        # iterate over the files and replace the same file with new name
        for index, file in enumerate(files_img):
            file_ext_img = files_img[index].split(".")[-1]
            file_ext_label = files_label[index].split(".")[-1]

            # img
            os.rename(os.path.join(val_path_img, files_img[index]), os.path.join(
                val_path_img, ''.join([img_name + "_" + str(index), "." + file_ext_img])))

            # label
            os.rename(os.path.join(val_path_label, files_label[index]), os.path.join(
                val_path_label, ''.join([label_name + "_" + str(index), "." + file_ext_label])))

    else:
        # sys.exit(
        #     "!! ERROR: both images and labels folder have same number of files !!")
        raise ValueError(
            '!! ERROR: both image and label folders must have same number of files !!')

    print("Renamed {} files".format(len(files_img)))


if __name__ == '__main__':
    main()
