#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
from os import listdir
from os.path import isfile, join
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """
    # parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    # parser.add_argument("bag_file", help="Input ROS bag.")
    # parser.add_argument("output_dir", help="Output directory.")
    # parser.add_argument("image_topic", help="Image topic.")

    # args = parser.parse_args()

    # print ("Extract images from %s on topic %s into %s" % (args.bag_file,
    #                                                       args.image_topic, args.output_dir))

    # input output paths
    input_bagfiles_dir = "../captured_images"
    input_rostopic = "/hsrb/head_rgbd_sensor/rgb/image_raw"
    output_dir = "../captured_images/extracted_images/"

    # get files from given dir
    onlyfiles = [f for f in listdir(input_bagfiles_dir) if isfile(join(input_bagfiles_dir, f))]

    # loop over all files
    for bag_file in range(len(onlyfiles)):

        file_path = input_bagfiles_dir + "/" + onlyfiles[bag_file]
        # print(file_path)
        bag = rosbag.Bag(file_path, "r")
        bridge = CvBridge()
        count = 0
        

        for topic, msg, t in bag.read_messages(topics=[input_rostopic]):
            
            # count = 0 means only storing 1st image
            if count == 0:

                # convert rosimg to opencv img format (bgr8 = for color img, passthrough = gray scale)
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

                # save images
                cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % bag_file), cv_img)
                print ("Wrote image %i" % bag_file)

            count += 1

        bag.close()

    return

if __name__ == '__main__':
    main()