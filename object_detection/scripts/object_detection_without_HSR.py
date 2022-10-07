#!/usr/bin/env python3

# Import the necessary libraries
from tokenize import String
from urllib import request

import numpy as np
from sympy import capture, re
import rospy  # Python library for ROS
from sensor_msgs.msg import Image  # Image is the message type
from std_msgs.msg import String  # String is the message type
# Package to convert between ROS and OpenCV Images
from cv_bridge import CvBridge, CvBridgeError
import cv2  # OpenCV library
from sensor_msgs.msg import Image
from metrics_refbox_msgs.msg import ObjectDetectionResult, Command
import rospkg
import os
from datetime import datetime

import yolov5

import pdb


class object_detection():
    def __init__(self) -> None:
        rospy.loginfo("Object Detection node is ready...")
        self.cv_bridge = CvBridge()
        self.image_queue = None
        self.clip_size = 2  # manual number
        self.stop_sub_flag = False
        self.cnt = 0

        # yolo model config
        self.model_name = 'best_overfit.pt'
        self.classes_path = 'heartmet.names'
        self.confidence_threshold = 0.5

        # publisher
        self.output_bb_pub = rospy.Publisher(
            "/metrics_refbox_client/object_detection_result", ObjectDetectionResult, queue_size=10)

        # subscriber
        self.requested_object = None
        self.referee_command_sub = rospy.Subscriber(
            "/metrics_refbox_client/command", Command, self._referee_command_cb)

        # waiting for referee box to be ready
        rospy.loginfo("Waiting for referee box ...")

    def _input_image_cb(self, msg):
        """
        :msg: sensor_msgs.Image
        :returns: None

        """
        try:
            if not self.stop_sub_flag:

                # convert ros image to opencv image
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                if self.image_queue is None:
                    self.image_queue = []

                self.image_queue.append(cv_image)
                # print("Counter: ", len(self.image_queue))

                if len(self.image_queue) > self.clip_size:
                    # Clip size reached
                    # print("Clip size reached...")
                    rospy.loginfo("Image received..")

                    self.stop_sub_flag = True

                    # pop the first element
                    self.image_queue.pop(0)

                    # deregister subscriber
                    self.image_sub.unregister()

                    # call object inference method
                    output_prediction = self.object_inference()

        except CvBridgeError as e:
            rospy.logerr(
                "Could not convert ros sensor msgs Image to opencv Image.")
            rospy.logerr(str(e))
            self._check_failure()
            return

    def object_inference(self):

        rospy.loginfo("Object Inferencing Started...")

        opencv_img = self.image_queue[0]

        #####################
        # Load YOLOv5 model for inferencing
        #####################
        # get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()

        # get the file path for object_detection package
        pkg_path = rospack.get_path('object_detection')
        model_path = pkg_path + "/models/"
        data_config_file = pkg_path + "/scripts/" + self.classes_path
        # Give the incoming image for inferencing
        # predictions = run(weights=model_path + self.model_name,
        #                   data=data_config_path + "heartmet.yaml",
        #                   source=opencv_img)

        # self._model = torch.hub.load(
        #     'ultralytics/yolov5', 'custom', path=model_path + self.model_name, force_reload=True)

        # some sanity checks
        if not os.path.isfile(model_path + self.model_name):
            raise FileExistsError(
                f"Weights not found ({model_path + self.model_name}).")

        if data_config_file:
            if not os.path.isfile(data_config_file):
                raise FileExistsError(
                    f"Classes file not found ({data_config_file}).")
            class_labels = self.parse_classes_file(data_config_file)
        else:
            rospy.loginfo(
                "No class file provided. Class labels will not be visualized.")
            class_labels = None

        ob_model = yolov5.load(model_path + self.model_name)
        np_image = np.array(opencv_img, dtype=np.uint8)
        prediction = ob_model(np_image)

        results = prediction.xyxy[0]

        box = np.array(results[:, 0:4], dtype=int)
        conf = np.array(results[:, 4])
        lbl = np.array(results[:, 5])

        #####################
        # print labels
        #####################
        print("conf: ", conf)
        print("lbl: ", lbl)

        bbox = []
        confi = []
        label = []
        # Write results
        for xyxy, cconf, ccls in zip(box, conf, lbl):
            bbox.append([x for x in xyxy])
            confi.append(cconf)
            lbs = class_labels[int(ccls)]
            lbs = lbs.split(':')[1]
            lbs = lbs.lower()
            label.append(lbs)

        predictions = {'boxes': box, 'labels': label, 'scores': confi}

        # extracting bounding boxes, labels, and scores from prediction output
        output_bb_ary = predictions['boxes']
        output_labels_ary = predictions['labels']
        output_scores_ary = predictions['scores']

        detected_object_list = []
        detected_object_score = []
        detected_bb_list = []

        # Extract required objects from prediction output
        print("---------------------------")
        print("Name of the objects, Score\n")
        for idx, value in enumerate(output_labels_ary):
            object_name = value
            score = output_scores_ary[idx]

            if score > self.confidence_threshold:
                detected_object_list.append(object_name)
                detected_object_score.append(score)
                detected_bb_list.append(output_bb_ary[idx])

                print("{}, {}".format(object_name, score))

        print("---------------------------")

        # Only publish the target object requested by the referee
        if (self.requested_object).lower() in detected_object_list:
            rospy.loginfo("--------> Object detected <--------")
            requested_object_string = (self.requested_object).lower()
            object_idx = detected_object_list.index(requested_object_string)
            print("---------------------------")
            print("Bounding Box: ", detected_bb_list)
            print("---------------------------")
            # Referee output message publishing
            object_detection_msg = ObjectDetectionResult()
            object_detection_msg.message_type = ObjectDetectionResult.RESULT
            object_detection_msg.result_type = ObjectDetectionResult.BOUNDING_BOX_2D
            object_detection_msg.object_found = True
            object_detection_msg.box2d.min_x = int(
                detected_bb_list[object_idx][0])
            object_detection_msg.box2d.min_y = int(
                detected_bb_list[object_idx][1])
            object_detection_msg.box2d.max_x = int(
                detected_bb_list[object_idx][2])
            object_detection_msg.box2d.max_y = int(
                detected_bb_list[object_idx][3])

            # convert OpenCV image to ROS image message
            ros_image = self.cv_bridge.cv2_to_imgmsg(
                self.image_queue[0], encoding="passthrough")
            object_detection_msg.image = ros_image

            # publish message
            rospy.loginfo("Publishing result to referee...")
            self.output_bb_pub.publish(object_detection_msg)

            # draw bounding box on target detected object
            # opencv_img = cv2.rectangle(opencv_img, (int(detected_bb_list[object_idx][0]),
            #                                         int(detected_bb_list[object_idx][1])),
            #                            (int(detected_bb_list[object_idx][2]),
            #                             int(detected_bb_list[object_idx][3])),
            #                            (255, 255, 255), 2)

            # display image
            # cv2.imshow('Output Img', opencv_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # requested object not detected
        else:
            rospy.loginfo("xxxxx > Object NOT FOUND < xxxxx")

            # Referee output message publishing
            object_detection_msg = ObjectDetectionResult()
            object_detection_msg.message_type = ObjectDetectionResult.RESULT
            object_detection_msg.result_type = ObjectDetectionResult.BOUNDING_BOX_2D
            object_detection_msg.object_found = False

            # convert OpenCV image to ROS image message
            ros_image = self.cv_bridge.cv2_to_imgmsg(
                self.image_queue[0], encoding="passthrough")
            object_detection_msg.image = ros_image

            # publish message
            self.output_bb_pub.publish(object_detection_msg)

        # draw bounding box on all detected objects (with score >0.5)
        # for i in detected_bb_list:
        #     opencv_img = cv2.rectangle(opencv_img, (i[0], i[1]), (i[2], i[3]), (255,255,255), 2)

        # display image
        # cv2.imshow('Output Img', opencv_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # ready for next image
        self.stop_sub_flag = False
        self.image_queue = []

        return predictions

    def _referee_command_cb(self, msg):

        # Referee comaand message (example)
        '''
        task: 1
        command: 1
        task_config: "{\"Target object\": \"Cup\"}"
        uid: "0888bd42-a3dc-4495-9247-69a804a64bee"
        '''

        # START command from referee
        if msg.task == 1 and msg.command == 1:

            print("\nStart command received")

            # start subscriber for image topic
            # HSR raw camera topic
            # self.image_sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw",
            #                                   Image,
            #                                   self._input_image_cb)
            # Intel Realsense camera topic
            # self.image_sub = rospy.Subscriber("/camera/color/image_raw",
            #                                   Image,
            #                                   self._input_image_cb)
            # ASUS openni camera topic
            # self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",
            #                                   Image,
            #                                   self._input_image_cb)

            # Tiago robot camera topic
            self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw",
                                              Image,
                                              self._input_image_cb)

            # extract target object from task_config
            self.requested_object = msg.task_config.split(":")[
                1].split("\"")[1]
            print("\n")
            print("Requested object: ", self.requested_object)
            print("\n")

        # STOP command from referee
        if msg.command == 2:
            self.stop_sub_flag = True
            self.image_sub.unregister()
            rospy.loginfo("Received stopped command from referee")
            rospy.loginfo("Subscriber stopped")

    def parse_classes_file(self, path):
        classes = []
        with open(path, "r") as f:
            for line in f:
                line = line.replace("\n", "")
                classes.append(line)
        return classes


if __name__ == "__main__":
    rospy.init_node("object_detection_node")
    object_detection_obj = object_detection()

    rospy.spin()
