#!/usr/bin/env python3

# Import the necessary libraries
from tokenize import String
from urllib import request

from sympy import re
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from std_msgs.msg import String # String is the message type
from cv_bridge import CvBridge, CvBridgeError # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
from sensor_msgs.msg import Image
from metrics_refbox_msgs.msg import ObjectDetectionResult, Command

#import pytorch
import torch
import torchvision

class convert_image():
    def __init__(self) -> None:
        rospy.loginfo("ROS image msg to OpenCV image converter node is ready...")
        self.cv_bridge = CvBridge()
        self.image_queue = None
        self.clip_size = 10 #manual number
        self.stop_sub_flag = False

        self.image_sub = rospy.Subscriber(
            "/hsrb/head_rgbd_sensor/rgb/image_raw", Image, self._input_image_cb)
        
        

        self.COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        #publisher
        self.output_bb_str_pub = rospy.Publisher("prediction_bb_str_output", String, queue_size=10)    
        self.output_bb_pub = rospy.Publisher("/metrics_refbox_client/object_detection_result", ObjectDetectionResult, queue_size=10)
        #subscriber
        self.requested_object = None
        self.referee_command_sub = rospy.Subscriber("/metrics_refbox/command", Command, self._referee_command_cb)
        
    def _input_image_cb(self, msg):
        """
        :msg: sensor_msgs.Image
        :returns: None

        """
        try:
            if not self.stop_sub_flag:
                rospy.loginfo("Image received..")
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                if self.image_queue is None:
                    self.image_queue = []
                
                self.image_queue.append(cv_image)
                # print("Counter: ", len(self.image_queue))

                if len(self.image_queue) > self.clip_size:
                    #Clip size reached
                    # print("Clip size reached...")
                    
                    self.stop_sub_flag = True
                    self.image_queue.pop(0)

                    # save all images on local drive
                    cnt = 0
                    for i in self.image_queue:
                        cv2.imwrite('/home/kvnptl/work/heart_met_competition/heart_met_ws/src/object_detection/temp_images/temp_images_' + str(cnt) + '.jpg',i)
                        cnt+=1

                    rospy.loginfo("Input images saved on local drive")

                    # call object inference method
                    # print("Image queue size: ", len(self.image_queue))

                    # waiting for referee box to be ready
                    rospy.loginfo("Waiting for referee box to be ready...")
                    while self.requested_object is None:
                        pass
                    
                    output_prediction = self.object_inference()  
            # else:
            #     print("Clip size reached")
                    
        except CvBridgeError as e:
            rospy.logerr("Could not convert ros sensor msgs Image to opencv Image.")
            rospy.logerr(str(e))
            self._check_failure()
            return
    
    def object_inference(self):

        rospy.loginfo("Object Inferencing Started...")
        
        opencv_img = self.image_queue[0]

        # opencv image dimension in Height x Width x Channel
        clip = torch.from_numpy(opencv_img)

        #convert to torch image dimension Channel x Height x Width
        clip = clip.permute(2, 0, 1)

        # print(clip.shape)

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        clip = ((clip / 255.) * 2) - 1.

        # For inference
        model.eval()
        x = [clip]
        predictions = model(x)

        # print("---------------------------")
        # print("Fast RCNN output: \n",predictions)
        # print("---------------------------")

        #print prediction boxes on input image
        output_bb_ary = predictions[0]['boxes'].detach().numpy()
        output_labels_ary = predictions[0]['labels'].detach().numpy()
        output_scores_ary = predictions[0]['scores'].detach().numpy()

        detected_object_list = []
        detected_object_score = []
        detected_bb_list = []

        # Extract required objects from prediction output
        print("---------------------------")
        print("Name of the objects, Score\n")
        for idx, value in enumerate(output_labels_ary):
            object_name = self.COCO_INSTANCE_CATEGORY_NAMES[value]
            score = output_scores_ary[idx]

            if score > 0.5:
                detected_object_list.append(object_name)
                detected_object_score.append(score)
                detected_bb_list.append(output_bb_ary[idx])

                print("{}, {}".format(object_name, score))

        print("---------------------------")
        
        # if len(detected_object_list) > 0:
        #     self.object_detected = True
        # else:
        #     self.object_detected = False

        # detected_objects = []
        
        
        # Only publish the target object requested by the referee
        if (self.requested_object).lower() in detected_object_list:
            rospy.loginfo("--------> Object detected <--------")
            requested_object_string = (self.requested_object).lower()
            object_idx = detected_object_list.index(requested_object_string)

            # Referee output message publishing
            object_detection_msg = ObjectDetectionResult()
            object_detection_msg.message_type = ObjectDetectionResult.RESULT
            object_detection_msg.result_type = ObjectDetectionResult.BOUNDING_BOX_2D
            object_detection_msg.object_found = True
            object_detection_msg.box2d.min_x = int(detected_bb_list[object_idx][0])
            object_detection_msg.box2d.min_y = int(detected_bb_list[object_idx][1])
            object_detection_msg.box2d.max_x = int(detected_bb_list[object_idx][2])
            object_detection_msg.box2d.max_y = int(detected_bb_list[object_idx][3])

            #convert OpenCV image to ROS image message
            ros_image = self.cv_bridge.cv2_to_imgmsg(self.image_queue[0], encoding="passthrough")
            object_detection_msg.image = ros_image

            #publish message
            self.output_bb_pub.publish(object_detection_msg)

            #draw bounding box on input image
            opencv_img = cv2.rectangle(opencv_img, (detected_bb_list[object_idx][0], detected_bb_list[object_idx][1]), (detected_bb_list[object_idx][2], detected_bb_list[object_idx][3]), (255,255,255), 2)
        
            cv2.imshow('Output Img', opencv_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # requested object not detected
        else:
            rospy.loginfo("xxxxx > Object NOT FOUND < xxxxx")

            # Referee output message publishing
            object_detection_msg = ObjectDetectionResult()
            object_detection_msg.message_type = ObjectDetectionResult.RESULT
            object_detection_msg.result_type = ObjectDetectionResult.BOUNDING_BOX_2D
            object_detection_msg.object_found = False

            #convert OpenCV image to ROS image message
            ros_image = self.cv_bridge.cv2_to_imgmsg(self.image_queue[0], encoding="passthrough")
            object_detection_msg.image = ros_image

            #publish message
            self.output_bb_pub.publish(object_detection_msg)
            

        # for i in detected_bb_list:
        #     opencv_img = cv2.rectangle(opencv_img, (i[0], i[1]), (i[2], i[3]), (255,255,255), 2)
        

        # cv2.imshow('Output Img', opencv_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # ready for next image
        # self.stop_sub_flag = False

        return predictions

    def _referee_command_cb(self, msg):
        # task: 1
        # command: 1
        # task_config: "{\"Target object\": \"Cup\"}"
        # uid: "0888bd42-a3dc-4495-9247-69a804a64bee"
        # if self.object_detected:
        if msg.task == 1 and msg.command == 1:
            self.requested_object = msg.task_config.split(":")[1].split("\"")[1]
            # print("#########Requested object: ", requested_object)
                

if __name__ == "__main__":
    rospy.init_node("convert_rosImg_to_cvImg")
    object_img = convert_image()
    
    rospy.spin()