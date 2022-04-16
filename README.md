# HEART-MET METRICS project | Object Detection Functional Benchmarking

### System config:
- Ubuntu 20.04 LTS
- ROS Noetic
- Python 3

### Clone dependent repositories/ros packages (put it in src directory):

1. `git clone https://github.com/kvnptl/metrics_refbox.git`
2. `git clone https://github.com/kvnptl/metrics_refbox_client.git`
3. `git clone https://github.com/kvnptl/rosbag_recorder.git`
4. `git clone https://github.com/HEART-MET/metrics_refbox_msgs.git`
5. `sudo apt install ros-noetic-rospy-message-converter`
6. `pip3 install sympy`

### How to run the code:

1. Run object detection code (and metrics refbox client node):  
`roslaunch object_detection object_detection_benchmark.launch`

2. Publish images from rosbag file (-l means publish continuosly):  
`rosbag play -l /home/kvnptl/work/heart_met_competition/bagfiles-001/bagfiles/b-it-bots_2020_11_24_10-17-01.bag`

3. Launch Refbox node:  
`roslaunch metrics_refbox metrics_refbox.launch`

4. Select target object from referee box GUI(checkmark configuration button) and press "Start" to send command to ros node
