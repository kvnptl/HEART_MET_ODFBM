# heart-met-metrics-project

### How to run the code:

1. Run object detection code:
`roslaunch object_detection object_detection_benchmark.launch`

2. Publish images from rosbag file:
`rosbag play /home/kvnptl/work/heart_met_competition/bagfiles-001/bagfiles/b-it-bots_2020_11_24_10-17-01.bag`

3. Launch Refbox node:
`roslaunch metrics_refbox metrics_refbox.launch`

4. Launch Ref Client node:
`roslaunch metrics_refbox_client metrics_refbox_client.launch`