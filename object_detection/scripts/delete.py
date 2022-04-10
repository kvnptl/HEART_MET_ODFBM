
###############################

# a = "{\"Target object\": \"Cup\"}"
# print(a.split(":")[1].split("\"")[1])
# print("Hello")

# b = [1,2,3,5,6,7,8,9,10]

# if 5 in b:
#     print("Yes")

################################

# import rospkg

# # get an instance of RosPack with the default search paths
# rospack = rospkg.RosPack()

# # get the file path for rospy_tutorials
# pkg_path = rospack.get_path('object_detection')
# print(pkg_path)

# # This requires Python’s OS module
# import os
 
# # 'makedirs' creates a directory with it's path, if applicable.
# name = os.makedirs(pkg_path + '/temp_img_1/') 
# print(name)

################################

from datetime import datetime

# datetime object containing current date and time
now = datetime.now()

# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
# dt_string = dt_string.replace("/", "_")
# dt_string = dt_string.replace(":", "_")
# dt_string = dt_string.replace(" ", "_")
print(dt_string)

################################