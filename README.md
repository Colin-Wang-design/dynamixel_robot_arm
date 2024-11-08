# Dynamixel Robot Arm

This project is about controling a Dynamixel motors built arm robot with a camera using Python in Ubuntu 24.04. Mac should be also works by specifying the serial port number and camera ID.

## Create a new Python environment

First we need to create a new Python environment to install the dynamixel library, otherwise, it will confilct with system base environment. Two options we can choose, first using python3 to build:
```
mkdir my-project   #create a folder named my-project
cd my-project
python3 -m venv myenv    #create a python environment called myenv
source myenv/bin/activate    # activte the env
```
Second, using conda to build:
```
conda create -n myenv 
```
After build successgully, activate it then install the dynamixel library
```
conda activate myenv
```

## Clone this repository
Using the following command in your work folder:
```
git clone https://github.com/Colin-Wang-design/dynamixel_robot_arm.git
``` 
Then go into the dynamixel_robot_arm directory:
```
cd dynamixel_robot_arm
```
Make sure your env is activated, then install the dynamixel library:
```
python3 setup.py install
```
## Run sample code

You can open the folder dynamixel_robot_arm in a Python editer e.g. VScode. To run sample scripts, make sure you are in the environment where dynamixel library was installed, in VS code press shift + ctrl + p, then type:
```
python select interpreter
```
Select the env you installed the library.

The "project.py" script connect with the robot and read four motors positon then stop.

Then make sure "DEVICENAME = '/dev/ttyACM0' " is correspond to you port number. It will return the current pose of four motors.

The "camera_sample.py" script uses the camera on the robot, it opens a video capture device with a resolution of 800x600, convert images to gray scale, then converts images into a binary image at the specified threshold, finally find boundaries and darw in red lines.

Make sure the "camera_id" is correspond to your device.