# Dynamixel Robot Arm

This project is about controling a Dynamixel motors built arm robot with a camera using Python in Ubuntu 24.04.

## create a new Python environment

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