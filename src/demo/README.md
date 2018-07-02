## Realsense Realtime Demo Using Pose-REN


<p align="center">
    <b>ICVL</b><br>
    ![demo_icvl_cxh](../../doc/demo_icvl_cxh.gif)
</p>

<p align="center">
    <b>NYU</b><br>
    ![demo_nyu_cxh](../../doc/demo_nyu_cxh.gif)
</p>

<p align="center">
    <b>MSRA</b><br>
    ![demo_msra_cxh](../../doc/demo_msra_cxh.gif)
</p>

### Realsense Realtime Demo
We provide a realtime hand pose estimation demo using Intel Realsense device.
Note that we just use a naive depth thresholding method to detect the hand. Therefore, the hand should be in the range of [0, 650mm] to run this demo.
We tested this realtime demo with an [Intel Realsense SR300](https://software.intel.com/en-us/realsense/sr300camera).

**Please use your right hand for this demo and try to avoid clustered background and redundant arm around the hand.**

#### Python demo with [librealsense](https://github.com/IntelRealSense/librealsense) [recommended]
First compile and install the [librealsense](https://github.com/IntelRealSense/librealsense) and its [python wrapper](https://github.com/IntelRealSense/librealsense/tree/5285629b4ddb374f1). After everything is working properly, just run the following python script for demo:
``` bash
python src/demo/realsense_realtime_demo_librealsense2.py
```

By default this script uses pre-trained weights on ICVL dataset. You can change the pre-trained model by specifying the dataset.
``` bash
python src/demo/realsense_realtime_demo_librealsense2.py  nyu/msra/icvl
```

Notes: The speed of this python demo is not optimal and it runs slower than the c++ demo.

#### C++ demo

First compile the codes:

```
cd src/demo/pose-ren-demo-cpp
mkdir build
cd build
cmake ..
make -j16
```
Run the demo by:
```
cd ..                 # redirect to src/demo/pose-ren-demo-cpp
./build/src/PoseREN   # run
```

By default it uses pre-trained weights on Hands17 dataset. You can change the pre-trained model by specifying the dataset.
``` bash
./build/src/PoseREN  nyu/msra/icvl
```

Notes: This C++ demo is not fully developed and you may have to play with some dependency problems to make it works. It servers as a preliminary project to demonstrate how to use Pose-REN in C++.
