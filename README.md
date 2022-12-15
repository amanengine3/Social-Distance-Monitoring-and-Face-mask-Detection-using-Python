## Social Distance Monitoring and Face mask Detection using Python :mask:
In this project, We have used python in conjunction with deep learning and computer vision to track social distancing and detect masks in a public area.

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](https://img.shields.io/badge/yolo%20-v3-yellowgreen?style=for-the-badge&logo=Yolo)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


![63552](https://user-images.githubusercontent.com/60310409/207961393-356f565a-cc83-4ef5-b328-9b285270d2f6.jpg)
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; 
 
> ## About-- 
Based on the bird's eye view and 6 feet distance points marked by the user, the
algorithm calculates and reports the following parameters.
1. # 6ft violation: Number of times the pedestrians violated the 6 feet safe distance
threshold
2. Stay-at-home Index: Quantifies how many people are staying at home as
compared to the normal routine pedestrian traffic. 0% means that there is no change in
the pedestrian traffic compared to normal days, 50% means half of the people are
staying at home
3. Social-distancing Index: Quantifies the social distancing being maintained. 50%
means half of the interactions violate the safe 6 feet distance criteria.
4. Face Mask Prediction: Detects the face mask with 98% accuracy. Green rectangle
means the person is wearing the mask and red rectangle means not wearing the mask.
> ## System Requirement -- :desktop_computer:

 -  **SOFTWARE--**
	 &nbsp; &nbsp; &nbsp;   
	 * Software: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Anaconda + Python 3.x (3.8 or earlier) 
	 * Editor: &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; VS Code/ PyCharm/ Sublime/ Spyder </br>
	 * Environment: &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; TensorFlow 
	 * GPU Drivers:&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; Nvidia® CUDA® 11.0 requires 450.x or above 
		</br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; CUDA® Toolkit (TensorFlow >= 2.4.0) 
		</br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cuDNN SDK 8.0.4 (TensorFlow >= 2.4.0)
* **HARDWARE--**
	* GPU: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Graphics Processor (NVIDIA) ̶min 2GB 	
	*	Camera: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CCTV/ Webcam/ Mobile Camera (Sharing Camera) 
	*	Storage Disk (Optional): &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SSD – Min 400MB/s Read Speed	


 
> ## Installation Process--
 * Download Anaconda Software -- 

||Operating System | Download Link  |
|--|--|--|
|:point_right:|Windows | [click here](https://docs.anaconda.com/anaconda/install/windows/)  |
|:point_right:|Mac|[click here](https://docs.anaconda.com/anaconda/install/mac-os/) |
|:point_right:|Linux | [click here](https://docs.anaconda.com/anaconda/install/linux/) |


:loudspeaker: During Installation be sure to check to set *Environmental variable path* 

* Create new Environment for the installation of libraries:
	* Open Command Prompt / Anaconda Prompt and type `conda create --name tf_python`  
	you can set any name in place of tf_python to create a new envionment. and after type `y` and enter.
	* Install all required Libraries given in <font color="red">requirement.txt</font> by using command `pip install -r requirement.txt`
	 
> #### Required Libraries--
* ![](https://img.shields.io/badge/TensorFlow-v2.4.0-blue)   &nbsp; &nbsp; &nbsp;  &nbsp;&nbsp;  [TensorFlow](https://pypi.org/project/tensorflow/)
* ![](https://img.shields.io/badge/TensorFlow--GPU-v2.4.0-blue) &nbsp;   &nbsp;[TensorFlow-GPU](https://pypi.org/project/tensorflow-gpu/)
* ![](https://img.shields.io/badge/python-v3.7-blue) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;  [Python](https://www.python.org/downloads/)
*  ![](https://img.shields.io/badge/SciKit%20Learn-v0.24.0-blue) &nbsp; &nbsp; &nbsp;&nbsp; [Scikit Learn](https://pypi.org/project/scikit-learn/)
* ![](https://img.shields.io/badge/Open%20CV-v4.4.0.46-blue)    &nbsp;   &nbsp; &nbsp; &nbsp; [Computer Vision](https://pypi.org/project/opencv-python/)
* ![](https://img.shields.io/badge/SciPy-v1.6.0-blue) &nbsp;   &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp;  &nbsp;[Scientific Python](https://pypi.org/project/scipy/)

 
> ## File Required to Download -- 
 * **DATASETS :** </br>
	 &nbsp;  &nbsp;  &nbsp;  &nbsp; Using datasets to train the model for ***Face Mask Detection*** model. To download the dataset -- :point_right:<a href="https://www.kaggle.com/shantanu1118/face-mask-detection-dataset-with-4k-samples"> *Click here* </a>  &nbsp;:point_left: (Dataset with 4,000 Images Sampels) :star2:File contain 2 Sub-Folder i.e. With_mask & Without_mask (each folder contain 2k samples of images). 
	 
This is a balanced dataset containing faces with and without masks with a mean height of 283.68 and mean width of 278.77

![data](https://user-images.githubusercontent.com/47710229/97522777-a243d080-19f4-11eb-93c9-04dea6ceec6c.png)


 *	 **Yolo Weights  (V3) -- Pre-Trained model:**   
 &nbsp; &nbsp; &nbsp; &nbsp; YOLO *(You Only Live Once)*, the pre-trained weights of the neural network are stored in `yolov3.weights`
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Download the Weight File :point_right: <a href="https://drive.google.com/file/d/1MILq56BADd3Tj173HekMm6aycLx9gruk/view?usp=sharing">*Click here* </a> :point_left: 


> ## Trained Result of Face Mask Model--
![](https://github.com/Shantanugupta1118/Social-Distancing-and-Face-Mask-Detection/blob/main/plot.png)
> ## File Structure 
Set all downloaded files to their respective folders/path as given in Folder Structure Diagram.
![](https://github.com/Shantanugupta1118/Social-Distancing-and-Face-Mask-Detection/blob/main/Folder%20Structure.png)
> ## RUN the Main Module--
*	Using Command Prompt or Anaconda Prompt:
	*  To activate environment:--&nbsp; &nbsp; &nbsp; `conda activate tf_python`
	* Run main module:-- `python main.py`
	

> ## Outputs--
![](https://github.com/Shantanugupta1118/Social-Distancing-and-Face-Mask-Detection/blob/main/Output%20Data/output_2.jpg)
> ## Contribute:
&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<b>:fire: Contributions are always welcome!</b>
# Drop a :star: if you like this Repository.. :smile: 
	
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[![enjoy][enjoy-image]][linkedin-url] &nbsp;&nbsp;&nbsp;&nbsp; [![status][issue-image]][issue-url]

[enjoy-image]: https://img.shields.io/badge/Enjoy%20this%3F-Say%20Thanks!-yellow
[linkedin-url]: https://www.linkedin.com/in/shang1118/
[issue-image]: https://img.shields.io/badge/Any%20Issues%3F-Track%20Issue-red
[issue-url]: https://github.com/Shantanugupta1118/Social-Distancing-and-Face-Mask-Detection/issues
:point_right::point_right::point_right: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[![](https://img.shields.io/badge/Shantanu%20-Gupta-orange?style=for-the-badge&logo=Coder)]:point_left::point_left::point_left:
