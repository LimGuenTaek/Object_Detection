# Feature Fusion and Small object Detection

## Detection with Better Features

* The quality of feature representations is critical for object detection. 

* In recent years , many researchers have made efforts to further improve the quailty of image features on basis of some latest engines , where the most important two groups of methods are 

    * **Feature fusion**
    * Learning high-resolution features with large receptive fields
---
**Q . Why Feature Fusion is important?**

- **Invariance** and **equivariance** are two important properties in image feature representations

- Classification desires invariant feature representations since it aims at learning high-level semantic information

- Object localization desires equivariant representations since it aims at discriminating position and scale changes

- As object detection consists of two sub-tasks of object recognition and localization , it is crucial for a detector to learn both invariance and equivariance at the same time

- As CNN model consists of a series of convolutional and pooling layer , features in deeper layers will have stronger invariance but less equivariance 

- Although this could be beneficial to category recognition , it suffers from low localization accuracy in object detection.
	
- On the contrary , features in shallower layers is not conducive to learning semantics , but it helps object localization as it contains more information about edges and contours

####  Therefore , the integration of deep and shallow features in a CNN model helps improve both invariance and equivariance
---
**Q . How to Fuse the Feature_map?**

* There are many ways to perform feature fusion in object detection 
    * **processing flow**
    * **element-wise operation**

	**Processing flow**

    * Recent feature fusion methods in object detection can be divided into two categories : **1) bottom-up fusion** , **2) top-down fusion**

    * Bottom-up fusion feeds forward shallow features to deeper layers via **skip connections**.

    * In comparison , top-down fusion feeds back the features of deeper layers into the shallower ones

    * Apart from these methods , there are more complex approaches proposed recently [weaving features across different layers] 

    * As the feature maps of different layers may have different sizes both in terms of their spatial and channel dimensions

    * one may need to accommodate the feature maps such as by adjusting the number of channels , **up-sampling** low resolution maps or **down-sampling** high resolution maps to proper size

    * The easiest way to do this is to use **nearest** or **bilinear interpolation**

    * Besides **fractional strided convolution (a.k.a transpose convolution)** is another recent populaer way to resize the feature maps and adjust the number of channels 
        * The advantage of using fractional strided convolution is that it can learn an appropriate way to perform up-sampling by it self

	**Element-wise operation**

    * From a local point of view , feature fusion can be considered as the element wise operation between different feature maps : **1) element-wise sum , 2) element-wise product , 3) concatenation**

    * The element-wise sum is the easiest way to perform feature fusion. It has been frequently used in many recent object detectors.

    * The element-wise product is very similar to the element wise sum , while the only difference is the use of multiplication instead of summation
        * The advantage of element-wise product is that it can be used to suppress or highlight the features within a certain area , which may further benefit small object detection

    * Feature concatenation is another way of feature fusion.
        * The advantages of concatenation is that it can be used to integrate context information of different regions , while disadvantage is the increase of the memory

---

## Small Object Detection Algorithm Based on Feature Pyramid-Enhanced Fusion SSD(2019 , Haotian Li et al)

---
Reference : https://downloads.hindawi.com/journals/complexity/2019/7297960.pdf
---
#### Abstract

* Firstly the selected multiscale feature layer is merged with the scale-invariant convolutional layer through feature pyramid network structure , at the same time the multiscale feature map is separately converted into the channel number using the scale- invariant convolution kernel. 

* Then , the obtained two sets of pyramid shaped feature layers are further feature fused to generate a set of enhanced multiscale feature maps and the scale-invariant convolution is performed again on these layer

* Finally , the obtained layer is used for detection and localization

#### Introduction

* Although the traditional SSD model uses the multiscale pyramid feature layer for boudning box extraction , the shallow features used in the structure are only one layer , and the different-sized feature maps are not related to each other , resulting in less feature details , while the detection of small objects the requires high resolution feature maps , resulting in a weaker effect on small object detection

* the feature pyramid network is used to improve the SSD algorithm , and combined with the feature fusion , feature pyramid-enhanced fusion based on the SSD(FPEF-SSD) is proposed which uses the feature pyramid network to fusion the feature of the upsampling layer and scale-invariant convolutional layer while retaining the multiscale feature layer extracted by the traditional SSD structure


#### Related Works

**1. The single-Shot-Detector(SSD)**

  - Based on the VGG 16 network structure the SSD algorithm extracts multiple sets of feature layers in a shape of pyramid for object class prediction and object frame labeling
	
  - Copaerd with the regional proposal-based convolutional neural network , the SSD algorithm cancels a large numbers of regions
	
  - Feature 4_3 need to be L2-regularized because it has a significant fluctuation ratio compared to that of other feature extraction layers
	
  - Then , through two sets of convolution kernels with a size of 3 and a quantity of N , for each feature extracted and for each position on it , K default boxes will be generated and K will be 4 or 6 ; in each default box position , a confidence 
	  value will be generated for c categories. 
	
  - After summarizing the output of each training image , the positive and negative sample data need to be determined accordingly using IoU
	
  - The output of all the feature maps can be classified into a positive sample or a negative sample according to whether the IoU is greater than a specified threshold and the ratio is 1 : 3 (positive : hard negative)
	

**2. Multiscale Feature Analysis of SSD**

  - According to the reasoning of the SSD architecture , the feature information of each layer is only determined by the previous layer.
	
  - Therefore , every feature layers need to be complex and abstract enough to detect the object more accurately
	
  - This means the selected feature map needs a certain resolution basis to provide better detailed expression for the detector
	
  - The problem is that SSD algorithm mainly uses high-level abstract features for detection (7, 8 , 9, 10 ,11) and uses low-level feature layer only conv4_3 for small object detection


### Feature Pyramid-Enhanced Fusion SSD

![image](https://user-images.githubusercontent.com/70448161/108473762-cdb73780-72d1-11eb-96c1-d2481bcd8b62.png)

* **How to Fuse the multi-scale feature map on SSD model**

    * Based on the SSD algorithm and pyramid network structure , an SSD object detection algorithm combined with the improved feature pyramid network fusion method is proposed , called FPEF-SSD

    * The network first inputs the image from left to right and the size of the input image is cropped to 300.

    * The first part is the original SSD model feature selection layer and then the six pyramid feature maps are obtained

    * The first five feature maps are subjected to a scale-invariant convolution operation by using a convolution kernel of size 1 , step size 1 , and channel number 256 , whose aim is to unify the number of channels of all feature maps with the channel of the highest layer

    * The feature of edge information is preserved to the greatest extent because of the complementary operation 

    * And this convolved layer is named X-1 , where X represents the original feature layer name ; then , the upper sampling operation is carried out for these five layers except the first layer and these layers are enlarged two times to the original one by using the nearest neighbot interpolation  

    * Next starting from the bottom layer , feature fusion is carried out successively with the upper sampling layer of the previous layer (black solid circles in Figure)

    * Here the feature fusion is element-wise addition , which means the values at the corresponding positions of the two sets of features are added , so the condition is that the size of layers and the number of channels are exactly the same size

    * After the first fusion operation , because of the combination of deep and shallow features , the interpolation operation of upper sampling in the shallow layer will bring errors , so the convolution operation is generally required to complete the fuzzy removal

    * The algorithm in this paper enhances the feature of this layer before the convolution operation

    * Specifically , the first five layers of features after the first fusion are fused with the features before the upsampling again (black solid squares in Figure) by using the fusion feature cascade (Concat)

    * This time , the number of channels in both sets of features 256 , so there is no need for additional batch normalization processing which can as far as possible ensure the detection speed 

    * Finally , these enhanced features are convolved again , the convolution kernel of size 3 is used from the high to the low feature layer , but the number of channel is 512 , 1024 , 512 , 256 ,256 successively

    * Then , taking NMS and prediction


---
