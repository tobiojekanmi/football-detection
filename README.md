# **Object Detection Report: Football Localization in Soccer Field Images**

## **1. Project Overview**
### **1.1. Goal**
The objective of this project is to detect and localize the position of a football (soccer ball) within images of a soccer field.

### **1.2. Given**
- A labeled dataset containing images annotated with **bounding boxes** following the **PASCAL VOC** format (`xmin`, `ymin`, `xmax`, `ymax`).
- The dataset is divided into **training**, **validation**, and **testing** splits.
    
### **1.3. Tasks Completed**
- Analyzed and explored the given data to identify the appropriate modeling strategies.
- Developed a **Convolutional Neural Network (CNN)** to detect the football in images.
- Optimized the model on the **training** dataset, validate during training, and evaluate performance on the **test** dataset.
- Report the **mean detection accuracy** (or equivalent metric) on the test set.


---

## **2. Data Exploration**

### **2.1 Dataset Size and Distribution**

##### **Purpose:**

To determine how many images are in the dataset and how they are distributed across the training, validation, and test sets. This helped determine the appropriate modeling strategy, since:

- A **small dataset** typically requires a simpler model and extensive data augmentation to generalize.
- A **large dataset** enables training more complex architectures with reduced overfitting concerns.

Additionally, verifying the train/val/test split adheres to the standard data splitting conventions.

##### **Method:**

- Calculated the total number of images in the dataset.
- Determined the distribution of images across the training, validation, and test sets.

##### **Findings:**

- The dataset contains a total of 330 images.
- The training set contains 230 images (69.7%), the validation set contains 65 images (19.7%), and the test set contains 35 images (10.6%).
- The dataset is balanced across the three classes.

<!-- ![Data Distribution Across Training, Validation and Test Sets](assets/data-distribution.png "Data Distribution Across Training, Validation and Test Sets"){ width=50% } -->

<!-- <img src="assets/data-distribution.png" alt="Data Distribution Across Training, Validation and Test Sets" width="600" height="500"> -->
<!-- <img src="assets/data-distribution.png" 
     alt="Data Distribution Across Training, Validation and Test Sets" 
     style="max-width: 100%; height: auto; width: 600px;"> -->

<p align="center">
  <img src="assets/data-distribution.png" 
       alt="Data Distribution Across Training, Validation and Test Sets" 
       width="600">
       <br><br>
       <b>Figure 1: Data Distribution Across Training, Validation and Test Sets</b>
</p>

##### **Conclusion:**

The dataset was properly split across the three categories. However, the small size of the dataset may limit the model's ability to learn or generalize to new, unseen data. Therefore, it is recommended to train a small model and use data augmentation techniques to increase the size and diversity of the dataset, as this could improve the model's performance.

---
### **2.2 Dataset Quality Analysis**

##### **Purpose:**

To assess the accuracy of bounding box annotations and the diversity of the image dataset. A high-quality dataset requires both precise labels and sufficient variability to ensure robust model generalization under real-world conditions.

##### **Method:**

- Randomly sample images from the dataset.
- Perform a visual inspection of the samples and confirm the tightness of their bounding boxes.
- Audit the dataset for a wide range of visual and environmental factors (e.g., lighting, background variation, weather, object occlusion, weather e.t.c.).


##### **Findings:**

As depicted in Figure 2 below,

- The annotations appears to be accurate and consistent, as there are a no loose boxes, and any boxes shifted away from the object.

- The dataset also exhibits a fair amount of diversity, as the images display variability in lighting, camera viewpoints, and the type of football used. The dataset, however, seems to lack variable environmental conditions, such as cloudy or rainy weather; hence, any model developed from it might not perform well under such conditions. The objects in the dataset also appear to be of a consistent scale.

- The target object (football) is very small and constitute only a small percentage of the entire image. This would likely make training difficult as we cannot reduce the size of the given image or zoom out the image without worrying that the entire ball could disappear.

- The training, validation, and testing data appear to have been selected randomly from a single pool. This approach helps to ensure that the model is trained and evaluated on the same data distribution.

<p align="center">
  <img src="assets/quality-analysis.png" 
       alt="Quality Analysis of Sample Data" 
       width="600">
       <br><br>
    <b>Figure 2: Quality Analysis of Sample Data</b>
</p>

  
  

##### **Conclusion:**

Based on the findings, any additional augmentation performed must be robust and prioritize the visibility of the target object in transformed images. Geometric transformations such as scaling (only zooming in) and flipping are good choices, but it is essential we test a whole range of both geometric and photometric augmentations to determine which would contribute better and which wouldn't.


---



## **3. Data Augmentation**

##### **Purpose:**

To evaluate how various data augmentation strategies affect dataset quality, with a specific focus on the visibility of the football after transformation.

##### **Method:**
- Apply a range of geometric and photometric augmentation techniques to sample images from the training set.
- Assess the impact of each technique on football visibility to determine the optimal augmentation strategy.
- Tune the augmentation parameters to quantify the risk associated with each technique.  

##### **Strategies tested and Findings:**

The football, being a small object, is highly vulnerable to geometric transformations that reduce its size or shift it outside the image frame. Its typical position near the edges of the frame further increases this risk. However, its uniform color and circular shape make it resilient to photometric changes and rotational transforms, respectively.

The risk profile of each augmentation technique analyzed is summarized below:  

| Transform | Safe Range | Ball Visibility Risk Level | Rationale |
| :--- | :--- | :--- | :--- |
| **Horizontal Flip** | `p=0.0 to 1.0` | 🟢 Low | Preserves all image content; only mirrors it. No risk of the ball disappearing. |
| **Vertical Flip** | `p=0.0 to 1.0` | 🟢 Low | Preserves all image content; only flips it vertically. No risk of the ball disappearing. |
| **Rotation** | `limit <= 10°` | 🟢 Low | Minimal content loss at the edges. The ball's visibility is largely unaffected. |
| **Zoom In** | `scale=1.0 to 1.1` (0-10%) | 🟢 Low | Increases the relative size of the ball, making it more visible and easier to detect. |
| **Color/Brightness/Contrast** | `brightness limit <= 0.1, contrast limit <= 0.1` | 🟢 Low | Alters only pixel values; no geometric changes to object size or location. Change is generally uniform. |
| **Translation** | `translate_percent=0.0 to 0.05` (0-5% shift) | 🟡 Medium | Risk is positional. A small shift is safe unless the ball is on the very edge of the frame. |
| **Perspective Warp** | `scale ≤ 0.1` (Mild) | 🟡 Medium | Can warp the ball or shift it near the edge. Mild distortion is generally safe. |
| **Cropping** | `-` | 🔴 High | While risk is positional, the ball will be lost if it lies within the area being cropped out. This is a huge risk to allow. |
| **Zoom Out** | `-` | 🔴 High | The ball becomes minuscule, and could easily fall become undetectable. |


##### **Conclusion:**  

Based on the risk analysis, we will implement an augmentation strategy that prioritizes football visibility by exclusively using low-risk transforms.




## **4. Modeling**

For the model, I designed a custom CNN enhanced with **residual connections** to improve gradient flow and stabilize training. I used **ReLU** activations throughout the network to ensure efficient non-linear learning without vanishing gradients. I adopted **Layer Normalization** due to its robustness with varying batch sizes, unlike BatchNorm. For stable early training, all layers used **Kaiming (He) Initialization**. The network architecture as shown in Figure 3 Below consists of an initial base convolutional layer to extract features from the input image, multiple residual blocks to process these features, a global average pooling stage, and an MLP head predicting the final bounding box parameters.

<p align="center">
  <img src="assets/model.png" 
       alt="Model Architecture" 
       width="600">
       <br><br>
    <b>Figure 3: Model Architecture</b>
</p>
