# Active Learning in the Context of Traffic Light Detection (Group Project)

This project explores the potential of active learning to minimize labeling efforts while maintaining performance in training an object detection model for traffic lights. We develop an active learning framework and evaluate its effectiveness on two datasets: the Bosch Small Traffic Lights Dataset and a custom dataset from the DAI-Labor at TU Berlin.

NOTE: Not all files and function were uploaded, due to privacy reasons. Moreover, all work were done in Gitlab

## Key Objectives
Establish a Framework: Develop a framework for training an object detection model with an automatic active learning loop using the labeled Bosch Small Traffic Lights Dataset.

Test Query Strategies: Evaluate various query strategies (uncertainty-based, diversity-based, hybrid, and random) to identify the most effective approach for minimizing labeling while maximizing model performance.

Apply to Real-World Data: Apply the insights gained from the previous phases to refine the object detection model on real-world data from the DAI-Labor dataset.

## Methods
Object Detection: We utilize the YOLOv8 object detection model, chosen for its superior performance in detecting traffic lights.

Active Learning: We implement and test various query strategies, including uncertainty-based, diversity-based, and hybrid approaches.

Datasets: We use the Bosch Small Traffic Lights Dataset for initial framework development and strategy evaluation, and the DAI-Labor dataset for real-world application.

## Results
Our results demonstrate that active learning can effectively reduce labeling efforts in traffic light detection. We find that query strategies involving a high number of predicted bounding boxes tend to outperform other methods. Additionally, we observe that pool-based training, where the model is trained on the entire labeled pool in each iteration, consistently outperforms batch-based training.

## Conclusion
This project provides valuable insights into the application of active learning for object detection tasks, particularly in the context of traffic light detection. Our findings highlight the potential of active learning to streamline the labeling process and improve model performance, offering a promising direction for future research and development in this field.

