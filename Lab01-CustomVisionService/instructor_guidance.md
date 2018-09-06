# Instructor guide
This is a hands-on instructor guided lab. As such it is expected that the instructor will walk students through the exercises of the lab and while doing that 
deep dive into the key topics and answer questions as necessary.

The goal of this document is to provide the instructor with hints on what should be covered at each step of the lab.

## Audience
This lab targets developers who may not have an AI/ML background. If there are participants who are new to AI/ML, the instructor should introduce key AI, Machine Learning and Deep Learning concepts. 


## Hints for each step of the lab
**Before you start the hands-on exercises**
- Introduce key Machine Learning and Deep Learning concepts (Optional)
- Introduce key computer vision tasks including:
  - image classification
  - object detection
  - object segmentation
- Introduce key computer vision applications. E.g.:
  - Autonomous vehicles
  - Quality assurance
  - Workplace safety
  - Medical imaging
  - Robotics
- Provide and overview of Azure AI services with the focus on Cognitive Services

**Step 1 - Environment Set Up**
- Briefly review Azure concepts that are relevant in the context of the lab. E.g.  Azure Subscription, Azure Access, Azure Portal, Azure CLI, Azure SDK, Azure Resource Groups, Azure Storage
- Describe different ways of provisionig Custom Vision Service, including Azure Portal and Custom Vision Service web GUI
- Review key components and capabilities of Custom Vision Service, including: 
  - types of models you can train
  - domains
  - Custom Vision Service workflow
  - web GUI
  - supported SDKs 
  - billing model
  - prediction points
  - exporting trained models.
- Provide a high level overview of what is happening behind the scenes: transfer learning
- Guide the students through the provisioning steps

**Step 2 - Training**
- Deep dive into image classification task
- Emphasize the importance of good quality training data
- Review in detail load-train-evaluate-fine tune-repeat workflow of Custom Vision Service
- Explain Precision and Recall measures used by Custom Vision Service for evaluation
- Explain the concept of iteration in Custom Vision Service
- Remind the participants that the training workflow can be controlled through SDK in addition to GUI
- Guide the students through the training steps

**Step 3 - Use prediction endpoint**
- Explain what a prediction endpoint is, including reviewing core parameters: Project ID, Prediction Key, Training Key
- Review the key concepts behind Jupyter notebook
- Walkthrough the `predict.ipynb` notebook explaining the code snippets

**Step 4 - Export the model**
- Explain why being able to export models is important
- Review different export types supported by Custom Vision Service
- Review key concepts behind Tensorflow
- Review key concepts behind Docker
- Guide the students through the export steps



