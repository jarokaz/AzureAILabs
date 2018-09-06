# Image Classification with Custom Vision Service
In this lab you will train, evaluate, deploy, and use a custom image classification model using Microsoft Cognitive Services Custom Vision Service. 

The lab is designed to be instructor guided.  In addition to walking you through the lab's steps, the instructor will explain key concepts and as necessary deep dive into technical details. 

Don't hesitate to ask questions !

## What will you learn during the lab?
The lab consists of 2 parts:
- In the first part you will learn how to train, evaluate, fine tune, and invoke a custom image classification model.
- In the the second part, you will export your model as a docker image so it can be deployed into an arbitrary inference environment

Both parts are implemented as Jupyter notebooks and utilize Custom Vision Service Python SDK. Note, that there are other ways of working with Custom Vision Service, including Go and C# SDKs and Custom Vision Service GUI. In fact all these interfaces are front-ends to Custom Vision Service REST API that is thoroughly documented in the following links:

https://southcentralus.dev.cognitive.microsoft.com/docs/services/d0e77c63c39c4259a298830c15188310/operations/39b14cb5f7f34977a6e6a290

https://southcentralus.dev.cognitive.microsoft.com/docs/services/450e4ba4d72542e889d93fd7b8e960de/operations/5a6264bc40d86a0ef8b2c290




## Scenario

You will train a custom image classification model to automatically classify the type of land shown in aerial images of 224-meter x 224-meter plots. Land use classification models can be used to track urbanization, deforestation, loss of wetlands, and other major environmental trends using periodically collected aerial imagery. The images used in this lab are based on imagery from the U.S. National Land Cover Database. U.S. National Land Cover Database defines six primary classes of land use: *Developed*, *Barren*, *Forested*, *Grassland*, *Shrub*, *Cultivated*. For the sake of simplicity, in this lab you will train and operationalize a classifier to recognize three classes: *Barren*, *Developed*, *Cultivated*.  Example images in each land use class are shown here:

Developed | Cultivated | Barren
--------- | ------ | ----------
![Developed](/Datasets/AerialSmall/train/Developed/ortho_1-1_hn_s_ca025_2016_1_104257.png) | ![Barren](/Datasets/AerialSmall/train/Cultivated/ortho_1-1_hn_s_ca025_2016_1_9900.png) | ![Cultivated](/Datasets/AerialSmall/train/Barren/ortho_1-1_hn_s_ca025_2016_1_7359.png)


## Lab environment

Although you can install all of the components required to complete the lab on your local workstation, we will use Azure Data Science Virtual Machine (DSVM). Azure DSVM comes with most of the components pre-installed, which will make the setup faster and easier.

To complete the lab you will need the following pre-requisities:

- A basic proficiency in Python programming
- A valid Microsoft account or an Azure Active Directory OrgID ("work or school account")
- An Azure subscription associated with your Microsoft Account or OrgID. If you don’t have an Azure subscription, you can create a trial subscription before you begin.
- A workstation with the latest version of Chrome browser

To configure the lab's environment follow the below steps:

### Provision and configure Azure DSVM

You will use Azure Portal to provision Azure DSVM. 

1. Navigate and log in to Azure Portal

https://portal.azure.com

2. Click on **Create a resource** in the top left corner

3. Enter *Data Science Virtual Machine* in the search text box

![Create DSVM](images/img16.PNG)

4. Select *Data Science Virtual Machine for Linux (Ubuntu)*

5. Configure DSVM:
  - Use password rather than SSH key for authentication
  - Use *Standard SSD*
  - Create a new resource group
  - Use *D4s_V3* or similar for the VM type
  - Leave all other parameters at default values
  
6. DSVM comes preconfigured with Jupyter Notebook and Jupyter Lab. You will use Jupyter Lab. After your DSVM is ready, navigate to the below URL. Your browser may complain abouth an invalid certificate - ignore the warnings and proceed to log in using the credentials you created during the VM setup. 

https://https://your-vm-ip:8000/user/your-username/lab.

7. Your instructor will explain key elements of the Jupyter Lab interface and show you how to clone the lab's repo using the Jupyter Lab's terminal window. Make sure to clone the repos under the *Notebooks* folder in your home directory.

To configure the lab:

1. Activate your conda environment

On Linux/MacOs

```
source activate <environment name>
```

On Windows

```
activate <environment name>
```

2. Install Custom Vision Service SDK

```
pip install azure-cognitiveservices-vision-customvision
```

3. Clone or download and unzip the AzureAI labs repo into the preferred location on your file system

```
git clone 





[Next Step](https://github.com/Microsoft/MTC_AzureAILabs/tree/master/Lab01%20-%20Image%20Classification%20with%20Custom%20Vision%20Service/EnvironmentSetup)


