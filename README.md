# Assignment 3: Using Pretrained CNNs for Image Classification

## Repository Overview
1. [Description](#description)
2. [Repository Tree](#tree)
3. [Usage](#usage)
4. [Model Description](#model)
5. [Results](#results)
6. [Discussion](#discuss)

## Description <a name="description"></a>
This repository forms the solution by *Anton Drasbæk Schiønning (202008161)* to assignment 3 in the course "Visual Analytics" at Aarhus University.

It trains a convelutional neural network, based on [VGG16](https://www.mathworks.com/help/deeplearning/ref/vgg16.html), for the task of classifying garnement types from the [Indofasion dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). This dataset contains over 106,000 images of 15 cloth categories originating from Indian ethnic clothes. <br/><br/> 

## Repository Tree <a name="tree"></a>
```
.
├── README.md
├── assign_desc.md              <---- original assignment description
├── images                      <---- the indofashion data (SHOULD BE DOWNLOADED AND PLACED HERE)
│   ├── metadata                    <---- metadata containing json files with labels for images
│   ├── test                        <---- 7500 testing images
│   ├── train                       <---- 91166 training images
│   └── val                         <---- 7500 validation images
├── out
│   ├── model_card.txt              <---- model architecture
│   ├── classification_report.txt   <---- classification report for model
│   └── loss.png                    <---- training/validation loss plot
├── requirements.txt
├── run.sh
└── src
    └── main.py                     <---- script for building, training and testing model.
```
## Usage <a name="usage"></a>
Due to the size of the dataset, the computational demand is heavy and the usage pipeline has therefore only been tested on cloud computing. Specifically, it has been tested on [Ucloud](https://cloud.sdu.dk/app/dashboard) which used Ubuntu v22.10 as the operating system. Please note that *venv* from Python must also be installed if you wish to run the analysis on *Ucloud*.

### Setup
Firstly, the Indofashion dataset should be downloaded from Kaggle [here](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). This dataset is 3 GB so this make take some time. <br>

The downloaded folder must be named `images` and contain four subdirectories: `metadata`, `train`, `test` and `val`. <br> 
These should be the default names, but please ensure that they are correct and place this folder directly into the root directory (see *Repository Tree* if in doubt where). <br/><br/>

### Run Analysis
With the folder placed, you can run the full analysis with the bash script as such:
```
bash run.sh
```
This achieves the following:
* Sets up a virtual environment
* Installs requirements to this environment
* Builds, trains and tests pretrained CNN model on the data (`main.py`)
* Deactivates virtual environment

The classification report and training/validation loss plot will be saved to the `out` directory. <br/><br/>

## Model Description <a name="model"></a>
The model was constructed using *VGG16* without its top layer to gain embeddings of images. This was fed to a 128 node hidden layer, then to a 64 node layer (both relu activation) and finally the output layer representing the 15 classes. For the full model architecture, please check `model_card.txt` in the `out` directory.

The training duration was set for a maximum of 10 epochs, but an early callback function was defined that would restore best weights from earlier if no improvements to the validation loss occured for two consequtive epochs. <br>

As seen from the loss plot, the model training was stopped after 8 epochs, restoring weights from the sixth epoch as those proved the highest. <br/><br/>
![alt text](https://github.com/AU-CDS/assignment3-pretrained-cnns-drasbaek/blob/main/out/loss.png?raw=True)

## Results <a name="results"></a>
This final model achieved an overall **F1-score of 0.84** on the test data. <br>
| Class                | Precision | Recall | F1-Score | Support |
|----------------------|-----------|--------|----------|---------|
| Blouse               | 0.95      | 0.97   | 0.96     | 500     |
| Dhoti_Pants          | 0.89      | 0.66   | 0.76     | 500     |
| Dupattas             | 0.81      | 0.71   | 0.76     | 500     |
| Gowns                | 0.75      | 0.52   | 0.62     | 500     |
| Kurta_men            | 0.88      | 0.90   | 0.89     | 500     |
| Leggings_and_salwars | 0.69      | 0.88   | 0.77     | 500     |
| Lehenga              | 0.93      | 0.92   | 0.93     | 500     |
| Mojaris_men          | 0.87      | 0.88   | 0.88     | 500     |
| Mojaris_women        | 0.88      | 0.87   | 0.87     | 500     |
| Nehru_jackets        | 0.94      | 0.94   | 0.94     | 500     |
| Palazzos             | 0.96      | 0.76   | 0.84     | 500     |
| Petticoats           | 0.95      | 0.89   | 0.92     | 500     |
| Saree                | 0.84      | 0.94   | 0.89     | 500     |
| Sherwanis            | 0.90      | 0.91   | 0.91     | 500     |
| Women_kurta          | 0.58      | 0.89   | 0.70     | 500     |
| Accuracy             |           |        | 0.84     | 7500    |
| Macro Avg            | 0.85      | 0.84   | 0.84     | 7500    |
| Weighted Avg         | 0.85      | 0.84   | 0.84     | 7500    |

## Discussion <a name="discuss"></a>
The high F1-score reveal that the pretrained CNN was able to fit to the data and identify patterns related to garnment type. This result is just slightly lower than the F1-score of 0.88 for the best model in the original paper on the dataset (Rajput & Aneja, 2021). <br>

Interestingly, *blouses* were almost identified correctly on all instances with an F1 of 0.96. Contrarily, the model struggled with *Women Kurtas*. The low precision for this class suggests that there were many false positives, i.e. many garnements were suggested to be *Woman Kurtas* when they were not. As noted by the Rajput & Aneja (2021), this likely pertains to the fact that *Woman Kurtas* and *Gowns* are visually similar.

## References
* Singh Rajput, P., & Aneja, S. (2021). IndoFashion: Apparel Classification for Indian Ethnic Clothes. arXiv e-prints, arXiv-2104.


