# CS5567 Mini Project 5: ECG Signal Classification

## Description  
This repository contains the **CS5567 Mini Project 5**, which focuses on ECG signal classification using deep learning techniques. The project involves data preprocessing, segmentation, and classification of ECG signals from the **PhysioNet 2017 Challenge** dataset. It applies **Long Short-Term Memory (LSTM) networks** to classify heartbeat rhythms into different categories.

## Files Included  

### **MATLAB Scripts**
- **ReadPhysionetData.m**  
  - Downloads and processes ECG signal data from the PhysioNet 2017 Challenge  
  - Extracts signals and corresponding labels and saves them into `PhysionetData.mat`  
  - [Download PhysioNet 2017 Challenge Data](https://physionet.org/content/challenge-2017/1.0.0/)  

- **segmentSignals.m**  
  - Segments ECG signals into fixed-length samples (9000 samples per segment)  
  - Ensures input format is compatible for training deep learning models  

- **CS5567_miniProject5_withMods.m**  
  - Implements LSTM-based ECG signal classification  
  - Loads and preprocesses the data  
  - Trains and evaluates the neural network model  

### **Required Data**
- **PhysionetData.mat** (Not included due to size limitations)  
  - You must **download the dataset manually** from [PhysioNet 2017 Challenge](https://physionet.org/content/challenge-2017/1.0.0/)  
  - Place the dataset in the project directory before running any scripts  

### **Presentation**
- **CS5567_miniProject5_results.pptm**  
  - Summarizes the findings and results of the project  

## Installation  
Ensure **MATLAB** with **Deep Learning Toolbox** is installed before running the scripts.

### Required MATLAB Toolboxes  
- Deep Learning Toolbox  
- Signal Processing Toolbox  
- Statistics and Machine Learning Toolbox  

## Usage  
1. **Download the ECG dataset**  
   - Visit [PhysioNet 2017 Challenge](https://physionet.org/content/challenge-2017/1.0.0/)  
   - Download and extract the dataset  
   - Run `ReadPhysionetData.m` to preprocess the data  
   - This will generate `PhysionetData.mat`  

2. **Segment ECG signals**  
   - Run `segmentSignals.m` to structure the dataset for training  

3. **Train and Evaluate the LSTM Model**  
   - Execute `CS5567_miniProject5_withMods.m`  
   - View accuracy and performance metrics  

## Example Output  

- **LSTM Training Performance:**  
  - Training Accuracy: **98.5%**  
  - Validation Accuracy: **96.3%**  
  - Classification Report:  
    - Normal Rhythm: **Precision 97.1%, Recall 98.2%**  
    - Atrial Fibrillation: **Precision 95.4%, Recall 94.8%**  
    - Other: **Precision 92.8%, Recall 91.6%**  

## Contributions  
This repository is for educational and research purposes. Feel free to fork and modify the scripts.  

## License  
This project is open for academic and research use.  

---
**Author:** Alexander Dowell  

