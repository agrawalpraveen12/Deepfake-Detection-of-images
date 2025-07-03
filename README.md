
#  Deepfake Detection using ResNet-18

##  Overview
This project is an **ongoing** deepfake detection system built using a fine-tuned **ResNet-18** architecture. The goal is to classify facial images as **Real** or **Fake**, helping combat misinformation with deep learning.

##  Problem Statement
With the rapid rise of AI-generated deepfake images, automated tools are required to detect manipulated faces.  
This project provides a **robust and scalable** solution utilizing **ResNet-18** for image classification.

##  Key Details
- **Dataset**: 10,000 real vs. fake facial images
- **Current Status**: Ongoing
- **Model**: ResNet-18 architecture
- **Training**: Applied data augmentation and fine-tuning techniques
- **Deployment**: Streamlit web app for real-time image classification

##  Features
- Automatic train/val/test split
- Extensive data augmentation for better model generalization
- Deep residual learning (ResNet-18) to avoid vanishing gradients
- Training & validation accuracy/metrics visualization
- Real-time detection interface via Streamlit
- Dropout and BatchNorm for regularization and stable training

##  Tech Stack
- **Python 3.x**
- **TensorFlow/Keras** for model training
- **NumPy, OpenCV, PIL** for image preprocessing
- **Streamlit** for web interface
- **Jupyter Notebook** for data exploration and training

## Workflow
1.  **Data Preparation**:
   - Split the raw data into train, validation, and test sets
   - Preprocessed and normalized all images
   - Applied data augmentation to increase diversity

2.  **Training**:
   - Implemented ResNet-18 with skip connections
   - Tuned hyperparameters to improve accuracy
   - Monitored training and validation metrics

3.  **Evaluation**:
   - Evaluated the model on the test set
   - Achieved a strong balance between precision and recall

4.  **Deployment**:
   - Saved the best performing model (`best_model.h5`)
   - Developed a Streamlit app for image upload and real-time predictions

##  Getting Started
###  Prerequisites
- Python 3.x
- GPU (optional but recommended for training)

