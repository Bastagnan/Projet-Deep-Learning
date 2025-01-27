# Projet-Deep-Learning

## Description
This project focuses on developing a deep learning model to recognize and decode CAPTCHAs. The model is trained using a dataset of CAPTCHA images and aims to accurately predict the text contained within these images.

## Notebook Overview
The `notebook.ipynb` file contains the following steps:

1. **Data Loading and Preprocessing**:
    - Load the CAPTCHA images and corresponding labels.
    - Preprocess the images by resizing and normalizing them.
    - Convert the labels into a format suitable for training the model.

2. **Model Architecture**:
    - Define several single output models such as:
        - Convolutional Neural Network (CNN) model
        - CNN - LSTM model
        - CRNN model
        - Transformer model
        - CNN with attention mechanism model
    - Implement CNN + RNN + Connectionist Temporal Classification (CTC) layer to handle the sequence prediction problem.
    - Implement a multi-output CNN with 5 output branches which outperforms classic single output models.

3. **Training**:
    - Compile the model with an appropriate optimizer and loss function.
    - Train the model on the preprocessed dataset.

4. **Evaluation**:
    - Evaluate the model's performance on a separate test set.
    - Calculate metrics such as accuracy and loss to assess the model's effectiveness.

5. **Prediction**:
    - Use the trained model to predict the text in new CAPTCHA images.
    - Display the predicted text alongside the actual CAPTCHA image for visual verification.

## References
- [Intuitively Understanding Connectionist Temporal Classification](https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c)
- [Keras CAPTCHA OCR Example](https://keras.io/examples/vision/captcha_ocr/)
- [Kaggle CAPTCHA Recognition](https://www.kaggle.com/code/shawon10/captcha-recognition)