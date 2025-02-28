# Lion vs Tiger Image Classification # 
This project is a Deep Learning-based image classification system to detect whether an image contains a lion or a tiger. The model is trained using Convolutional Neural Networks (CNNs) and deployed using Streamlit for an easy-to-use web interface.

## Dataset Structure ##
The dataset is not stored in this repository. Please place your images in the following folder structure:
lion/: Contains images of lions.
tiger/: Contains images of tigers.

## Installation & Setup ##
1. Clone this repository:
   git clone https://github.com/Simra-Kazi/Deep-Learning/tree/main/Lion%20and%20tiger%20Project%20Using%20Deep%20Learning
   cd Deep-Learning

2. Install dependencies:
  pip install -r requirements.txt

3.Train the model
  python train_lion_tiger_model.py

4. Run the Streamlit app
   streamlit run app.py

## Usage ##
Open the Streamlit app in your browser.
Upload an image (JPG, JPEG, PNG).
The model will predict if the image contains a Lion or a Tiger, along with a confidence score.

## Model Details ##
Attribute	                  Details
Model Type	                Convolutional Neural Network (CNN)
Framework	                  TensorFlow / Keras
Input Size	                128x128 pixels (resized for consistency)
Output	                    Binary classification - Lion or Tiger

## File Descriptions ##
File	                                  Description
train_lion_tiger_model.py              	Python script to train the CNN model
app.py	                                Streamlit app for uploading images and getting predictions
requirements.txt	                      Python dependencies
lion_tiger_model.h5	                    Saved trained model (not uploaded to GitHub)
README.md	                              Project documentation

## Technologies Used ##
Python
TensorFlow/Keras
Streamlit
NumPy

## Future Improvements ##
Expand the dataset with more diverse images.
Implement data augmentation during training.
Add explainability features (e.g., Grad-CAM for heatmaps).
Deploy to Streamlit Community Cloud for online access.
Explore transfer learning with pre-trained models like VGG16, ResNet, etc

## Author ##
https://github.com/Simra-Kazi

## License ##
This project is licensed under the MIT License 
