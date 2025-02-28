## Glasses Detection Using Deep Learning ##
This project is a Deep Learning-based image classification system to detect whether a person is wearing glasses or not. The model is trained using Convolutional Neural Networks (CNNs) and deployed using Streamlit for an easy-to-use web interface.

.
├── data/                     # (Not included in GitHub - local dataset directory)
│   ├── with_glasses/         # Images of people with glasses
│   ├── without_glasses/      # Images of people without glasses
├── model/                     # Saved trained model (glasses_model.h5) - Not uploaded to GitHub
├── app.py                     # Streamlit app for image upload and classification
├── train_glasses_model.py     # Training script for CNN model
├── requirements.txt           # Required libraries
└── README.md                  # Project documentation

The dataset is not stored in this repository.
Please place your images in the following folder structure:
data/
├── with_glasses/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── without_glasses/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...

with_glasses/: Contains images of people wearing glasses.
without_glasses/: Contains images of people without glasses.

## Installation & Setup ##
1.Clone this repository:
git clone https://github.com/Simra-Kazi/Deep-Learning.git
cd Deep-Learning

2.Install dependencies:
pip install -r requirements.txt

3.Train the model :
python glasses.py

4.Run the Streamlit app:
streamlit run app.py

## Usage ##
Open the Streamlit app in your browser.
Upload an image (JPG, JPEG, PNG).
The model will predict if the person is wearing glasses or not, along with a confidence score.

## Model Details ##
Model Type: Convolutional Neural Network (CNN)
Framework: TensorFlow / Keras
Input Size: 100x100 pixels (resized for consistency)
Output: Binary classification - With Glasses or Without Glasses

## File Descriptions ##
File                                 	Description
train_glasses_model.py	              Python script to train the CNN model
app.py	                              Streamlit app for uploading images and getting predictions
requirements.txt	                    Python dependencies
glasses_model.h5	                    Saved trained model (not uploaded to GitHub)
README.md	                            Project documentation

## Technologies Used ##
Python
TensorFlow/Keras
Streamlit
Pillow (PIL)
NumPy

## Future Improvements ##
Expand the dataset with more diverse faces.
Implement data augmentation during training.
Add explainability features (e.g., Grad-CAM for heatmaps).
Deploy to Streamlit Community Cloud for online access.

## Author ##
https://github.com/Simra-Kazi

## License ##
This project is licensed under the MIT License - see LICENSE for details.

