# Speech Emotion Recognition using Shemo Dataset

This project focuses on recognizing emotions from speech by leveraging both acoustic features and textual features extracted from audio data. The Shemo dataset is used to train and evaluate our models.

## Project Structure

- **shemo/**: Directory where the Shemo dataset should be placed after download.
- **openSMILE**: Contains the scripts for feature extraction using the openSMILE toolkit.
- **whisper**: Contains the scripts for converting audio to text using the Whisper model.
- **model.py**: The core file where the neural network model is defined and trained for emotion recognition.
- **TEST_Modified_ShEMO.py**: Script to test or evaluate the model.
- **inference.py**: A file that predicts the use of the model and the feeling of the sound

## Setup Instructions

1. **Download the Shemo Dataset**:
   - Place the dataset inside the `shemo/` directory.

2. **Run Feature Extraction**:
   - Ensure that the openSMILE and Whisper tools are set up and configured properly in their respective directories.

3. **Execute the Test Script**:
   - Run the test script to verify that everything is set up correctly.
   ```bash
   python3 TEst

Train the Model:
To train the model, run the following command

python3 model.py

Objective
The goal of this project is to detect emotions from speech by combining acoustic features and textual features extracted from audio files using the Shemo dataset.

Dependencies
Python 3.x
openSMILE Toolkit
Whisper Model
Other dependencies as mentioned in requirements.txt (if applicable)