# Emotion Recognition In Persian Speech Using Deep Neural Networks 

This project focuses on recognizing emotions from speech by leveraging both acoustic features and textual features extracted from audio data. The Shemo dataset is used to train and evaluate our models.

## DataSet(Sharif Emotional Speech Database (ShEMO))
the ShEMO dataset contains 3000 audio files along with 3000 text files for each sentence as a ground-truth transcription. The text file of the sentence related to the corresponding audio file can be found through the names of the files. In fact, the audio and the text file of an utterance have the same name. But out of 3000 files, only 2838 have the same name. With further investigations, we found that some of these text files have the wrong names and referred to the wrong audio file. In the picture below, examples of errors in referencing audio and text files can be seen. We fixed the errors and inconsistencies in ShEMO dataset by using an Automatic Speech Recognition (ASR) system.


## Project Structure

- **shemo**: Directory where the Shemo dataset should be placed after download.
- **opensmile_preprocessing**: Contains the scripts for feature extraction using the openSMILE toolkit.
- **whisper**: Contains the scripts for converting audio to text using the Whisper model.
- **model.py**: The core file where the neural network model is defined and trained for emotion recognition.
- **TEST_Modified_ShEMO**: Script to test or evaluate the model.
- **inference**: A file that predicts the use of the model and the feeling of the sound

## Setup Instructions


1. **Install dependencies**:
   ```bash
   pip freeze > requirements.txt



2. **Download the Shemo Dataset**:
   - Place the dataset inside the `shemo` directory.

3. **Run Feature Extraction**:
   - Ensure that the openSMILE and Whisper tools are set up and configured properly in their respective directories.

4. **Execute the Test Script**:
   - Run the test script to verify that everything is set up correctly.
   ```bash
   python TEST_Modified_ShEMO.py


5. **Train the Model**:
   - To train the model, run the following command
   ```bash
   python model.py

6. **Predict Emotion**:
   - Use the following command to detect the emotion
   -Instead of this line
    Upload your audio => audio_file = 'shortvoice.wav'
   ```bash
   python inference.py


## Objective
The goal of this project is to detect emotions from speech by combining acoustic features and textual features extracted from audio files using the Shemo dataset.

## Dependencies
Python 3.x
openSMILE Toolkit
Whisper Model
Other dependencies as mentioned in requirements.txt



### contact
esmaeilimobina98@gmail.com