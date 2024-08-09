import json
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from opensmile_preprocessing import extract_features
import soundfile as sf
from whisper import transcribe_audio
import torchaudio
import torch
import numpy as np
import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output

def extract_bert_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors='tf', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.numpy().mean(axis=1)

# Load the model
model = tf.keras.models.load_model('multimodal_emotion_recognition_model.keras', custom_objects={'MultiHeadAttention': MultiHeadAttention})

model = tf.keras.models.load_model('multimodal_emotion_recognition_model.keras', custom_objects={'MultiHeadAttention': MultiHeadAttention})
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')


emo_labels = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry"
}

def predict_emotion(audio_path, transcript):
    # Extract MFCC features from the audio file
    mfcc_features = extract_features(audio_path)
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  

    # Extract BERT embeddings from the transcript
    bert_embeddings = extract_bert_embeddings([transcript], tokenizer, bert_model)  
    bert_embeddings = np.expand_dims(bert_embeddings, axis=0)  


    # Make the prediction
    predictions = model.predict([mfcc_features, bert_embeddings])


    emotion_labels = ['anger', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']  
    predicted_emotion = emotion_labels[np.argmax(predictions)]
    return predicted_emotion

# Example usage
audio_file = 'shortvoice.wav'
audio_tensor, sample_rate = torchaudio.load(audio_file)
transcript = transcribe_audio(audio_tensor, sample_rate)

predicted_emotion = predict_emotion(audio_file, transcript)
print(f'The predicted emotion is: {predicted_emotion}')
