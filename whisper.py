from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
from torchaudio.transforms import Resample
from fastapi import APIRouter
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq ,AutoTokenizer ,AutoModelForSequenceClassification

processor = AutoProcessor.from_pretrained("steja/whisper-small-persian")
model = AutoModelForSpeechSeq2Seq.from_pretrained("steja/whisper-small-persian")
router = APIRouter()


app = FastAPI()
def transcribe_audio(waveform, sample_rate):
    if sample_rate != 16000:
        resampler = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    input_features = processor(waveform[0], sampling_rate=sample_rate).input_features[0]
    input_features = torch.tensor([input_features])
    generated_ids = model.generate(
        input_features,
        max_length=225, 
        num_beams=5,  
    )
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return transcription


audio_tensor, sample_rate = torchaudio.load('shortvoice.wav')
    
transcription = transcribe_audio(audio_tensor, sample_rate)
print(transcription)