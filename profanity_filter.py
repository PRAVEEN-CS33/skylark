from profanity_check import predict
from scipy.io.wavfile import write
from pydub import AudioSegment
import numpy as np
import whisper
import torch
import os


def beep(duration):
    sps = 44100
    freq_hz = 1000.0
    vol = 0.3
    esm = np.arange(duration * sps)
    wf = np.sin(2 * np.pi * esm * freq_hz / sps)
    wf_quiet = wf * vol
    wf_int = np.int16(wf_quiet * 32767)
    beep_audio = AudioSegment(
        wf_int.tobytes(), 
        frame_rate=sps,
        sample_width=wf_int.dtype.itemsize, 
        channels=1
    )
    return beep_audio

def profanity_filter_add(audio_tensor: torch.Tensor,srate):
    audio = AudioSegment(audio_tensor.tobytes(), 
    frame_rate=srate,
    sample_width=audio_tensor.dtype.itemsize, 
    channels=1)
    model = whisper.load_model("small")
    transcription = model.transcribe(audio_tensor, word_timestamps=True, language=None)
    profane_intervals = []

    for segment in transcription['segments']:
        for word in segment['words']:
            if predict([word['word']])[0] == 1:
                start_time = int(word['start'] * 1000)
                end_time = int(word['end'] * 1000)
                profane_intervals.append((start_time, end_time))

    for start_time, end_time in profane_intervals:
        beep_sound = beep(duration=(end_time - start_time) / 1000.0)
        audio = audio[:start_time] + beep_sound + audio[end_time:]

    # output_file_path = "filtered_sample.mp3"
    # audio.export(output_file_path, format="mp3")

    # print(f"Filtered audio saved as {output_file_path}")
    return audio, transcription

if __name__ == "__main__":
    profanity_filter_add("fk.mp3")

