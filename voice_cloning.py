from TTS.api import TTS
import torch

#Get device
device="cuda:0" if torch.cuda.is_available() else "cpu"

SAMPLE_AUDIOS = ["LOTR.wav"]
OUTPUT_PATH = "reSpoken.wav"
TEXT = "आप कैसे हैं?"

#Init TTS with the target model name
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)

#Run TTS
tts.tts_to_file(TEXT,
               speaker_wav=SAMPLE_AUDIOS,
               language="en",
               file_path=OUTPUT_PATH,
               split_sentences=False
               )

# #Listen to the generated audio
# import IPython
# IPython.display.Audio(OUTPUT_PATH)