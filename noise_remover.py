from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio

run_opts = {"device": "cuda","data_parallel_count": -1,"data_parallel_backend": False,"distributed_launch": False,"distributed_backend": "nccl","jit_module_keys": None}


def noise_removal_filter(curr_audio):
    """8k sr processing"""
    model = separator.from_hparams(source="speechbrain/sepformer-wham-enhancement", savedir='pretrained_models/sepformer-wham16k-enhancement',run_opts=run_opts)
    audio_sources = model.forward(curr_audio)
    # torchaudio.save("denoised.wav", audio_sources[:, :, 0], 8000)
    return audio_sources[:, :, 0]