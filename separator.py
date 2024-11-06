from speechbrain.inference.separation import SepformerSeparation as separator
import soundfile as sf


run_opts = {"device": "cuda","data_parallel_count": -1,"data_parallel_backend": False,"distributed_launch": False,"distributed_backend": "nccl","jit_module_keys": None}

# sep_model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')
sep_model = separator.from_hparams(source="speechbrain/sepformer-libri3mix", savedir='pretrained_models/sepformer-libri3mix',run_opts=run_opts)

def separate_vocals(audio):
    est_sources = sep_model.separate_file(audio)
    return est_sources
    # sf.write("test_0.wav", est_sources[:, :, 0].detach().cpu().squeeze(), 8000, subtype='PCM_16')
    # sf.write("test_1.wav", est_sources[:, :, 1].detach().cpu().squeeze(), 8000, subtype='PCM_16')