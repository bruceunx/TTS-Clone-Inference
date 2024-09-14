from io import BytesIO
import time
import json

import scipy
import numpy as np

import torch
import torch.nn as nn

import pysbd

from tqdm import tqdm
from xtts import Xtts, XTTSConfig


def save_wav(*,
             wav: np.ndarray,
             path: str,
             sample_rate: int = None,
             pipe_out=None,
             **kwargs) -> None:

    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

    wav_norm = wav_norm.astype(np.int16)
    if pipe_out:
        wav_buffer = BytesIO()
        scipy.io.wavfile.write(wav_buffer, sample_rate, wav_norm)
        wav_buffer.seek(0)
        pipe_out.buffer.write(wav_buffer.read())
    scipy.io.wavfile.write(path, sample_rate, wav_norm)


def trim_silence(wav, ap):
    return wav[:ap.find_endpoint(wav)]


def interpolate_vocoder_input(scale_factor, spec):
    print(" > before interpolation :", spec.shape)
    spec = torch.tensor(spec).unsqueeze(0).unsqueeze(0)  # pylint: disable=not-callable
    spec = torch.nn.functional.interpolate(spec,
                                           scale_factor=scale_factor,
                                           recompute_scale_factor=True,
                                           mode="bilinear",
                                           align_corners=False).squeeze(0)
    print(" > after interpolation :", spec.shape)
    return spec


def load_config(config_path: str):

    config_dict = {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    config_dict.update(data)
    config = XTTSConfig()
    config.from_dict(config_dict)
    return config


class Synthesizer(nn.Module):

    def __init__(self, tts_checkpoint="", tts_config_path="", use_cuda=False):
        super().__init__()
        self.tts_checkpoint_dir = tts_checkpoint
        self.tts_config_path = tts_config_path
        self.use_cuda = use_cuda
        self.tts_model = None
        self.language_manager = None
        self.num_languages = 0
        self.tts_languages = {}
        self.d_vector_dim = 0
        self.seg = self._get_segmenter("en")

        if self.use_cuda:
            assert torch.cuda.is_available(
            ), "CUDA is not availabe on this machine."

        if tts_checkpoint:
            self._load_tts(tts_checkpoint, tts_config_path, use_cuda)
            self.output_sample_rate = self.tts_config.audio["sample_rate"]

    @staticmethod
    def _get_segmenter(lang: str):
        return pysbd.Segmenter(language=lang, clean=True)

    def _load_tts(self, tts_checkpoint_dir: str, tts_config_path: str,
                  use_cuda: bool) -> None:

        self.tts_config = load_config(tts_config_path)
        self.tts_model = Xtts(self.tts_config)
        self.tts_model.load_checkpoint(tts_checkpoint_dir, eval=True)
        if use_cuda:
            self.tts_model.cuda()

    def split_into_sentences(self, text) -> list[str]:
        return self.seg.segment(text)

    def tts(
        self,
        text: str = "",
        language_name: str = "en",
        speaker_wav=None,
        split_sentences: bool = True,
        **kwargs,
    ) -> list[int]:
        start_time = time.time()
        wavs = []

        if text:
            sens = [text]
            if split_sentences:
                print(" > Text splitted to sentences.")
                sens = self.split_into_sentences(text)

        vocoder_device = "cpu"
        use_gl = True
        if not use_gl:
            vocoder_device = next(self.vocoder_model.parameters()).device
        if self.use_cuda:
            vocoder_device = "cuda"

        print(sens)
        sens_tqdm = tqdm(sens, desc="Synthesizing")
        for sen in sens_tqdm:
            outputs = self.tts_model.synthesize(
                text=sen,
                config=self.tts_config,
                speaker_wav=speaker_wav,
                language=language_name,
                **kwargs,
            )
            waveform = outputs["wav"]
            if not use_gl:
                mel_postnet_spec = outputs["outputs"]["model_outputs"][
                    0].detach().cpu().numpy()
                # denormalize tts output based on tts audio config
                mel_postnet_spec = self.tts_model.ap.denormalize(
                    mel_postnet_spec.T).T
                # renormalize spectrogram based on vocoder config
                vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                # compute scale factor for possible sample rate mismatch
                scale_factor = [
                    1,
                    self.vocoder_config["audio"]["sample_rate"] /
                    self.tts_model.ap.sample_rate,
                ]
                if scale_factor[1] != 1:
                    print(" > interpolating tts model output.")
                    vocoder_input = interpolate_vocoder_input(
                        scale_factor, vocoder_input)
                else:
                    vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                # run vocoder model
                # [1, T, C]
                waveform = self.vocoder_model.inference(
                    vocoder_input.to(vocoder_device))
            if torch.is_tensor(waveform) and waveform.device != torch.device(
                    "cpu") and not use_gl:
                waveform = waveform.cpu()
            if not use_gl:
                waveform = waveform.numpy()
            waveform = waveform.squeeze()

            # trim silence
            if "do_trim_silence" in self.tts_config.audio and self.tts_config.audio[
                    "do_trim_silence"]:
                waveform = trim_silence(waveform, self.tts_model.ap)

            wavs += list(waveform)
            wavs += [0] * 10000

        # compute stats
        process_time = time.time() - start_time
        audio_time = len(wavs) / self.tts_config.audio["sample_rate"]
        print(f" > Processing time: {process_time}")
        print(f" > Real-time factor: {process_time / audio_time}")
        return wavs

    def save_wav(self, wav: list[int], path: str, pipe_out=None) -> None:
        # if tensor convert to numpy
        new_wav = np.array(wav)
        save_wav(wav=new_wav,
                 path=path,
                 sample_rate=self.output_sample_rate,
                 pipe_out=pipe_out)


class TTS(nn.Module):

    def __init__(
        self,
        model_path: str = None,
        config_path: str = None,
        gpu=False,
    ):
        super().__init__()
        self.config = load_config(config_path) if config_path else None
        self.synthesizer: Synthesizer | None = None
        if model_path:
            self.load_tts_model_by_path(model_path, config_path, gpu=gpu)

    def load_tts_model_by_path(self,
                               model_path: str,
                               config_path: str = None,
                               gpu: bool = False):

        self.synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            use_cuda=gpu,
        )

    def tts(
        self,
        text: str,
        language: str = None,
        speaker_wav: str = None,
        split_sentences: bool = True,
        **kwargs,
    ):
        if self.synthesizer is None:
            return
        wav = self.synthesizer.tts(
            text=text,
            language_name=language or "en",
            speaker_wav=speaker_wav,
            split_sentences=split_sentences,
            **kwargs,
        )
        return wav

    def tts_to_file(
        self,
        text: str,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        file_path: str = "output.wav",
        split_sentences: bool = True,
        **kwargs,
    ):

        wav = self.tts(
            text=text,
            speaker=speaker,
            language=language,
            speaker_wav=speaker_wav,
            split_sentences=split_sentences,
            **kwargs,
        )
        if self.synthesizer is None:
            return
        self.synthesizer.save_wav(wav=wav, path=file_path)
        return file_path


if __name__ == "__main__":

    tts = TTS(model_path="models", config_path="models/config.json", gpu=False)
    # with open("./tmp/sample.txt", "r") as f:
    #     lines = f.readlines()
    #
    # cleaned_lines = [line.strip() for line in lines if line.strip()]
    #
    # text = "".join(cleaned_lines)

    tts.tts_to_file(text="hello world",
                    file_path="./tmp/sample1.wav",
                    speaker_wav="./tmp/output.wav",
                    enable_text_splitting=True,
                    language="en")
