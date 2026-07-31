from io import BytesIO
import os
import time
import json
import sys
import argparse

import scipy
import numpy as np
import torch
import pysbd
from tqdm import tqdm
import librosa

from xtts import Xtts, XTTSConfig

import transformers

transformers.logging.set_verbosity_error()

def save_wav(
    *, wav: np.ndarray, path: str, sample_rate: int = None, pipe_out=None, **kwargs
) -> None:
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

    wav_norm = wav_norm.astype(np.int16)
    if pipe_out:
        wav_buffer = BytesIO()
        scipy.io.wavfile.write(wav_buffer, sample_rate, wav_norm)
        wav_buffer.seek(0)
        pipe_out.buffer.write(wav_buffer.read())
    scipy.io.wavfile.write(path, sample_rate, wav_norm)


def trim_silence(wav, ap):
    return wav[: ap.find_endpoint(wav)]


def interpolate_vocoder_input(scale_factor, spec):
    print(" > before interpolation :", spec.shape)
    spec = torch.tensor(spec).unsqueeze(0).unsqueeze(0)  # pylint: disable=not-callable
    spec = torch.nn.functional.interpolate(
        spec,
        scale_factor=scale_factor,
        recompute_scale_factor=True,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
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


class Synthesizer:
    def __init__(self, tts_checkpoint, tts_config_path, language, use_cuda=False):
        super().__init__()
        self.tts_checkpoint_dir = tts_checkpoint
        self.tts_config_path = tts_config_path
        self.use_cuda = use_cuda
        self.tts_model = None
        self.language_manager = None
        self.num_languages = 0
        self.tts_languages = {}
        self.d_vector_dim = 0
        self.seg = self._get_segmenter(language)

        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA is not availabe on this machine."

        self._load_tts(tts_checkpoint, tts_config_path, use_cuda)
        self.output_sample_rate = self.tts_config.audio["sample_rate"]

    @staticmethod
    def _get_segmenter(lang: str):
        return pysbd.Segmenter(language=lang, clean=True)

    def _load_tts(
        self, tts_checkpoint_dir: str, tts_config_path: str, use_cuda: bool
    ) -> None:
        self.tts_config = load_config(tts_config_path)
        self.tts_model = Xtts(self.tts_config)
        self.tts_model.load_checkpoint(tts_checkpoint_dir, eval=True)
        if use_cuda:
            self.tts_model.cuda()

    def split_into_sentences(self, text) -> list[str]:
        return self.seg.segment(text)

    def tts(
        self,
        text: str,
        language: str,
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

        sens_tqdm = tqdm(sens, desc="Synthesizing")
        for sen in sens_tqdm:
            outputs = self.tts_model.synthesize(
                text=sen,
                config=self.tts_config,
                speaker_wav=speaker_wav,
                language=language,
                **kwargs,
            )
            waveform = outputs["wav"]
            if not use_gl:
                mel_postnet_spec = (
                    outputs["outputs"]["model_outputs"][0].detach().cpu().numpy()
                )
                # denormalize tts output based on tts audio config
                mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
                # renormalize spectrogram based on vocoder config
                vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                # compute scale factor for possible sample rate mismatch
                scale_factor = [
                    1,
                    self.vocoder_config["audio"]["sample_rate"]
                    / self.tts_model.ap.sample_rate,
                ]
                if scale_factor[1] != 1:
                    print(" > interpolating tts model output.")
                    vocoder_input = interpolate_vocoder_input(
                        scale_factor, vocoder_input
                    )
                else:
                    vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                # run vocoder model
                # [1, T, C]
                waveform = self.vocoder_model.inference(
                    vocoder_input.to(vocoder_device)
                )
            if (
                torch.is_tensor(waveform)
                and waveform.device != torch.device("cpu")
                and not use_gl
            ):
                waveform = waveform.cpu()
            if not use_gl:
                waveform = waveform.numpy()
            waveform = waveform.squeeze()

            # trim silence
            if (
                "do_trim_silence" in self.tts_config.audio
                and self.tts_config.audio["do_trim_silence"]
            ):
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
        save_wav(
            wav=new_wav,
            path=path,
            sample_rate=self.output_sample_rate,
            pipe_out=pipe_out,
        )


def read_text_file(file_path, max_lines=None):
    """Read and clean text from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        cleaned_lines = [line.strip() for line in lines if line.strip()]

        if max_lines:
            cleaned_lines = cleaned_lines[:max_lines]

        return "".join(cleaned_lines)
    except FileNotFoundError:
        print(f"Error: Text file '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading text file: {e}")
        sys.exit(1)


def validate_file_exists(file_path, file_type):
    """Validate that a file exists."""
    if not os.path.exists(file_path):
        print(f"Error: {file_type} file '{file_path}' not found.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="TTS Console Application with time stretching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      python tts_app.py --text "Hello world" --speaker speaker.wav --output output.wav
      python tts_app.py --file input.txt --speaker speaker.wav --duration 15.0 --output output.wav
      python tts_app.py --text "你好世界" --speaker speaker.wav --language zh-cn --output output.wav
            """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", "-t", type=str, help="Text to synthesize directly"
    )
    input_group.add_argument(
        "--file", "-f", type=str, help="Path to text file to read from"
    )

    parser.add_argument(
        "--speaker",
        "-s",
        type=str,
        required=True,
        help="Path to speaker reference audio file (WAV format)",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output audio file path"
    )

    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        help="Target duration in seconds (enables time stretching)",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="zh",
        help="Language code (default: zh-cn)",
    )

    parser.add_argument(
        "--max-lines", type=int, help="Maximum number of lines to read from file"
    )

    parser.add_argument(
        "--no-cuda", action="store_false", help="Disable CUDA acceleration"
    )
    parser.add_argument(
        "--split-sentences",
        action="store_true",
        default=True,
        help="Split text into sentences for processing (default: True)",
    )

    parser.add_argument(
        "--ratio",
        type=float,
        help="Speed ratio to apply to the output audio (e.g., 0.5 = slower, 2 = faster)",
    )

    args = parser.parse_args()

    if args.file:
        validate_file_exists(args.file, "Text")
    validate_file_exists(args.speaker, "Speaker audio")

    if args.text:
        text = args.text
        print(f"Using direct text input: {text[:50]}{'...' if len(text) > 50 else ''}")
    else:
        text = read_text_file(args.file, args.max_lines)
        print(f"Read text from file: {text[:50]}{'...' if len(text) > 50 else ''}")

    if not text.strip():
        print("Error: No text to synthesize.")
        sys.exit(1)

    try:
        print("Initializing TTS synthesizer...")

        syn = Synthesizer(
            tts_checkpoint="models",
            tts_config_path="models/config.json",
            language=args.language,
            use_cuda=not args.no_cuda,
        )

        print("Synthesizing speech...")
        wav = syn.tts(
            text=text,
            language=args.language,
            speaker_wav=args.speaker,
            split_sentences=args.split_sentences,
        )

        wav_array = np.array(wav, dtype=np.float32)
        sample_rate = syn.output_sample_rate

        print(
            f"Generated audio: {len(wav_array)/sample_rate:.2f} seconds at {sample_rate} Hz"
        )

        final_wav = wav_array

        if args.duration and args.ratio:
            print("Error: You cannot use both --duration and --ratio at the same time.")
            sys.exit(1)

        if args.duration:
            current_duration = len(wav_array) / sample_rate
            stretch_ratio = args.duration / current_duration
            print(
                f"Time stretching to match duration: {current_duration:.2f}s -> {args.duration:.2f}s (ratio: {stretch_ratio:.3f})"
            )
            final_wav = librosa.effects.time_stretch(wav_array, rate=1 / stretch_ratio)

        elif args.ratio:
            print(f"Time stretching using speed ratio: {args.ratio}")
            final_wav = librosa.effects.time_stretch(wav_array, rate=args.ratio)

        # if args.duration:
        #     current_duration = len(wav_array) / sample_rate
        #     stretch_ratio = args.duration / current_duration
        #
        #     print(
        #         f"Time stretching: {current_duration:.2f}s -> {args.duration:.2f}s (ratio: {stretch_ratio:.3f})"
        #     )
        #
        #     stretched_wav = librosa.effects.time_stretch(
        #         wav_array, rate=1 / stretch_ratio
        #     )
        #     final_wav = stretched_wav
        # else:
        #     final_wav = wav_array

        print(f"Saving audio to: {args.output}")
        syn.save_wav(wav=final_wav, path=args.output)

        final_duration = len(final_wav) / sample_rate
        print(f"Successfully created audio file: {final_duration:.2f} seconds")

    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please ensure all required libraries are installed:")
        print("- TTS library (Coqui TTS or your custom implementation)")
        print("- librosa")
        print("- numpy")
        sys.exit(1)
    except Exception as e:
        print(f"Error during synthesis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
