# TTS-Clone-Inference: Clean Inference and Performance Improvements

This repository contains a modified version of a TTS from coqui-ai that focuses on optimizing the inference pipeline for **clean output** and **improved performance**.

    - parse inference pipeline without training or other parts

## Installation

    - use poetry to install dependencies

## Download models

- download model from `https://huggingface.co/bruceunx/tts-clone-inference/tree/main`

## change code in `main.py`

```python

tts = TTS(model_path="models", config_path="models/config.json", gpu=False)

tts.tts_to_file(text="hello world",
                file_path="./tmp/sample.wav",
                speaker_wav="./tmp/output.wav",
                enable_text_splitting=True,
                language="en")

```
