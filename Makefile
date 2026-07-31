.PHONY: run

run:
	time uv run python tts_clone_inference/main.py \
		--text "The sailor agreed, following Bathardish down and dogging the hatch behind him. You never know when the barbarians are going to go nuts." \
		--speaker tmp/johnlee.wav \
		--output tmp/out1.wav
