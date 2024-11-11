
# AI Vision Companion ( [F5-TTS fork for convenience](https://github.com/SWivid/F5-TTS) )

This repository features an AI vision companion/assistant that merges visual input capture with audio transcription and synthesis through various APIs and libraries. The script detects microphone input, transcribes it, processes vision input from the specified window, creates very detailed caption with Florence-2, and produces responses using a Large Language Model (OpenAI API) and Text-To-Speech (F5-TTS).

## Features

- Near real-time interaction.
- Multiple monitor support.
- Captures and processes vision locally from a specified window.
- Transcribes audio input locally using Whisper-Large-3-Turbo model.
- Synthesizes responses locally using F5 text-to-speech model.
- Support for GPU acceleration using CUDA.

## Installation

### Prerequisites

- Windows OS
- Python 3.10 or higher
- CUDA-compatible GPU (NVIDIA RTX 3090/4090 recommended for faster processing)
- Microphone set as the default input device in the system settings.

### Requirements
The following environment configuration was used for testing: Windows 10 Pro x64, Python 3.10.11 64-bit, and CUDA 11.8.

Create a python 3.10 conda env (you could also use virtualenv)
```
conda create -n visioncompanion python=3.10
conda activate visioncompanion
```

First make sure you have git-lfs installed (https://git-lfs.com)
```
git lfs install
```
Install F5-TTS using the command below. You can install F5-TTS in a different directory using `cd` command (e.g., `cd c:`).
```
git clone https://github.com/Vinventive/F5-TTS-AIVisionCompanion.git
cd F5-TTS-AIVisionCompanion
pip install -e .
```

Download the F5 `model_1200000.safetensors` model checkpoint and place it inside `ckpts` folder 

(e.g., C:/F5-TTS-AIVisionCompanion/ckpts/F5TTS_Base/model_1200000.safetensors): 

[https://huggingface.co/SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS/blob/main/F5TTS_Base/model_1200000.safetensors)

Install torch with your CUDA version, e.g. :
```bash
pip install torch==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.18.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

### Required Environment Variables

Rename the `.env.example` file to `.env` and keep it in the root directory of the project. Fill in the missing `OPENAI_API_KEY=xxxxxxxx...` variable with your own API key, voice sample and voice sample transcription you can leave unchanged.
Use 2-15 seconds long voice sample in .wav format. If you swap a voice sample make sure to update filename in `F5_TTS_SAMPLE` and write an audio transcription for the new sample in `F5_TTS_REF_TEXT`. 

```
OPENAI_API_KEY=your_openai_api_key

F5_TTS_PATH="./..."
F5_TTS_SAMPLE="./sample.wav"
F5_TTS_REF_TEXT="How introverts make friends?"

VISION_KEYWORDS=scene,sight,video,frame,activity,happen,going
```

## Usage

### 1. Run the main script.
```
python visioncompanion.py
```
When running the script for the first time, it might take a while to download the `faster-whisper-large-v3` model for local use.

### 2. Choose if you want to launch the app with vision capture preview window. ('y' for yes or 'n' for no)
```
Using cuda:0 device
Loading Florence-2 model...
Florence-2 model loaded.
Do you want to launch the application with the preview window? (y/n):
```
### 2. Type the window title.
```
Do you want to launch the application with the preview window? (y/n): y
Enter the title of the window:
```
The script will prompt you to enter the title of the window you want to capture. 
You can specify a window by typing a simple keyword like `calculator` or `minecraft` etc. Searching process is not case-sensitive and will look for windows containing the provided keyword, even if the keyword is an incomplete word, prefix, or suffix. If you want to capture the view from your web browser's window and switch between different tabs, you can use simple keywords like `chrome` or `firefox` etc. In case you have multiple instances of an app open on multiple displays and you want to specify the tab use keywords like `youtube` or `twitch` etc. 

Make sure the captured window is always in the foreground and active - not minimized or in the background. (In case the window is minimized, the script will attempt to maximize it.)

### 3. Wait until the vision capture completes collecting the initial sequence of frames and speech recognition becomes active.
```
Enter the title of the window: youtube
Window title set to: youtube
Starting continuous audio recording...
```

### 4. Start by speaking into your microphone :)
```
Starting continuous audio recording...
User: Hi there, how are you doing?
```
### 5. Seamlessly talk about your view by naturally using vision-activating keywords during the conversation.
```
User: Hi there, how are you doing?
Assistant: Just hanging out, enjoying the vibes.
Converting audio...
User:Can you describe what you can see on the screen?
Assistant: There's an anime girl with long blonde hair, looking stylish in a red dress.
Converting audio...
User: How can I say it in Spanish?
Assistant: "Chica con cabello rubio y vestido rojo." Simple and stylish!
Converting audio...
```

Keywords analyzing the sequence of the last 10 seconds.
```
"scene", "sight",  "video", "frame", "activity", "happen", "going"
```
You have the option to add or remove keywords in the `.env` file.

## Acknowledgements
- [AIVisionCompanion](https://github.com/Vinventive/AIVisionCompanion) Original Repository for AIVisionCompanion.
- [Test voice sample credit](https://www.youtube.com/@miminggu) (used here only for testing) This voice sample comes from a random meme I've found on internet by author miminggu. I Don't remember which one exactly.


- [E2-TTS](https://arxiv.org/abs/2406.18009) brilliant work, simple and effective
- [Emilia](https://arxiv.org/abs/2407.05361), [WenetSpeech4TTS](https://arxiv.org/abs/2406.05763) valuable datasets
- [lucidrains](https://github.com/lucidrains) initial CFM structure with also [bfs18](https://github.com/bfs18) for discussion
- [SD3](https://arxiv.org/abs/2403.03206) & [Hugging Face diffusers](https://github.com/huggingface/diffusers) DiT and MMDiT code structure
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) as ODE solver, [Vocos](https://huggingface.co/charactr/vocos-mel-24khz) as vocoder
- [FunASR](https://github.com/modelscope/FunASR), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [UniSpeech](https://github.com/microsoft/UniSpeech) for evaluation tools
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) for speech edit test
- [mrfakename](https://x.com/realmrfakename) huggingface space demo ~
- [f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx/tree/main) Implementation with MLX framework by [Lucas Newman](https://github.com/lucasnewman)
- [F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) ONNX Runtime version by [DakeQQ](https://github.com/DakeQQ)

## Citation
If our work and codebase is useful for you, please cite as:
```
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```
## License

Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license due to the training data Emilia, which is an in-the-wild dataset. Sorry for any inconvenience this may cause.
