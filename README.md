# arxiv-podcast
Axion and Neutrino robot podcast for a daily arXiv digest

To add this podcast to your podcast app, please add the URL https://sumitaghosh.github.io/arxiv-podcast/feed.xml to your RSS reader! If you want a suggestion, I like Overcast because it's super simple: https://overcast.fm/

If you want to make your own, please use the python script and change it however you like! Right now it looks for mentions of axions and neutrinos, and it has toggles to turn on generative AI and emailing that are both set to False. But you can change the key words and add in your own domain and email and API keys to make your own podcast.

## Setup

### 1. Create and activate a conda environment

```bash
conda create -n arxiv-podcast python=3.12
conda activate arxiv-podcast
```

> Note: Python 3.13+ is not yet supported. Use 3.10–3.12.

### 2. Install Python dependencies

```bash
pip install requests
```

### 3. Install ffmpeg (for audio concatenation)

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install -y ffmpeg
```

### 4. Run the script

```bash
python make_podcast.py
```

That's it if you're happy using macOS voices! The script defaults to `TTS_MODE = "kokoro"` but you can set it to `"mac_say_ffmpeg"` in the CONFIGURATION section at the top of the script and skip the rest of this README.

---

## Optional: Kokoro TTS (free, local, better voices)

If you want better-sounding voices, keep reading. [Kokoro-82M](https://github.com/hexgrad/kokoro) is a free, open-source TTS model that runs entirely on your machine — no API key, no internet connection after the first download (~327MB), and noticeably more natural than the built-in macOS voices.

To use it, make sure `TTS_MODE = "kokoro"` in the CONFIGURATION section (this is already the default), then do the following:

### 1. Install the system dependency

**macOS:**
```bash
brew install espeak-ng
```

**Linux:**
```bash
sudo apt-get install -y espeak-ng
```

### 2. Install Kokoro Python packages

```bash
pip install kokoro soundfile torch
```

If you're on Apple Silicon and see PyTorch errors, set this before running:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

To make this permanent, add it to your `~/.zshrc` (or `~/.bashrc`) and run `source ~/.zshrc`.

### Voice options

Kokoro ships 54 built-in voices. The defaults are `am_adam` (HostA) and `af_bella` (HostB). You can change these in the `CONFIGURATION` section at the top of the script. Other good options for a science podcast:

- `bm_george` — British male
- `bf_emma` — British female  
- `am_michael` — American male
- `af_sarah` — American female

---

## Optional: Generative AI scripts

By default the script just reads out titles and abstracts. If you set `USE_AI = True` (or pass `--genAI` on the command line), it will use an LLM to write a proper two-host conversational script instead. This requires an API key from an OpenAI-compatible provider.

Set your key as an environment variable so you don't have to paste it into the script:

```bash
export AI_API_KEY="your-api-key-here"
```

To make this permanent, add that line to your `~/.zshrc` (or `~/.bashrc`) and run `source ~/.zshrc`.

---

## Running the script

```bash
python make_podcast.py                      # process yesterday's papers (default)
python make_podcast.py 2025-06-26           # process a specific date
python make_podcast.py --genAI              # use an LLM to write the script (requires API key)
python make_podcast.py 2025-06-26 --genAI  # specific date + AI script
```

Each run will:
1. Query arXiv for papers matching your keywords on the target date
2. Generate a two-host script (static by default, or AI-written with `--genAI`)
3. Synthesize audio and save an MP3 to the output directory
4. Update `feed.xml` with the new episode
5. Save a `.txt` copy of the script alongside the MP3
6. Commit and push everything to your GitHub Pages repo automatically

Papers that have already been processed are tracked in `processed_ids.json`, so re-running the same date won't produce duplicates.

---

## Configuration toggles

| Variable | Default | Description |
|---|---|---|
| `TTS_MODE` | `"kokoro"` | `"kokoro"` for free local voices, `"mac_say_ffmpeg"` for built-in macOS voices |
| `USE_AI` | `False` | Use an LLM to write the script (requires `AI_API_KEY`) |
| `ENABLE_EMAIL` | `False` | Email the script to yourself (requires `sendmail`) |
