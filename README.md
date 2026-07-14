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

If you want better-sounding voices, keep reading. [Kokoro-82M](https://github.com/hexgrad/kokoro) is a free, ethically-trained open-source TTS ~327MB model that runs entirely on your machine and sounds noticeably more natural than the built-in macOS voices.

To use it, make sure `TTS_MODE = "kokoro"` in the CONFIGURATION section (the default), then do the following:

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

By default the script just reads out titles and abstracts. If you set `USE_AI = True` (or pass `--genAI` on the command line), it will use an LLM to write a proper two-host conversational script instead. This requires an API key from an OpenAI-compatible chat-completions provider.

Set your key as an environment variable so you don't have to paste it into the script:

```bash
export AI_API_KEY="your-api-key-here"
```

To make this permanent, add that line to your `~/.zshrc` (or `~/.bashrc`) and run `source ~/.zshrc`.

### OpenAI-compatible provider settings

The script reads these AI settings from environment variables:

```bash
export AI_API_KEY="your-api-key-here"
export AI_BASE_URL="your-chat-completions-endpoint"
export AI_MODEL="your-model-name"
export AI_TEMPERATURE="0.4"
```

For example, if your provider uses a model named `gpt-5.5` and requires temperature `1`, run with:

```bash
AI_API_KEY="your-api-key-here" \
AI_BASE_URL="your-chat-completions-endpoint" \
AI_MODEL="gpt-5.5" \
AI_TEMPERATURE="1" \
python make_podcast.py 2026-06-15 --genAI
```

Do not commit real API keys to the repository. Prefer environment variables for secrets.

### Model temperature note

Some OpenAI-compatible providers accept only specific values for `temperature`. For example, some `gpt-5.5` endpoints reject `temperature=0.4` and require the default value `1`.

If you see an error like:

```text
Unsupported value: 'temperature' does not support 0.4 with this model. Only the default (1) value is supported.
```

set:

```bash
export AI_TEMPERATURE="1"
```

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

### Important: automatic git commit and push

When you run `python make_podcast.py ...` directly, the script automatically runs git commands at the end. It pulls, stages `feed.xml` and MP3 files, commits, and pushes.

If you want to preview audio locally before publishing, either comment out the final `git_commit_and_push(...)` call in `make_podcast.py`, or use the local-only recipe below.

### Generate a date range

For a short date range where you are comfortable with the script's automatic git behavior, run one date at a time:

```bash
for d in 2026-07-05 2026-07-06 2026-07-07; do
  python make_podcast.py "$d" --genAI
done
```

For longer ranges, check `feed.xml`, the MP3 files, and `git status` afterward.

### Local-only date range recipe

This recipe imports `make_podcast.py` as a module so it can generate files without triggering the script's automatic commit/push block. It also shows how to pass the AI provider settings through environment variables.

```bash
AI_API_KEY="your-api-key-here" \
AI_BASE_URL="your-chat-completions-endpoint" \
AI_MODEL="gpt-5.5" \
AI_TEMPERATURE="1" \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
python - <<'PY'
import datetime
import importlib.util
from pathlib import Path
from xml.etree import ElementTree as ET

repo = Path.cwd()
spec = importlib.util.spec_from_file_location("make_podcast", repo / "make_podcast.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)

mp.USE_AI = True

def upsert_feed_item(date, audio_filename, audio_length):
    feed_path = repo / mp.FEED_FILENAME
    tree = mp.load_or_init_feed(str(feed_path))
    root = tree.getroot()
    channel = root.find("channel")
    if channel is None:
        channel = ET.SubElement(root, "channel")

    guid_text = f"axion-neutrino-{date.isoformat()}"
    audio_url = f"{mp.BASE_AUDIO_URL}/{audio_filename}"

    for item in channel.findall("item"):
        guid = item.find("guid")
        if guid is not None and guid.text == guid_text:
            enclosure = item.find("enclosure")
            if enclosure is None:
                enclosure = ET.SubElement(item, "enclosure")
            enclosure.set("url", audio_url)
            enclosure.set("length", str(audio_length))
            enclosure.set("type", "audio/mpeg")
            tree.write(feed_path, encoding="utf-8", xml_declaration=True)
            return

    mp.add_item_to_feed(
        tree=tree,
        title=f"Axion and Neutrino arXiv Digest - {date.isoformat()}",
        description=f"Conversational two-voice daily digest of axion and neutrino related arXiv papers for {date.isoformat()}.",
        audio_rel_path=audio_url,
        audio_length=audio_length,
        guid=guid_text,
        pub_date=datetime.datetime.combine(date, datetime.time(8, 0)),
    )
    tree.write(feed_path, encoding="utf-8", xml_declaration=True)

start = datetime.date(2026, 7, 5)
end = datetime.date(2026, 7, 14)

for offset in range((end - start).days + 1):
    date = start + datetime.timedelta(days=offset)
    print(f"=== Generating {date.isoformat()} with {mp.AI_MODEL} ===", flush=True)

    papers = mp.query_arxiv_for_date(date)
    if not papers:
        print(f"No matching arXiv entries for {date.isoformat()}; skipping.", flush=True)
        continue

    suffix = "-genAI"
    script_path = repo / f"{date.isoformat()}-axion-neutrino-conversation{suffix}.txt"
    audio_filename = f"{date.isoformat()}-axion-neutrino-conversation{suffix}.mp3"
    audio_path = repo / audio_filename

    script = mp.generate_conversational_episode(date, papers)
    script_path.write_text(script, encoding="utf-8")
    mp.synthesize_speech(script, str(audio_path))
    upsert_feed_item(date, audio_filename, audio_path.stat().st_size)
PY
```

Adjust `start` and `end` for the date range you want.

---

## Generated files

A successful run creates or updates files like:

```text
2026-07-05-axion-neutrino-conversation-genAI.mp3
2026-07-05-axion-neutrino-conversation-genAI.txt
feed.xml
processed_ids.json
```

The MP3 is the podcast audio. The `.txt` file is the generated script. `feed.xml` is the RSS feed.

This repository's `.gitignore` currently ignores `*.txt` and `*.json`, so generated scripts and `processed_ids.json` may exist locally without showing up as untracked files. MP3s and `feed.xml` are the main publishable outputs.

---

## Regenerating an existing date

The normal `main()` workflow skips arXiv IDs that already appear in `processed_ids.json`. If you re-run a date and see:

```text
No new papers to process for YYYY-MM-DD.
```

that means those papers were already processed.

To regenerate an episode, you can either:

1. remove the relevant paper IDs from `processed_ids.json`, then run the script again, or
2. use the local-only recipe above, which queries the date and regenerates the script/audio directly.

Be careful when regenerating, because it may overwrite an existing MP3 and update `feed.xml`.

---

## Troubleshooting

### The AI script falls back to reading titles and abstracts

If the API call fails, the script falls back to the static script. Check the terminal output for API errors. Common causes are:

- `AI_API_KEY` is missing or invalid
- `AI_BASE_URL` is not set
- `AI_MODEL` is not available from your provider
- the provider rejects the requested `temperature`
- the request timed out

### The model rejects `temperature=0.4`

Use `temperature=1` for models/providers that require the default value.

### No episode was generated for a date

The arXiv query may have found no matching papers for that date, or all matching paper IDs may already be listed in `processed_ids.json`.

### Kokoro prints Hugging Face warnings

Kokoro may print warnings about unauthenticated Hugging Face downloads. This is usually harmless, but setting a Hugging Face token can improve rate limits and download reliability.

### ffmpeg is missing

Install ffmpeg with Homebrew on macOS or your package manager on Linux.

### The script pushed before you reviewed the audio

The direct script entry point commits and pushes automatically. For local preview, comment out `git_commit_and_push(...)` or use the local-only recipe above.

---

## Configuration toggles

| Variable | Default | Description |
|---|---|---|
| `TTS_MODE` | `"kokoro"` | `"kokoro"` for free local voices, `"mac_say_ffmpeg"` for built-in macOS voices |
| `USE_AI` | `False` | Use an LLM to write the script (requires `AI_API_KEY`) |
| `AI_BASE_URL` | `None` | OpenAI-compatible chat-completions endpoint; set with an environment variable before using `--genAI` |
| `AI_MODEL` | `"gpt-4.1-mini"` | Model name sent to the provider; can be set with an environment variable |
| `AI_TEMPERATURE` | `0.4` | Sampling temperature sent to the provider; use `1` for models that require the default |
| `ENABLE_EMAIL` | `False` | Email the script to yourself (requires `sendmail`) |
