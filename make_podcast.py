import os
import sys
import json
import datetime
import requests
import textwrap
import tempfile
import traceback
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import List, Dict, Any, Tuple

# -------------------------------
# CONFIGURATION
# -------------------------------

# 1. Date handling: default to "yesterday" UTC
DEFAULT_DAYS_OFFSET = 1

# 2. Output folder: OneDrive-LLNL on Mac
GIT_REPO_DIR = "."  # change this to whereever you want your files to be saved!
OUTPUT_DIR = GIT_REPO_DIR  # write directly into the repo
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 3. RSS feed metadata (local index)
FEED_TITLE = "Axion and Neutrino arXiv Daily Digest"
FEED_DESCRIPTION = "Daily conversational summaries of arXiv papers mentioning axion or neutrino."
BASE_AUDIO_URL = "https://sumitaghosh.github.io/arxiv-podcast"  # change this to your own repository!
FEED_LINK = f"{BASE_AUDIO_URL}/feed.xml"
FEED_FILENAME = "feed.xml"

# 4. Generative AI configuration (OpenAI-compatible)
USE_AI = False  # Boolean to use Generative AI to write the script; otherwise the voices on your computer will just read the abstracts!
AI_BASE_URL = None  # put a string here for what the URL of your model is
AI_MODEL = None  # put a string here for the name of your model (like the type of gpt)
AI_API_KEY = None  # put your API key here!

# 5. macOS TTS configuration (two voices)
TTS_MODE = "mac_say_ffmpeg"  # two-voice pipeline
MAC_VOICE_HOSTA = "Evan (Enhanced)"  # Pick your favorite for announcing the voices
MAC_VOICE_HOSTB = "Matilda (Enhanced)"  # Pick your favorite for reading the abstracts!

# 6. ffmpeg configuration
FFMPEG_BIN = "ffmpeg"           # assume on PATH; or set full path

# 7. Email configuration
ENABLE_EMAIL = False  # Boolean to email the script to yourself
EMAIL_FROM = "username@domain.end"  # your LLNL email
EMAIL_TO = ["username@domain.end"]  # list of recipients
EMAIL_SUBJECT_PREFIX = "[Arxiv Axion/Neutrino Digest] "


# -------------------------------
# UTILITIES
# -------------------------------


def git_commit_and_push(repo_dir: Path, commit_message: str) -> None:
    """
    Run git add, commit, and push in the given repo directory.
    If there is nothing to commit, it will print a warning and skip.
    """
    try:
        # Ensure we are in the repo directory
        repo_dir = repo_dir.resolve()

        # git status to confirm it is a repo
        subprocess.run(
            ["git", "-C", str(repo_dir), "status"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Add feed.xml and all mp3 files
        subprocess.run(
            ["git", "-C", str(repo_dir), "add", "feed.xml", "*.mp3"],
            check=True,
        )

        # Check if there is anything to commit
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "diff", "--cached", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            print("[Git] No changes to commit; skipping commit and push.")
            return

        # Commit
        subprocess.run(
            ["git", "-C", str(repo_dir), "commit", "-m", commit_message],
            check=True,
        )
        print("[Git] Commit created.")

        # Push
        subprocess.run(
            ["git", "-C", str(repo_dir), "push"],
            check=True,
        )
        print("[Git] Pushed to remote.")

    except subprocess.CalledProcessError as e:
        print(f"[Git] Error running git command: {e}")
    except Exception as e:
        print(f"[Git] Unexpected error during git commit/push: {e}")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_processed_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data)
    except Exception:
        print("Warning: Could not read processed IDs file, starting fresh.")
        return set()


def save_processed_ids(path: str, ids: set) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(ids)), f, indent=2)


def iso_to_rfc2822(dt: datetime.datetime) -> str:
    return dt.strftime("%a, %d %b %Y %H:%M:%S GMT")


# -------------------------------
# STEP 1: Query arXiv
# -------------------------------

def query_arxiv_for_date(date: datetime.date) -> List[Dict[str, Any]]:
    """
    Query arXiv API for papers that contain 'axion' or 'neutrino' in title or abstract,
    then filter by the published date.
    """
    base_url = "http://export.arxiv.org/api/query"
    search_query = "(ti:axion OR abs:axion OR ti:neutrino OR abs:neutrino)"

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": 200,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    print(f"Querying arXiv for date {date.isoformat()}...")
    resp = requests.get(base_url, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries: List[Dict[str, Any]] = []

    for entry in root.findall("atom:entry", ns):
        arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]

        title = (entry.find("atom:title", ns).text or "").strip().replace("\n", " ")
        summary = (entry.find("atom:summary", ns).text or "").strip()
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]

        published_str = entry.find("atom:published", ns).text
        published_dt = datetime.datetime.fromisoformat(
            published_str.replace("Z", "+00:00")
        ).date()

        if published_dt != date:
            continue

        pdf_url = None
        abs_url = None
        for link in entry.findall("atom:link", ns):
            href = link.attrib.get("href", "")
            if link.attrib.get("type") == "application/pdf":
                pdf_url = href
            elif link.attrib.get("rel") == "alternate":
                abs_url = href

        entries.append(
            {
                "id": arxiv_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "published": published_str,
                "pdf_url": pdf_url,
                "abs_url": abs_url,
            }
        )

    print(f"Found {len(entries)} entries for {date.isoformat()} after date filter.")
    return entries


# -------------------------------
# STEP 2: Episode script
# -------------------------------

def call_api(messages: List[Dict[str, str]], temperature: float = 0.4, max_tokens: int = 2000) -> str:
    """
    Call AI (OpenAI-compatible chat/completions).
    """

    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": AI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(AI_BASE_URL, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()

    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        print("Unexpected response format:", data)
        raise


def build_paper_list_for_prompt(papers: List[Dict[str, Any]]) -> str:
    """
    Build a compact list of papers.
    """
    items = []
    for i, p in enumerate(papers, start=1):
        item = textwrap.dedent(
            f"""
            Paper {i}:
            ID: {p['id']}
            Title: {p['title']}
            Authors: {', '.join(p['authors'])}
            Abstract: {p['summary']}
            """
        ).strip()
        items.append(item)
    return "\n\n".join(items)


def generate_conversational_episode(date: datetime.date, papers: List[Dict[str, Any]]) -> str:
    """
    Ask AI to generate a two-host conversational script for the whole episode.
    Output format is structured and labeled for easy reading and parsing.
    """
    paper_block = build_paper_list_for_prompt(papers)

    prompt = f"""
You are generating a concise, technically accurate podcast script for a particle physicist.

There are two hosts:
- HostA: More structured and narrative. Opens the show, announces each paper, keeps things moving.
- HostB: Provides more technical details and context, focusing on experiment/theory, results, and significance.

Audience: HEP/astro postdocs. Assume they know the basics of axion and neutrino physics, cosmology, and standard acronyms.

Topic date: {date.isoformat()}

Here is the list of papers for this date:
{paper_block}

Requirements for the script:
1. Start with a short intro where HostA welcomes the listener and explains that this is a daily axion and neutrino arXiv digest.
2. For each paper, have:
   - HostA briefly announce the paper: title, very short one-line hook.
   - HostB give a 2-3 sentence explanation focusing on:
        - what was actually done (experiment, computation, theory),
        - the main result,
        - why it matters in axion / neutrino / related cosmology.
   - Optionally a very short exchange (1-2 extra lines total) if something is surprising, connects to other work, or is especially important. Keep this tight.
3. End with a short outro from HostA wrapping up the day.
4. Tone: professional but conversational; avoid jokes that rely on non-technical humor. Occasional light phrasing is fine, but the emphasis is on scientific clarity.
5. Do not exceed a few sentences per host per paper; total script should be comfortably listenable in a few minutes, not a long lecture.

Formatting requirements (very important):
- Use plain text.
- Use labels "HostA:" and "HostB:" at the start of each spoken line.
- Group sections with headers:
    [INTRO]
    [PAPER 1]
    [PAPER 2]
    ...
    [OUTRO]
- Under each [PAPER N] section, always mention the arXiv ID explicitly once.
- Each spoken line must be on a single line, for example:
    HostA: Welcome to the show ...
    HostB: Today we are looking at ...

Do NOT include any markdown, bullet points, or additional headings beyond [INTRO], [PAPER N], and [OUTRO].
"""

    messages = [
        {
            "role": "system",
            "content": "You are a concise, technically accurate host of a high-energy physics and cosmology podcast.",
        },
        {"role": "user", "content": prompt.strip()},
    ]
    script = call_api(messages)
    return script.strip()


def generate_static_script(date: str, papers: List[Dict[str, Any]]) -> str:
    """
    Build a simple two-voice script using only arXiv metadata.
    No external API calls, no banter.
    """
    lines = []

    # Intro from Host A
    lines.append("HOST A: Welcome to the Axion and Neutrino arXiv Daily Digest.")
    lines.append(f"HOST A: This episode covers papers posted on {date}.")
    lines.append("HOST A: In this edition, I'll read the titles,")
    lines.append("HOST A: and then Host B will read the full abstracts for each paper.")
    lines.append("")  # blank line

    if not papers:
        lines.append("HOST A: There were no new axion or neutrino papers found today.")
        lines.append("HOST A: Thanks for listening, and see you next time.")
        return "\n".join(lines)

    # Titles section by Host A
    lines.append("HOST A: Here are today's papers.")
    for i, paper in enumerate(papers, start=1):
        title = paper["title"].strip().replace("\n", " ")
        # Optional: shorten author list if long
        authors = paper.get("authors", "")
        if isinstance(authors, list):
            authors_str = ", ".join(authors)
        else:
            authors_str = str(authors)

        lines.append(f"HOST A: Paper {i}: {title}.")
        if authors_str:
            lines.append(f"HOST A: Authors: {authors_str}.")
        lines.append("")  # spacing

    lines.append("HOST A: Now, Host B will read the full abstracts for each paper.")
    lines.append("")  # spacing

    # Abstracts section by Host B
    for i, paper in enumerate(papers, start=1):
        title = paper["title"].strip().replace("\n", " ")
        abstract = paper["summary"].strip()

        lines.append(f"HOST B: Abstract for paper {i}, titled {title}.")
        # Normalize whitespace in abstract so TTS behaves nicely
        abstract_clean = " ".join(abstract.split())
        lines.append(f"HOST B: {abstract_clean}")
        lines.append("")  # spacing

    # Outro from Host A
    lines.append("HOST A: That concludes today's Axion and Neutrino arXiv Daily Digest.")
    lines.append("HOST A: Thanks for listening, and see you next time.")

    return "\n".join(lines)


# -------------------------------
# STEP 3: Parse script into utterances
# -------------------------------

def parse_script_to_utterances(script: str) -> List[Tuple[str, str]]:
    """
    Parse the script into ordered utterances: (speaker, text).
    Only lines starting with 'HostA:' or 'HostB:' are used.
    """
    utterances: List[Tuple[str, str]] = []
    for line in script.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("HostA:"):
            text = line[len("HostA:"):].strip()
            if text:
                utterances.append(("HostA", text))
        elif line.startswith("HostB:"):
            text = line[len("HostB:"):].strip()
            if text:
                utterances.append(("HostB", text))
        else:
            # ignore headers like [INTRO], [PAPER 1], [OUTRO]
            continue
    return utterances


# -------------------------------
# STEP 4: macOS TTS with two voices + ffmpeg
# -------------------------------

def synthesize_clip_with_say(text: str, voice: str, output_aiff: str) -> None:
    """
    Use macOS `say` to synthesize text to AIFF.
    """
    # Write text to a temporary file so we do not have to deal with shell quoting
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as tf:
        tf.write(text)
        tf_path = tf.name

    try:
        cmd = ["say", "-v", voice, "-o", output_aiff, "-f", tf_path]
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.remove(tf_path)
        except OSError:
            pass


def concatenate_aiff_with_ffmpeg(aiff_files: List[str], output_path: str) -> None:
    """
    Use ffmpeg to concatenate multiple AIFF files into a single MP3.
    We use the 'concat' demuxer with a temporary list file.
    """
    if not aiff_files:
        raise ValueError("No AIFF files to concatenate.")

    # Create a temporary file listing all the AIFF files
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as lf:
        for f in aiff_files:
            lf.write(f"file '{f}'\n")
        list_path = lf.name

    try:
        # ffmpeg -f concat -safe 0 -i list.txt -acodec libmp3lame -q:a 4 output.mp3
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-acodec",
            "libmp3lame",
            "-q:a",
            "4",
            output_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


def synthesize_two_voice_episode(script: str, output_path: str) -> None:
    """
    Two-voice synthesis:
    - Parse script into utterances.
    - For each utterance, synthesize a small AIFF clip with HostA/HostB voice.
    - Concatenate clips with ffmpeg into a single MP3.
    """
    utterances = parse_script_to_utterances(script)
    if not utterances:
        print("[TTS] No utterances found in script; creating empty audio file.")
        with open(output_path, "wb") as f:
            f.write(b"")
        return

    print(f"[TTS] Synthesizing {len(utterances)} utterances with two voices...")

    temp_dir = tempfile.mkdtemp(prefix="arxiv_podcast_tts_")
    aiff_files: List[str] = []

    try:
        for idx, (speaker, text) in enumerate(utterances):
            voice = MAC_VOICE_HOSTA if speaker == "HostA" else MAC_VOICE_HOSTB
            clip_path = os.path.join(temp_dir, f"clip_{idx:04d}.aiff")
            print(f"[TTS] {speaker} -> {clip_path}")
            synthesize_clip_with_say(text, voice, clip_path)
            aiff_files.append(clip_path)

        # Concatenate to final MP3
        concatenate_aiff_with_ffmpeg(aiff_files, output_path)
        print(f"[TTS] Final two-voice episode at {output_path}")
    finally:
        # Clean up temporary AIFF files and directory
        for f in aiff_files:
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


def synthesize_speech(script: str, output_path: str) -> None:
    if TTS_MODE == "mac_say_ffmpeg":
        synthesize_two_voice_episode(script, output_path)
    else:
        print(f"[TTS] Unknown TTS_MODE '{TTS_MODE}', creating empty file.")
        with open(output_path, "wb") as f:
            f.write(b"")


# -------------------------------
# STEP 5: Local RSS-like feed
# -------------------------------

def load_or_init_feed(feed_path: str) -> ET.ElementTree:
    if os.path.exists(feed_path):
        return ET.parse(feed_path)

    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")

    title_el = ET.SubElement(channel, "title")
    title_el.text = FEED_TITLE

    link_el = ET.SubElement(channel, "link")
    link_el.text = FEED_LINK

    desc_el = ET.SubElement(channel, "description")
    desc_el.text = FEED_DESCRIPTION

    return ET.ElementTree(rss)


def add_item_to_feed(
    tree: ET.ElementTree,
    title: str,
    description: str,
    audio_rel_path: str,
    audio_length: int,
    guid: str,
    pub_date: datetime.datetime,
) -> None:
    root = tree.getroot()
    channel = root.find("channel")
    if channel is None:
        channel = ET.SubElement(root, "channel")

    item = ET.Element("item")

    title_el = ET.SubElement(item, "title")
    title_el.text = title

    desc_el = ET.SubElement(item, "description")
    desc_el.text = description

    enclosure_el = ET.SubElement(item, "enclosure")
    enclosure_el.set("url", audio_rel_path)
    enclosure_el.set("length", str(audio_length))
    enclosure_el.set("type", "audio/mpeg")

    guid_el = ET.SubElement(item, "guid")
    guid_el.text = guid

    pub_el = ET.SubElement(item, "pubDate")
    pub_el.text = iso_to_rfc2822(pub_date)

    channel.insert(0, item)


def save_feed(tree: ET.ElementTree, feed_path: str) -> None:
    tree.write(feed_path, encoding="utf-8", xml_declaration=True)


# -------------------------------
# STEP 6: Email sender
# -------------------------------

def send_email(subject: str, body: str) -> None:
    if not ENABLE_EMAIL:
        print("[Email] Email sending disabled; skipping.")
        return

    msg = f"From: {EMAIL_FROM}\nTo: {', '.join(EMAIL_TO)}\nSubject: {subject}\n\n{body}"

    try:
        p = subprocess.Popen(["/usr/sbin/sendmail", "-t", "-oi"], stdin=subprocess.PIPE)
        p.communicate(msg.encode("utf-8"))
        if p.returncode != 0:
            print(f"[Email] sendmail returned exit code {p.returncode}")
        else:
            print("[Email] Sent daily digest email via sendmail.")
    except FileNotFoundError:
        print("[Email] /usr/sbin/sendmail not found; email disabled.")
    except Exception as e:
        print(f"[Email] Error sending email via sendmail: {e}")
        traceback.print_exc()


# -------------------------------
# MAIN WORKFLOW
# -------------------------------

def main(target_date: datetime.date = None) -> None:
    if target_date is None:
        today_utc = datetime.datetime.utcnow().date()
        target_date = today_utc - datetime.timedelta(days=DEFAULT_DAYS_OFFSET)

    ensure_output_dir(OUTPUT_DIR)
    processed_ids_path = os.path.join(OUTPUT_DIR, "processed_ids.json")
    processed_ids = load_processed_ids(processed_ids_path)

    papers = query_arxiv_for_date(target_date)
    new_papers = [p for p in papers if p["id"] not in processed_ids]

    if not new_papers:
        print(f"No new papers to process for {target_date.isoformat()}.")
        return

    print(f"{len(new_papers)} new papers to process.")

    # Step 1: Generate conversational episode script
    if USE_AI:
        try:
            episode_script = generate_conversational_episode(target_date, new_papers)
        except Exception as e:
            episode_script = generate_static_script(target_date, new_papers)
    else:
        episode_script = generate_static_script(target_date, new_papers)

    # Step 2: Save text script
    script_filename = f"{target_date.isoformat()}-axion-neutrino-conversation.txt"
    script_path = os.path.join(OUTPUT_DIR, script_filename)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(episode_script)
    print(f"Wrote conversational script to {script_path}")

    # Step 3: Generate two-voice audio
    audio_filename = f"{target_date.isoformat()}-axion-neutrino-conversation.mp3"
    audio_path = os.path.join(OUTPUT_DIR, audio_filename)
    synthesize_speech(episode_script, audio_path)

    try:
        audio_length = os.path.getsize(audio_path)
    except OSError:
        audio_length = 0

    # Step 4: Update local feed
    feed_path = os.path.join(OUTPUT_DIR, FEED_FILENAME)
    tree = load_or_init_feed(feed_path)

    episode_title = f"Axion and Neutrino arXiv Digest – {target_date.isoformat()}"
    episode_desc = f"Conversational two-voice daily digest of axion and neutrino related arXiv papers for {target_date.isoformat()}."
    guid = f"axion-neutrino-{target_date.isoformat()}"
    pub_date = datetime.datetime.combine(target_date, datetime.time(8, 0))

    audio_url = f"{BASE_AUDIO_URL}/{audio_filename}"

    add_item_to_feed(
        tree=tree,
        title=episode_title,
        description=episode_desc,
        audio_rel_path=audio_url,  # now a full URL
        audio_length=audio_length,
        guid=guid,
        pub_date=pub_date,
    )
    save_feed(tree, feed_path)
    print(f"Updated local feed at {feed_path}")

    # Step 5: Update processed IDs
    for p in new_papers:
        processed_ids.add(p["id"])
    save_processed_ids(processed_ids_path, processed_ids)
    print("Updated processed IDs list.")

    # Step 6: Send email with text script
    subject = EMAIL_SUBJECT_PREFIX + target_date.isoformat()
    send_email(subject, episode_script)


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            d = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
            main(d)
        else:
            main()
    except Exception as e:
        print("Fatal error:", e)
        traceback.print_exc()

    # Auto commit and push
    today_str = datetime.date.today().isoformat()
    commit_message = f"Update arxiv digest for {today_str}"
    git_commit_and_push(GIT_REPO_DIR, commit_message)
