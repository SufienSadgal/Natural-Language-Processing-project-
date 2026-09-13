"""
Speech mode client for PoliMillionaire.

Starts a game in "speech" mode, fetches WAV audio for the question and each
option, transcribes them with faster-whisper, then routes to the normal
ask_gemma() pipeline.

Usage (in the notebook):
    from speech_client import play_game_speech
    game = client.game.start(competition_id=comp_id, mode="speech")
    play_game_speech(game)
"""

import concurrent.futures
import io
import logging
import re
import sys
import threading
import time

import numpy as np

try:
    from IPython.display import Audio, display as ipy_display
    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False

# ── ASR setup ────────────────────────────────────────────────────────────────
# "small" (244M params) handles unusual prosody and accents far better than
# "base" (74M). The extra ~0.5s per clip is worth it given the 30s game window.
# Change to "medium" for even better accuracy if time allows.
def _load_whisper_model(name: str = "small"):
    """Load a faster-whisper model in a background thread with a live spinner."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("WARNING: faster-whisper not installed — run: pip install faster-whisper")
        return None

    # Use CUDA when available (T4/A100 on Colab), fall back to CPU
    try:
        import torch
        _use_cuda = torch.cuda.is_available()
    except ImportError:
        _use_cuda = False
    device       = "cuda" if _use_cuda else "cpu"
    compute_type = "float16" if _use_cuda else "int8"

    _container = [None]
    _error     = [None]
    _ready     = threading.Event()

    def _load():
        try:
            _container[0] = WhisperModel(name, device=device, compute_type=compute_type)
        except Exception as exc:
            _error[0] = exc
        finally:
            _ready.set()

    threading.Thread(target=_load, daemon=True).start()

    _spin = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    _si, _t0 = 0, time.time()
    print(f"Loading Whisper '{name}' model (first run downloads ~500 MB) ...")
    while not _ready.wait(timeout=0.15):
        elapsed = time.time() - _t0
        print(f'\r  {_spin[_si % len(_spin)]}  {elapsed:.0f}s elapsed ...', end='', flush=True)
        _si += 1

    elapsed = time.time() - _t0
    if _error[0]:
        print(f'\r❌  Failed to load Whisper: {_error[0]}')
        return None

    print(f'\r✅  Whisper {name!r} ready in {elapsed:.0f}s               ')
    logging.info("faster-whisper loaded (model=%s)", name)
    return _container[0]


# Model options (faster-whisper names):
#   "small"          — fast on CPU, weaker on adversarial audio (current default)
#   "medium"         — ~3-4× slower, noticeably better quality
#   "large-v3-turbo" — OpenAI distilled large-v3, ~2-3× slower than small, best quality/speed
#   "distil-large-v3"— community distilled, ~2× slower than small, near large-v3 quality
# Model selection guide (faster-whisper names):
#   "small"           — fast even on CPU, good quality (~244M params)
#   "large-v3-turbo"  — best quality/speed on GPU, too slow on CPU for a 30s timer
#   "distil-large-v3" — near large-v3 quality, ~2× slower than small
#
# On Colab T4 GPU: large-v3-turbo transcribes in <1s per clip (fine)
# On CPU:          large-v3-turbo takes 10-15s per clip (timer expires!)
# The code below auto-detects CUDA and uses float16 when available,
# so large-v3-turbo is safe to use — it will only be slow if there is no GPU.
ASR_MODEL_NAME = "large-v3-turbo"
_ASR_MODEL = _load_whisper_model(ASR_MODEL_NAME)


# ── Transcription cleaning ────────────────────────────────────────────────────

# Whisper sometimes emits special noise tags for non-speech sounds.
_WHISPER_TAG = re.compile(r'\[.*?\]')

# Repeated laugh/filler syllables injected adversarially: "ha ha ha ha", "oho", etc.
# We match sequences of these with optional punctuation between them.
_LAUGH_SEQ = re.compile(
    r'\b(?:(?:ha|ah|oh|oho|heh|hehe|haha|aha|lol|hm+|um|uh|er)\b[\s,]*){2,}',
    re.IGNORECASE,
)

# Single leading filler word (um, uh, er, hmm) at the very start of a segment.
_LEADING_FILLER = re.compile(r'^\s*\b(um|uh|er|hmm?)\b[\s,]*', re.IGNORECASE)

# "Option A, " prefix spoken by the TTS — also catches garbled versions like
# "Absin A.", "Ab A,", "Options D -", "Option D?" etc.
# Pattern: optional word(s) up to 8 chars, then a single letter A-D, then punctuation/space.
# Separator includes ? and ! for adversarial cases like "Option D? Painting."
_OPTION_PREFIX = re.compile(
    r'^\s*(?:[A-Za-z]{1,8}\s+)?([A-Da-d])\s*[,.\-?!\s]\s*',
)

# Repeated characters from screamed/elongated words: "Hoorrrrrrrr" → "Hoorr".
# We reduce any run of 3+ identical characters to exactly 2, which preserves
# double letters (e.g. "cool") while collapsing screams ("Hoorrrr" → "Hoorr").
# The model can then infer "hoorr" ≈ "horror", "draaama" ≈ "drama", etc.
_ELONGATED = re.compile(r'(.)\1{2,}')

# Detached-prefix repair: adversarial audio sometimes inserts a pause inside a
# prefixed word so Whisper hears two tokens instead of one.
# E.g. "decreasing" → "to creasing"  (de- sounds like "to" when aspirated+paused)
#      "increasing"  → "an creasing"  (in- → "an")
#      "undoing"     → "on doing"     (un- → "on")
#      "disconnected"→ "this connected" (dis- → "this")
#
# We match: a short phonetic-proxy word + space + word with a common suffix.
# The proxy must NOT be followed by a bare infinitive (e.g. "to run" is valid
# English and must not be altered) — hence we require the continuation to carry
# a bound morpheme (-ing/-ed/-tion/-ment/-er/-al/-ive/-ous) which cannot follow
# "to" grammatically as a gerund.
#
# False-positive risk is very low: "to creasing", "an creasing", "on doing" etc.
# are all ungrammatical in English, so the repair is safe whenever it fires.
_DETACHED_PREFIX = re.compile(
    r'\b(to|the|di|an|on|this)\s+([a-z]{4,}(?:ing|ed|tion|tions|ment|ments|er|ers|al|ive|ous))\b',
    re.IGNORECASE,
)
_PREFIX_MAP = {
    'to':   'de',   # "to creasing"    → "decreasing"
    'the':  'de',   # "the creasing"   → "decreasing"
    'di':   'de',   # "di creasing"    → "decreasing"
    'an':   'in',   # "an creasing"    → "increasing"
    'on':   'un',   # "on doing"       → "undoing"
    'this': 'dis',  # "this connected" → "disconnected"
}


def _repair_detached_prefix(text: str) -> str:
    """Reattach phonetically detached prefixes: 'to creasing' → 'decreasing'."""
    def _replace(m: re.Match) -> str:
        proxy = m.group(1).lower()
        rest  = m.group(2)
        prefix = _PREFIX_MAP.get(proxy, '')
        return prefix + rest.lower() if prefix else m.group(0)
    return _DETACHED_PREFIX.sub(_replace, text)


def _clean_transcription(text: str, is_option: bool = False) -> str:
    """
    Remove adversarial noise artifacts from a Whisper transcription.

    Handles:
    - Whisper special tags: [laughter], [cough], [applause], [music]
    - Repeated laugh/filler syllables: "ha ha ha ha", "oho", "aha"
    - Single leading filler: "um,", "uh,"
    - Spoken option prefixes: "Option A,", "Absin A.", "Option D?", "Option D, P,"
    - Elongated/screamed words: "Hoorrrrrrrr" → "Hoorr" (≈ "horror")
    - Detached prefixes: "to creasing" → "decreasing" (is_option=True only)
    - Spurious trailing "?": stripped when is_option=True (options are never questions;
      adversarial rising intonation tricks Whisper into adding "?")
    """
    # 1. Remove Whisper's special noise tags
    text = _WHISPER_TAG.sub(' ', text)

    # 2. Remove laugh/filler sequences (2+ repetitions)
    text = _LAUGH_SEQ.sub(' ', text)

    # 3. Remove single leading filler word
    text = _LEADING_FILLER.sub('', text)

    # 4. Strip spoken option prefix ("Option A,", "Absin A.", "Option D?", etc.)
    text = _OPTION_PREFIX.sub('', text)

    # 5. Collapse elongated/screamed characters: "Hoorrrrrr" → "Hoorr"
    #    Runs of 3+ identical chars are reduced to 2, preserving natural double
    #    letters while making screamed words recognisable ("hoorr" ≈ "horror").
    text = _ELONGATED.sub(r'\1\1', text)

    # 6. Option-specific repairs
    if is_option:
        # Reattach phonetically detached prefixes: "to creasing" → "decreasing"
        text = _repair_detached_prefix(text)

        # Strip spurious trailing "?" and "!" — options are declarative statements;
        # adversarial intonation (rising for "?", shouting for "!") tricks Whisper.
        text = re.sub(r'[?!]+\s*$', '', text)

        # Normalize ALL-CAPS options: adversarial TTS sometimes screams an option
        # so Whisper transcribes it in all caps (e.g. "CRATERS" → "Craters").
        # All-caps makes the word harder for the LLM to recognise. Convert to
        # sentence case only when the whole token stream is uppercase.
        words = text.split()
        if words and all(w.isupper() or not w.isalpha() for w in words):
            text = text.capitalize()

    # 7. Normalise whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ── Speech-mode prompt note ───────────────────────────────────────────────────
# Prepended to the context block so the model knows to handle transcription noise.
_SPEECH_NOTE = (
    "NOTE: This question and its answer options were transcribed from adversarial speech "
    "audio using ASR. The transcription may contain artifacts: "
    "garbled or mispronounced words (e.g. 'vocal' for 'local', 'vote' for 'road'), "
    "truncated phrases, residual noise, or elongated/screamed words where repeated "
    "characters survive (e.g. 'hoorr' means 'horror', 'draama' means 'drama'). "
    "IMPORTANT: the OPTIONS themselves may also be incorrectly transcribed. "
    "Before evaluating any option, ask: 'Is this grammatically correct and meaningful?' "
    "If an option contains a grammar error (e.g. 'to creasing'), a nonsensical word "
    "(e.g. 'Poor' where the question asks 'how many'), or an incomplete phrase, "
    "treat it as a mistranscription and recover the intended word before evaluating it "
    "(e.g. 'Poor' → 'Four', 'to creasing' → 'decreasing', 'liver' → 'lever'). "
    "Reason from the likely intended meaning of each option — the garbling does not "
    "change which answer is correct."
)


# ── Question-type-aware initial prompt builder ────────────────────────────────
# Whisper's LM is biased by initial_prompt. Appending a short type hint after
# the question text primes the decoder to expect a specific token class for the
# option clip, dramatically improving transcription of number words, names, etc.
_Q_NUMBER   = re.compile(r'\bhow many\b', re.IGNORECASE)
_Q_YEAR     = re.compile(r'\b(which|what|in what|in which)\s+year\b', re.IGNORECASE)
_Q_WHO      = re.compile(r'^(who\b|which (person|individual|man|woman|leader|president|author|scientist|artist|musician|director|actor|actress))', re.IGNORECASE)
_Q_WHERE    = re.compile(r'\b(where|which (country|city|state|continent|region|place|location|river|mountain|ocean|island))\b', re.IGNORECASE)


def _build_option_prompt(question_text: str) -> str:
    """
    Build an enriched initial_prompt for option transcription.

    Appends a short type hint to the question text so Whisper's LM knows
    what kind of token to expect next:
      - "how many …"   → "The answer is a number:"
      - "which year …" → "The answer is a year:"
      - "who …"        → "The answer is a name:"
      - "where …"      → "The answer is a place:"
      - default        → question text only (already helpful for vocabulary)

    This causes Whisper to decode number-words correctly even when adversarial
    audio makes them sound like homophones ('Poor' → 'Four', 'To' → 'Two').
    """
    base = question_text.strip().rstrip('?').strip()
    if _Q_NUMBER.search(question_text):
        hint = "The answer is a number:"
    elif _Q_YEAR.search(question_text):
        hint = "The answer is a year:"
    elif _Q_WHO.search(question_text):
        hint = "The answer is a name:"
    elif _Q_WHERE.search(question_text):
        hint = "The answer is a place:"
    else:
        hint = "The answer is:"
    return f"{base}. {hint}"


def _transcribe(wav_bytes: bytes, initial_prompt: str | None = None, is_option: bool = False) -> str:
    """
    Transcribe WAV audio bytes to cleaned text using faster-whisper.

    Args:
        wav_bytes:      Raw WAV audio bytes.
        initial_prompt: Optional text Whisper treats as spoken context immediately
                        before this clip. Biases the LM toward related vocabulary,
                        dramatically improving accuracy for domain-specific words.
                        Pass the question text when transcribing answer options so
                        Whisper knows what topic it is in (e.g. "smallest particle
                        of an element" → biases toward "atom" over "Adam").

    Improvements over basic transcription:
    - initial_prompt: question-context injection for option transcription.
    - VAD filter: removes silence, breath sounds, and coughs before ASR so
      long pauses do not cause content after the pause to be dropped.
    - beam_size=5: good accuracy/speed tradeoff.
    - Post-processing via _clean_transcription(): strips laugh sequences,
      Whisper noise tags, spoken option prefixes, and leading filler words.
    """
    if _ASR_MODEL is None:
        raise RuntimeError("faster-whisper is not installed. Run: pip install faster-whisper")

    import scipy.io.wavfile as wavfile

    rate, data = wavfile.read(io.BytesIO(wav_bytes))

    # Convert to mono float32 in [-1, 1]
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if data.max() > 1.0:
        data /= 32768.0  # int16 → float32

    segments, _ = _ASR_MODEL.transcribe(
        data,
        language="en",
        beam_size=5,
        initial_prompt=initial_prompt,
        # VAD filter: uses Silero VAD to detect and skip non-speech regions
        # (silence, breath, coughs) before feeding to Whisper.
        # min_silence_duration_ms=300: gaps shorter than 300ms are kept as
        # speech so we don't over-split natural sentence rhythm.
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    raw = " ".join(seg.text.strip() for seg in segments).strip()
    cleaned = _clean_transcription(raw, is_option=is_option)

    if raw != cleaned:
        logging.info("Transcribed (raw)    : %r", raw)
        logging.info("Transcribed (cleaned): %r", cleaned)
    else:
        logging.info("Transcribed: %r", cleaned)

    return cleaned


# ── Main play loop ────────────────────────────────────────────────────────────

def play_game_speech(
    game,
    username: str = "unknown",
    competition_id: int = 0,
    use_rag: bool = False,
) -> None:
    """
    Play a PoliMillionaire game in speech mode.

    Fetches WAV audio for the question and all four options, transcribes them,
    cleans adversarial noise, then calls ask_gemma() exactly as in text mode.

    Args:
        game: GameSession started with mode="speech"
        username: used only for logging / wrong-answer records
        competition_id: competition ID, used for wrong-answer logging
        use_rag: if True, run RAG on the transcribed question before asking the model
    """
    from ollama_client import ask_gemma
    from wrong_answers import save_wrong_answer
    from rag import fetch_rag_context, should_use_rag

    all_resources = []

    while game.in_progress:
        level = game.current_level
        print(f"\n--- Level {level} ---")

        # ── Fetch and transcribe question audio ───────────────────────────
        print("Fetching question audio...")
        t0 = time.time()
        try:
            question_audio = game.fetch_audio_question()
        except Exception as e:
            logging.error("Could not fetch question audio: %s", e)
            break

        if _IPYTHON_AVAILABLE:
            print("▶ Question audio:")
            ipy_display(Audio(question_audio, autoplay=False))

        question_text = _transcribe(question_audio)
        print(f"Q: {question_text}")

        # ── Start RAG in background immediately after question transcription ──
        # The 4 option audio fetches below take ~4-8s (sequential by protocol).
        # RAG runs concurrently in a thread pool so it finishes during that
        # window — effectively free wall-clock time.
        # In speech mode: enriched query and per-option reranking are disabled
        # (garbled option text would pollute both), so options=None is fine here.
        _rag_executor = None
        _rag_future   = None
        if use_rag and should_use_rag(question_text):
            _rag_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            _rag_future   = _rag_executor.submit(
                fetch_rag_context, question_text, None, 5, True  # speech_mode=True
            )
            print("RAG running in background...")

        # ── Fetch and transcribe option audios (must be sequential A→B→C→D) ─
        options_text = {}
        option_ids = {}

        q = game.current_question  # may have .text = None in speech mode
        for i in range(4):
            letter = chr(65 + i)  # A, B, C, D
            print(f"Fetching option {letter} audio...")
            try:
                opt_audio = game.fetch_audio_option_next()
            except Exception as e:
                logging.error("Could not fetch option %s audio: %s", letter, e)
                break

            if _IPYTHON_AVAILABLE:
                print(f"▶ Option {letter} audio:")
                ipy_display(Audio(opt_audio, autoplay=False))

            opt_text = _transcribe(opt_audio, initial_prompt=_build_option_prompt(question_text), is_option=True)
            options_text[i] = opt_text
            print(f"  {i}: {opt_text}")

            if q and i < len(q.options):
                option_ids[i] = q.options[i].id

        if len(options_text) < 4:
            print("Failed to fetch all option audios. Aborting.")
            if _rag_executor:
                _rag_executor.shutdown(wait=False)
            break

        # Refresh state so time_remaining is accurate (timer starts after option D)
        game.refresh_state()
        time_left = game.time_remaining
        if time_left is not None:
            print(f"\nTime remaining: {time_left:.1f}s")

        fetch_elapsed = time.time() - t0
        logging.info("Audio fetch + transcription took %.1fs", fetch_elapsed)

        # ── Collect RAG result (should already be done) ───────────────────
        # Give it 2s grace period in case transcription finished early and
        # RAG is still running. After that, proceed without context.
        rag_result = None
        if _rag_future is not None:
            try:
                rag_result = _rag_future.result(timeout=2.0)
            except concurrent.futures.TimeoutError:
                logging.warning("RAG did not finish in time — proceeding without context")
            except Exception as exc:
                logging.warning("RAG raised an error: %s", exc)
            finally:
                _rag_executor.shutdown(wait=False)

        # ── Build context: always prepend speech-mode note ────────────────
        # The note tells the model to be robust to ASR transcription artifacts.
        # RAG context (if any) is appended after the note.
        context = _SPEECH_NOTE

        if rag_result:
            context = context + "\n\n" + rag_result

        # ── Ask the model ─────────────────────────────────────────────────
        answer_input, reasoning, resources = ask_gemma(
            question_text, options_text, context=context, speech_mode=True
        )
        all_resources.append(resources)
        answer_idx = int(answer_input)

        # Map model answer index → server option ID
        if answer_idx not in option_ids:
            logging.warning(
                "Answer index %d not in option_ids %s — defaulting to 0",
                answer_idx, option_ids,
            )
            answer_idx = 0
        answer_id = option_ids[answer_idx]

        print(f"\nModel answer: option {answer_idx} = \"{options_text[answer_idx]}\"")

        # ── Submit answer ─────────────────────────────────────────────────
        try:
            result = game.answer(answer_id)
        except Exception as e:
            logging.error("Failed to submit answer: %s", e)
            break

        if result.timed_out or result.status == "timeout":
            print("TIMED OUT!")
            save_wrong_answer(
                username=username,
                competition_id=competition_id,
                level=level,
                question=question_text,
                options=options_text,
                reasoning=reasoning,
                answer_given=str(answer_idx),
                rag_context=context,
                resources=resources,
                timed_out=True,
            )
            print(f"\n Game Over! Final earnings: ${result.earned_amount:,.2f}")
            break

        if result.correct:
            print(f" CORRECT! Earned so far: ${result.earned_amount:,.2f}")
            if result.game_over:
                print(f"\n CONGRATULATIONS! Final earnings: ${result.earned_amount:,.2f}")
        else:
            print(" WRONG ANSWER!")
            save_wrong_answer(
                username=username,
                competition_id=competition_id,
                level=level,
                question=question_text,
                options=options_text,
                reasoning=reasoning,
                answer_given=str(answer_idx),
                rag_context=context,
                resources=resources,
                timed_out=False,
            )
            print(f"\n Game Over! Final earnings: ${result.earned_amount:,.2f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n=== Game Summary ===")
    print(f"Reached Level: {game.current_level}")
    print(f"Total Earnings: ${game.earned_amount:,.2f}")

    if all_resources:
        n = len(all_resources)
        avg_time   = round(sum(r.get("elapsed", 0)   for r in all_resources) / n, 2)
        total_time = round(sum(r.get("elapsed", 0)   for r in all_resources), 1)
        avg_tps    = round(sum(r.get("tps", 0)       for r in all_resources) / n, 1)
        avg_cpu    = round(sum(r.get("cpu_avg", 0)   for r in all_resources) / n, 1)
        peak_cpu   = round(max(r.get("cpu_peak", 0)  for r in all_resources), 1)
        avg_ram    = round(sum(r.get("ram_avg", 0)   for r in all_resources) / n, 2)
        peak_ram   = round(max(r.get("ram_peak", 0)  for r in all_resources), 2)
        print(f"\n=== Resource Usage ({n} question(s)) ===")
        print(f"  Avg response time : {avg_time:.2f}s  (total {total_time:.1f}s)")
        print(f"  Avg throughput    : {avg_tps:.1f} tok/s")
        print(f"  CPU  avg={avg_cpu:.1f}%  peak={peak_cpu:.1f}%")
        print(f"  RAM  avg={avg_ram:.2f}GB  peak={peak_ram:.2f}GB")
