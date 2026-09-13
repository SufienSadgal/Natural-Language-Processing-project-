"""
ollama_client.py — llama-cpp-python backend (Colab-compatible).

Drop-in replacement for the Ollama-based ollama_client.py.
Public API is identical: ask_gemma(question, options, context, speech_mode)
returns (answer_digit, reasoning, resources).

Setup on Colab:
    !CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --quiet
    from huggingface_hub import hf_hub_download
    MODEL_PATH = hf_hub_download("Qwen/Qwen2.5-7B-Instruct-GGUF",
                                  "qwen2.5-7b-instruct-q4_k_m.gguf")
"""

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time

import psutil
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Model configuration ───────────────────────────────────────────────────────
# Set MODEL_PATH in .env or before importing, e.g.:
#   os.environ["MODEL_PATH"] = "/path/to/model.gguf"
MODEL_PATH = os.getenv("MODEL_PATH", "model.gguf")

# MODEL_NAME is used only for capability detection (tool support, thinking).
# It does NOT need to be the exact filename — just a recognisable substring.
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5")

# llama.cpp inference parameters (override via env vars or edit here)
N_CTX        = int(os.getenv("N_CTX",        "8192"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))    # -1 = all layers on GPU
N_THREADS    = int(os.getenv("N_THREADS",    "0"))     # 0 = auto

_SAMPLE_INTERVAL = 0.5

# Models that do NOT support the tools API in llama-cpp-python.
# Extend as needed.
_TOOLS_UNSUPPORTED = {"deepseek-r1", "qwen2-math", "mistral"}
_SUPPORTS_TOOLS = not any(t in MODEL_NAME.lower() for t in _TOOLS_UNSUPPORTED)

# Models that output native <think>…</think> blocks (parsed from content).
_THINKING_SUPPORTED = {"deepseek-r1", "qwq", "qwen3"}
_SUPPORTS_THINKING = any(t in MODEL_NAME.lower() for t in _THINKING_SUPPORTED)

# Streaming timeouts
_SOFT_TIMEOUT = 22.0
_HARD_TIMEOUT = 26.0

_GREEN = "\033[92m"
_RED   = "\033[91m"
_RESET = "\033[0m"
_COLOR_THRESHOLD = 30.0

# RAG
ENABLE_RAG = os.getenv("ENABLE_RAG", "false").lower() == "true"

# ── Lazy model loader ─────────────────────────────────────────────────────────

_llm = None
_llm_lock = threading.Lock()


def _get_llm():
    """Load the llama-cpp-python model once and cache it."""
    global _llm
    if _llm is not None:
        return _llm
    with _llm_lock:
        if _llm is not None:
            return _llm
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is not installed.\n"
                "On Colab with CUDA:  CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python\n"
                "On Mac with Metal:   CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python\n"
                "CPU only:            pip install llama-cpp-python"
            )
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                "Download a GGUF from HuggingFace, e.g.:\n"
                "  from huggingface_hub import hf_hub_download\n"
                "  MODEL_PATH = hf_hub_download('Qwen/Qwen2.5-7B-Instruct-GGUF',\n"
                "                                'qwen2.5-7b-instruct-q4_k_m.gguf')"
            )
        logging.info("Loading model: %s (n_ctx=%d, n_gpu_layers=%d)", MODEL_PATH, N_CTX, N_GPU_LAYERS)
        kwargs = dict(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=False,
        )
        if N_THREADS > 0:
            kwargs["n_threads"] = N_THREADS
        _llm = Llama(**kwargs)
        logging.info("Model loaded successfully.")
    return _llm


# ── Prompts ───────────────────────────────────────────────────────────────────

SPEECH_SYSTEM_PROMPT = (
    "You are answering a quiz question transcribed from adversarial speech audio. "
    "Both the question and the options may be garbled by the speech recogniser.\n\n"

    "STEP 1 — RECONSTRUCT THE QUESTION. "
    "If the question is incoherent, read it phonetically and infer the most plausible topic and phrasing. "
    "Do not proceed until you have a clear question in mind.\n\n"

    "STEP 2 — FORM YOUR ANSWER independently, before reading the options. "
    "Name the specific fact, term, value, or person. "
    "Use provided context only if it directly and specifically addresses what is asked.\n\n"

    "STEP 3 — RECONSTRUCT EVERY OPTION. This step is mandatory and has no exceptions. "
    "Each option — no matter how garbled, absurd, or incomplete it looks — is a corrupted transcription "
    "of a real, grammatically correct, meaningful phrase. "
    "Your only task here is to find that phrase. "
    "For each option write: 'Option X decodes to: [your reconstruction]'. "
    "Rules for reconstruction:\n"
    "  • Proper nouns, titles, and technical terms are the most heavily distorted — work hardest on these.\n"
    "  • Words cut off mid-syllable are real answers truncated by the audio — complete them phonetically.\n"
    "  • Filler sounds, laughter, and noise embedded in an option mask the underlying word — read through them.\n"
    "  • Never write 'garbled', 'nonsensical', or 'unclear' — every option has a reconstruction, find it.\n\n"

    "STEP 4 — SELECT. "
    "Compare your reconstructions to your pre-formed answer. "
    "Pick the option whose reconstruction best matches. "
    "A garbled option that decodes to the correct answer always beats a clean option that is factually wrong — "
    "do not let surface clarity bias your choice.\n\n"

    "OUTPUT: write 'Therefore my answer is option X.' then X as a bare digit (0-3)."
)

SYSTEM_PROMPT = (
    "You are playing a quiz game. Reason step by step, then output one digit (0-3).\n\n"
    "ELIMINATE options that are factually wrong.\n "
    "Absolute words in an option require zero exceptions — one counterexample kills it. "
    "An impossible mechanism disqualifies an option even if its conclusion sounds right.\n\n"
    "CONTEXT: if a retrieved block is provided, accept it only if it covers the right subject "
    "and states the specific fact asked. If so, treat it as ground truth and do not override it. "
    "Otherwise discard it and use your knowledge.\n\n"
    "CHOOSE the most specific, precise surviving option. "
    "The option label is its index (0-3), never confuse it with a computed numerical result.\n\n"
    "OUTPUT: write 'Therefore my answer is option X.' then X as a bare digit."
)


# ── Resource sampling ─────────────────────────────────────────────────────────

def _start_resource_sampler() -> tuple:
    samples = []
    stop = threading.Event()

    def _run():
        while not stop.is_set():
            samples.append({
                "cpu_pct": psutil.cpu_percent(),
                "ram_gb":  psutil.virtual_memory().used / 1e9,
            })
            stop.wait(_SAMPLE_INTERVAL)

    threading.Thread(target=_run, daemon=True).start()
    return stop, samples


def _summarise_resources(samples: list) -> dict:
    if not samples:
        return {"cpu_avg": 0, "cpu_peak": 0, "ram_avg": 0, "ram_peak": 0}
    return {
        "cpu_avg":  round(sum(s["cpu_pct"] for s in samples) / len(samples), 1),
        "cpu_peak": round(max(s["cpu_pct"] for s in samples), 1),
        "ram_avg":  round(sum(s["ram_gb"]  for s in samples) / len(samples), 2),
        "ram_peak": round(max(s["ram_gb"]  for s in samples), 2),
    }


# ── Answer parsing ────────────────────────────────────────────────────────────

def _parse_answer(raw: str) -> str | None:
    raw = raw.strip()
    if raw in ("0", "1", "2", "3"):
        return raw
    matches = re.findall(r'\b([0-3])\b', raw)
    return matches[-1] if matches else None


_THEREFORE_RE = re.compile(
    r'\bTherefore\s+my\s+answer\s+is\s+option\s+([0-3])\b',
    re.IGNORECASE,
)


def _extract_therefore(text: str) -> str | None:
    matches = _THEREFORE_RE.findall(text)
    return matches[-1] if matches else None


# ── True/False combo ──────────────────────────────────────────────────────────

_VERDICT_RE = re.compile(r"VERDICT\s+S(\d+)\s*:\s*(TRUE|FALSE)", re.IGNORECASE)
_STMT_VERDICT_RE = re.compile(
    r"[Ss]tatement\s+(\d+)\s+is\s+\*{0,2}(True|False)\*{0,2}", re.IGNORECASE
)


def _extract_tf_verdicts(text: str) -> dict[int, str] | None:
    verdicts: dict[int, str] = {}
    for m in _VERDICT_RE.finditer(text):
        verdicts[int(m.group(1))] = m.group(2).lower()
    if len(verdicts) < 2:
        for m in _STMT_VERDICT_RE.finditer(text):
            n = int(m.group(1))
            if n not in verdicts:
                verdicts[n] = m.group(2).lower()
    return verdicts if len(verdicts) >= 2 else None


def _match_tf_to_option(verdicts: dict[int, str], options: dict) -> str | None:
    s1, s2 = verdicts.get(1, ""), verdicts.get(2, "")
    matches = []
    for k, v in options.items():
        tf_vals = re.findall(r"\b(true|false)\b", str(v), re.IGNORECASE)
        if len(tf_vals) >= 2 and tf_vals[0].lower() == s1 and tf_vals[1].lower() == s2:
            matches.append(str(k))
    return matches[0] if len(matches) == 1 else None


_STATEMENT_SPLIT_RE = re.compile(r"[Ss]tatement\s+(\d+)\s*[|:]\s*([^.!?]+[.!?]?)")
_IRREDUCIBLE_STMT_RE = re.compile(
    r"(.+?)\s+is\s+irreducible\s+over\s+([ZQR](?:_\d+)?|\bthe\s+\w+s?\b)",
    re.IGNORECASE,
)


def _python_verify_statement(stmt: str) -> str | None:
    m = _IRREDUCIBLE_STMT_RE.search(stmt)
    if not m:
        return None
    poly_str, domain_raw = m.group(1).strip(), m.group(2).strip().upper()
    if "INT" in domain_raw or domain_raw == "Z":
        domain = "'Z'"
    elif "RAT" in domain_raw or domain_raw == "Q":
        domain = "'Q'"
    elif re.match(r"Z_(\d+)", domain_raw):
        domain = re.match(r"Z_(\d+)", domain_raw).group(1)
    else:
        return None
    result = _run_python_tool(f"print(is_irreducible_over({poly_str}, x, {domain}))").strip().lower()
    return result if result in ("true", "false") else None


def _wolfram_verify_statements(question: str, verdicts: dict[int, str]) -> dict[int, str]:
    statements: dict[int, str] = {}
    for m in _STATEMENT_SPLIT_RE.finditer(question):
        statements[int(m.group(1))] = m.group(2).strip().rstrip(".")
    if not statements:
        return verdicts
    corrected = dict(verdicts)
    for n, stmt in statements.items():
        if n not in verdicts:
            continue
        py_verdict = _python_verify_statement(stmt)
        if py_verdict is not None:
            logging.info("Python verify S%d (%r) → %s", n, stmt[:60], py_verdict)
            if py_verdict != verdicts[n]:
                logging.info("Python corrects S%d: model=%s → python=%s", n, verdicts[n], py_verdict)
                corrected[n] = py_verdict
    return corrected


# ── Special instructions ──────────────────────────────────────────────────────

_TRUE_FALSE_COMBO = re.compile(r"\b(only|both|neither|true|false)\b", re.IGNORECASE)

_TRUE_FALSE_INSTRUCTION = (
    "SPECIAL INSTRUCTION — this is a true/false combination question. "
    "Evaluate each statement independently. "
    "After your analysis, end with a verdict block in exactly this format:\n"
    "  VERDICT S1: TRUE\n"
    "  VERDICT S2: FALSE\n"
    "(Use TRUE or FALSE in capitals. One per line. No other text on those lines.)\n"
    "The code will match your verdicts to the correct option automatically."
)

_NOT_QUESTION = re.compile(r"\bNOT\b")

_NOT_INSTRUCTION = (
    "SPECIAL INSTRUCTION — this is a NOT question: the correct answer is the ONE "
    "option that does NOT belong to the category asked about. "
    "Strategy: (1) for each option, ask 'Is this genuinely a member of the category "
    "in the question?' — if YES, eliminate it; "
    "(2) the option you cannot eliminate is the answer."
)


def _is_true_false_combo(options: dict) -> bool:
    return sum(1 for v in options.values() if _TRUE_FALSE_COMBO.search(str(v))) >= 2


# ── ASR artifact repair ───────────────────────────────────────────────────────

_ASR_DE_ARTIFACT_RE = re.compile(
    r"\bde(higher|lower|more|less|better|worse|greater|fewer|"
    r"smaller|larger|bigger|faster|slower|longer|shorter|"
    r"wider|narrower|deeper|shallower|harder|softer|"
    r"stronger|weaker|brighter|darker|louder|quieter|"
    r"heavier|lighter|earlier|later|closer|farther)\b",
    re.IGNORECASE,
)


def _fix_speech_option(text: str) -> str:
    return _ASR_DE_ARTIFACT_RE.sub(lambda m: m.group(1), text)


_EMBEDDED_LABEL = re.compile(r"^\s*[A-Da-d][).]\s*")


def _clean_option(text: str) -> str:
    return _EMBEDDED_LABEL.sub("", str(text))


# ── Message builder ───────────────────────────────────────────────────────────

def _build_messages(question, options, context: str | None = None, speech_mode: bool = False):
    if speech_mode:
        opts = {k: _fix_speech_option(_clean_option(v)) for k, v in options.items()}
    else:
        opts = {k: _clean_option(v) for k, v in options.items()}

    parts = []
    if context:
        label = ("RETRIEVED CONTEXT (PRIMARY SOURCE — extract the key fact first, "
                 "then find which option matches it):\n"
                 if speech_mode else
                 "RETRIEVED CONTEXT (supporting evidence — use alongside your knowledge):\n")
        parts.append(label + context)

    if speech_mode:
        if _NOT_QUESTION.search(question):
            parts.append(_NOT_INSTRUCTION)
    else:
        if _NOT_QUESTION.search(question):
            parts.append(_NOT_INSTRUCTION)
        elif _is_true_false_combo(opts):
            parts.append(_TRUE_FALSE_INSTRUCTION)

    parts.append(
        f"QUESTION = {question},\n"
        f"OPTION 0 = {opts[0]},\n"
        f"OPTION 1 = {opts[1]},\n"
        f"OPTION 2 = {opts[2]},\n"
        f"OPTION 3 = {opts[3]}"
    )
    system = SPEECH_SYSTEM_PROMPT if speech_mode else SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": "\n".join(parts)},
    ]


# ── Streaming helpers ─────────────────────────────────────────────────────────

def _write(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def _hold_size(buf, tag):
    for i in range(min(len(tag) - 1, len(buf)), 0, -1):
        if buf.endswith(tag[:i]):
            return i
    return 0


def _write_thinking(text, start, color_state):
    color = _RED if time.time() - start >= _COLOR_THRESHOLD else _GREEN
    if color != color_state["color"]:
        _write(_RESET + color)
        color_state["color"] = color
    _write(text)


def _drain(buf, in_think, answer_buf, thinking_buf, headers, start, color_state):
    while buf:
        tag = "</think>" if in_think else "<think>"
        idx = buf.find(tag)
        if idx != -1:
            segment = buf[:idx]
            if segment:
                if in_think:
                    _write_thinking(segment, start, color_state)
                    thinking_buf += segment
                else:
                    if not headers["answer"]:
                        _write("\n[Answer]\n")
                        headers["answer"] = True
                    _write(segment)
                    answer_buf += segment
            buf = buf[idx + len(tag):]
            if in_think:
                if color_state["color"] is not None:
                    _write(_RESET)
                    color_state["color"] = None
                _write("\n")
                in_think = False
            else:
                if not headers["thinking"]:
                    _write("\n[Thinking]\n")
                    headers["thinking"] = True
                in_think = True
        else:
            hold = _hold_size(buf, tag)
            safe = buf[:len(buf) - hold] if hold else buf
            if safe:
                if in_think:
                    _write_thinking(safe, start, color_state)
                    thinking_buf += safe
                else:
                    if not headers["answer"]:
                        _write("\n[Answer]\n")
                        headers["answer"] = True
                    _write(safe)
                    answer_buf += safe
            buf = buf[len(buf) - hold:] if hold else ""
            break
    return buf, in_think, answer_buf, thinking_buf


def _collect_stream(stream, start: float):
    """Consume a llama-cpp-python streaming response, parsing <think> tags."""
    buf = ""
    in_think = False
    answer_buf = ""
    thinking_buf = ""
    headers = {"thinking": False, "answer": False}
    color_state = {"color": None}
    output_tokens = 0

    for chunk in stream:
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content") or ""
        if content:
            output_tokens += 1  # rough token estimate (1 chunk ≈ 1 token)
            buf += content
            buf, in_think, answer_buf, thinking_buf = _drain(
                buf, in_think, answer_buf, thinking_buf, headers, start, color_state
            )

        elapsed = time.time() - start
        if elapsed >= _SOFT_TIMEOUT and _parse_answer(answer_buf.strip()):
            logging.warning("Soft timeout at %.1fs — stopping stream", elapsed)
            break
        if elapsed >= _HARD_TIMEOUT:
            logging.warning("Hard timeout at %.1fs — stopping stream", elapsed)
            break

    if buf:
        if in_think:
            _write_thinking(buf, start, color_state)
            thinking_buf += buf
        else:
            if not headers["answer"]:
                _write("\n[Answer]\n")
            _write(buf)
            answer_buf += buf

    if color_state["color"] is not None:
        _write(_RESET)

    reasoning_out = thinking_buf.strip() if thinking_buf.strip() else answer_buf.strip()
    return answer_buf.strip(), reasoning_out, 0, output_tokens


# ── RAG ───────────────────────────────────────────────────────────────────────

def _fetch_rag_context(question: str, options: dict) -> str | None:
    try:
        from rag import should_use_rag, fetch_rag_context
    except ImportError:
        logging.warning("rag.py not found — skipping RAG")
        return None
    if not should_use_rag(question):
        logging.info("RAG skipped: question type not suitable")
        return None
    options_dict = {i: str(v) for i, v in options.items()}
    return fetch_rag_context(question=question, options=options_dict, k=5)


# ── Core inference ────────────────────────────────────────────────────────────

def _ask_ollama(question, options, context: str | None = None,
                speech_mode: bool = False) -> tuple[str, str, dict]:
    """Run inference with llama-cpp-python (streaming)."""
    llm = _get_llm()
    messages = _build_messages(question, options, context, speech_mode=speech_mode)

    logging.info("Streaming response from %s %s...", MODEL_PATH, "[speech]" if speech_mode else "")
    start = time.time()
    stop, samples = _start_resource_sampler()

    extra = {}
    if speech_mode:
        extra["max_tokens"] = 1024

    stream = llm.create_chat_completion(
        messages=messages,
        stream=True,
        temperature=0.3,
        top_p=0.95,
        top_k=64,
        **extra,
    )

    raw, reasoning, _, output_tokens = _collect_stream(stream, start)
    stop.set()

    elapsed = time.time() - start
    tps = output_tokens / elapsed if elapsed > 0 else 0
    resources = _summarise_resources(samples)
    resources["tps"]     = round(tps, 1)
    resources["elapsed"] = round(elapsed, 2)

    if not raw:
        logging.warning("Stream returned empty output — using placeholder")
        raw = "[no output]"

    answer = _parse_answer(raw)
    if answer is None:
        logging.warning("Could not parse answer from %r — defaulting to '0'", raw)
        answer = "0"

    therefore = _extract_therefore(raw)
    if therefore and therefore != answer:
        logging.warning("Digit/Therefore mismatch: parsed=%s therefore=%s — trusting Therefore",
                        answer, therefore)
        answer = therefore

    if _is_true_false_combo(options):
        verdicts = _extract_tf_verdicts(raw)
        if verdicts:
            verdicts = _wolfram_verify_statements(question, verdicts)
            tf_answer = _match_tf_to_option(verdicts, options)
            if tf_answer is not None:
                if tf_answer != answer:
                    logging.info("T/F combo override: %s → %s", answer, tf_answer)
                answer = tf_answer
            else:
                logging.warning("T/F combo: no unique option match — keeping %s", answer)
        else:
            logging.warning("T/F combo: could not extract verdicts — keeping model answer")

    logging.info("Response in %.2fs: answer=%s | ~%d tok | %.1f tok/s | "
                 "CPU avg=%.1f%% | RAM avg=%.2fGB",
                 elapsed, answer, output_tokens, tps,
                 resources["cpu_avg"], resources["ram_avg"])
    _write("\n\n")
    return answer, reasoning, resources


# ── Python tool execution ─────────────────────────────────────────────────────

_CODE_PREAMBLE = """\
import math
import numpy as np
import scipy.stats as stats
from scipy import integrate as sci_integrate
from sympy import (
    symbols, Function,
    integrate, diff, solve, limit, series,
    pi, E, sqrt, ln, log, exp,
    sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
    Matrix, det, eye, zeros, ones,
    factorial, binomial,
    Rational, Integer, oo,
    simplify, expand, factor, collect,
    Eq, And, Or, Not,
    re as Re, im as Im,
    Abs, conjugate, arg,
    floor, ceiling, Mod,
    Sum, Product, summation,
    Float,
)
from sympy.combinatorics import (
    Permutation, PermutationGroup,
    SymmetricGroup, AlternatingGroup,
    CyclicGroup, DihedralGroup,
)
from sympy.ntheory import (factorint, isprime, nextprime,
                           primitive_root, is_primitive_root,
                           totient, n_order)
x, y, z, t, n, k, a, b, c = symbols('x y z t n k a b c')

from sympy import Poly as _Poly
from sympy.polys.domains import ZZ as _ZZ, QQ as _QQ

def is_irreducible_over(expr, var, domain):
    if domain in ('Z', 'ZZ'):
        p = _Poly(expr, var, domain=_ZZ)
        content, factors = p.factor_list()
        if abs(int(content)) != 1:
            return False
        return len(factors) == 1 and factors[0][1] == 1 and factors[0][0].degree() == p.degree()
    elif domain in ('Q', 'QQ'):
        p = _Poly(expr, var, domain=_QQ)
        _, factors = p.factor_list()
        return len(factors) == 1 and factors[0][1] == 1 and factors[0][0].degree() == p.degree()
    else:
        p = _Poly(expr, var, modulus=int(domain))
        return p.is_irreducible
"""


def _run_python_tool(code: str, timeout: int = 15) -> str:
    import ast
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}\n  {(e.text or '').rstrip()}"

    full_code = _CODE_PREAMBLE + "\n" + code
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout.strip()
        errors = result.stderr.strip()
        if errors:
            user_lines = [l for l in errors.splitlines()
                          if not ("line " in l and any(f"line {i}" in l for i in range(1, 41)))]
            errors = "\n".join(user_lines).strip()
        if errors and not output:
            return f"RuntimeError:\n{errors[:600]}"
        if errors:
            return f"{output}\n(stderr: {errors[:300]})"
        if output:
            return output
        fixed = _autofix_missing_print(code)
        if fixed != code:
            r2 = subprocess.run([sys.executable, "-c", _CODE_PREAMBLE + "\n" + fixed],
                                capture_output=True, text=True, timeout=timeout)
            if r2.stdout.strip():
                return r2.stdout.strip()
        return "(no output — did you forget print()?)"
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {timeout}s"
    except Exception as exc:
        return f"Error: {exc}"


def _autofix_missing_print(code: str) -> str:
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body:
        return code
    last = tree.body[-1]
    if not isinstance(last, _ast.Expr):
        return code
    if (isinstance(last.value, _ast.Call) and
            isinstance(last.value.func, _ast.Name) and
            last.value.func.id == "print"):
        return code
    lines = code.splitlines()
    expr_src = "\n".join(lines[last.lineno - 1:]).strip()
    prefix = "\n".join(lines[:last.lineno - 1])
    return (prefix + "\n" if prefix else "") + f"print({expr_src})"


# ── Math tool path ────────────────────────────────────────────────────────────

_COMPUTATIONAL = re.compile(
    r"\b(calculate|compute|"
    r"standard deviation|variance|expected value|"
    r"integral|integrate|derivative|differentiate|antiderivative|"
    r"eigenvalue|eigenvector|determinant|"
    r"p-value|confidence interval|normally distributed|z-score|t-score|"
    r"binomial|poisson distribution|geometric distribution|"
    r"modulo|modular inverse|imaginary|complex number|i\^[0-9]|"
    r"factor(i[zs]ation|i[zs]e)?|irreducible|"
    r"minimi[zs]e?|maximiz[zs]e?|lagrange|"
    r"minimum value|maximum value|critical point(s)?|linear programming|"
    r"permutation|combination|how many ways|"
    r"primitive root|generator of|"
    r"order of (the |an? |element )?group|order of (the |an? )?element|"
    r"finite field|galois field|euler|totient|"
    r"greatest common (divisor|factor)|least common multiple)\b"
    r"|\b(find|compute|determine|what\s+is)\s+the\s+"
    r"(volume|area|surface\s+area|arc\s+length|probability|expected\s+value|"
    r"eigenvalue|eigenvector|determinant|standard\s+deviation|variance|mean|"
    r"minimum|maximum|factori[zs]ation|roots?|zeros?|generator|primitive root|order|gcd|lcm)\b"
    r"|\bprobability\s+(that|of)\b|\bP\s*\("
    r"|\bsubject\s+to\s+(the\s+)?constraint\b"
    r"|[Zz]_\d+|\bGF\s*\(\s*\d+"
    r"|what\s+is\s+the\s+(sum|product|value|probability)\s+of\b"
    r"|i\^[0-9]|\bsystem\s+of\s+(linear\s+)?equations\b",
    re.IGNORECASE,
)

_CONCEPTUAL = re.compile(
    r"\bStatement\s+\d+\s*[|:]"
    r"|\bappropriate\s+(model|test|distribution|method|for|when)\b"
    r"|\bnot\s+appropriate\b"
    r"|\bwhich\s+(assumption|condition|property|of\s+the\s+following\s+is\s+(true|false|correct|best))\b"
    r"|\bassumption(s)?\s+of\b"
    r"|\bunder\s+what\s+condition\b"
    r"|\bwhen\s+(is|should|would|can)\b.{0,40}\b(used?|applied?|appropriate)\b"
    r"|\bwhich\s+of\s+the\s+following\s+(best\s+)?(describe|explain|represent|illustrate|justify|is\s+an\s+example)\b"
    r"|\bwhy\s+(is|are|would|does)\b"
    r"|\bbecause\b.{0,60}\b(skew|outlier|normal|distribut|sample\s+size)\b",
    re.IGNORECASE,
)

_MATH_SYSTEM_PROMPT = (
    "You are solving a multiple-choice math or statistics problem.\n"
    "All libraries are pre-loaded — DO NOT import anything. "
    "Variables x, y, z, t, n, k, a, b, c are already defined as sympy symbols.\n\n"
    "Use run_python to compute the answer precisely, then identify which option it matches.\n"
    "OPTION MATCHING — computed result is a VALUE; the answer digit is an INDEX (0/1/2/3).\n"
    "ALWAYS scan the option texts first, then output the matching index.\n"
    "Write 'Therefore my answer is option X.' then X as a bare digit (0-3)."
)

_MATH_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute Python code for mathematical computation. "
            "All imports pre-loaded. Symbols x y z t n k a b c pre-defined. "
            "Always use print() to output the result."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to execute"}},
            "required": ["code"],
        },
    },
}

_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def _build_option_reminder(tool_result: str, opts: dict) -> str:
    base = (
        f"{tool_result}\n\n"
        f"Options: 0={opts[0]!r}, 1={opts[1]!r}, 2={opts[2]!r}, 3={opts[3]!r}\n"
        f"Find which option VALUE matches your result, then output that option's INDEX."
    )
    try:
        result_val = float(tool_result.strip().splitlines()[-1])
        matched = any(
            abs(float(str(v)) - result_val) / max(abs(result_val), 1e-9) < 0.05
            for v in opts.values()
            if _parse_answer(str(v)) is not None or True
        )
        if not matched:
            base += ("\n\nWARNING: result does not match any numeric option. "
                     "Reconsider your approach before guessing.")
    except (ValueError, TypeError, IndexError):
        pass
    return base


def _ask_math(question, options) -> tuple[str, str, dict]:
    """Run mathematical inference using the Python tool."""
    llm = _get_llm()
    opts = {k: _clean_option(v) for k, v in options.items()}

    user_content = (
        f"QUESTION = {question}\n"
        f"OPTION 0 = {opts[0]}\nOPTION 1 = {opts[1]}\n"
        f"OPTION 2 = {opts[2]}\nOPTION 3 = {opts[3]}\n\n"
        "Compute the answer with Python, then output the matching option INDEX (0/1/2/3)."
    )

    _text_system = (
        _MATH_SYSTEM_PROMPT + "\n\nTOOL USE (text mode): write Python inside ```python blocks. "
        "The system will execute them and return results. Up to 4 blocks allowed."
    )

    messages = [
        {"role": "system", "content": _MATH_SYSTEM_PROMPT if _SUPPORTS_TOOLS else _text_system},
        {"role": "user",   "content": user_content},
    ]

    start = time.time()
    stop, samples = _start_resource_sampler()
    raw_response = ""
    reasoning_parts: list[str] = []
    MAX_CALLS = 4

    for call_num in range(MAX_CALLS):

        if _SUPPORTS_TOOLS:
            # ── Native tool-call path ─────────────────────────────────────
            resp = llm.create_chat_completion(
                messages=messages,
                tools=[_MATH_TOOL],
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2048,
            )
            msg = resp["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                # Append assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": tool_calls,
                })
                for tc in tool_calls:
                    fn   = tc["function"]
                    name = fn["name"]
                    args_raw = fn.get("arguments", "{}")
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw

                    if name == "run_python":
                        code = args.get("code", "")
                        logging.info("Math tool (call %d):\n%s", call_num + 1, code)
                        tool_result = _run_python_tool(code)
                        logging.info("Math tool result: %r", tool_result)
                        _write(f"\n[Python]\n{code}\n[Result] {tool_result}\n")
                        reasoning_parts.append(f"[Code]\n{code}\n[Result]\n{tool_result}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", f"call_{call_num}"),
                            "content": _build_option_reminder(tool_result, opts),
                        })
            else:
                raw_response = (msg.get("content") or "").strip()
                reasoning_parts.append(raw_response)
                break

        else:
            # ── Text-based tool-call path ─────────────────────────────────
            resp = llm.create_chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                stream=True,
            )
            text = ""
            for chunk in resp:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                text += delta.get("content") or ""
                _write(delta.get("content") or "")

            reasoning_parts.append(text)
            code_blocks = _CODE_BLOCK_RE.findall(text)

            if not code_blocks:
                raw_response = text.strip()
                break

            results = []
            for code in code_blocks:
                code = code.strip()
                tool_result = _run_python_tool(code)
                _write(f"\n[Result] {tool_result}\n")
                reasoning_parts.append(f"[Code]\n{code}\n[Result]\n{tool_result}")
                results.append(_build_option_reminder(tool_result, opts))

            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": "Python results:\n" + "\n\n".join(results) + "\n\nNow give your final answer.",
            })
            if call_num == MAX_CALLS - 1:
                raw_response = text.strip()

    stop.set()
    elapsed = time.time() - start
    resources = _summarise_resources(samples)
    resources["elapsed"] = round(elapsed, 2)
    resources["tps"] = 0.0

    if not raw_response:
        raw_response = "[no output]"

    answer = _parse_answer(raw_response)
    therefore = _extract_therefore(raw_response)
    if therefore and therefore != answer:
        answer = therefore
    if answer is None:
        logging.warning("Math: could not parse answer — defaulting to '0'")
        answer = "0"

    reasoning = "\n\n".join(reasoning_parts)
    _write(f"\n[Math answer] {answer}\n\n")
    logging.info("Math response in %.2fs → answer=%s", elapsed, answer)
    return answer, reasoning, resources


# ── Public entry point ────────────────────────────────────────────────────────

def ask_gemma(question, options, context: str | None = None,
              speech_mode: bool = False) -> tuple[str, str, dict]:
    """Run inference and return (answer_digit, reasoning, resources).

    Same public API as the Ollama version — speech_client.py works unchanged.
    """
    _is_computational = _COMPUTATIONAL.search(question) and not _CONCEPTUAL.search(question)
    if not speech_mode and _is_computational and _SUPPORTS_TOOLS:
        logging.info("Computational question — routing to Python tool path")
        return _ask_math(question, options)

    if context is None and ENABLE_RAG and not speech_mode:
        context = _fetch_rag_context(question, options)

    return _ask_ollama(question, options, context, speech_mode=speech_mode)
