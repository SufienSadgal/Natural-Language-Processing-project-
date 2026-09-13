"""Web RAG pipeline.

Steps:
  1. Clean the question into two query variants (question-only + enriched)
  2. Two parallel DuckDuckGo searches → pooled results
  3. Snippets kept immediately as guaranteed fallback chunks
  4. Parallel full-page fetch + Wikipedia extracts + trafilatura extraction
  5. Sentence-level chunking with 1-sentence overlap
  6. Cross-encoder re-ranked with [question + options, chunk] for better
     option-aware relevance; Wikipedia gets a near-tie preference bonus
  7. Top-2 chunks returned (if both pass threshold) as context

Rules compliance:
  - DuckDuckGo is a free API (no key, no cost)
  - Returns raw document text, NOT an LLM-generated answer
  - Must be mentioned in the video per assignment rules
"""

import re
import logging
import concurrent.futures

import requests
import trafilatura
from ddgs import DDGS
from sentence_transformers import CrossEncoder

_WIKI_API = "https://en.wikipedia.org/w/api.php"

_TIMEOUT = 5
_HEADERS = {"User-Agent": "PoliMillionaire-RAG/1.0 (NLP course project; open-source)"}
_MIN_CHUNK_LEN = 40
_MAX_CHUNK_LEN = 800          # reranker max is ~512 tokens ≈ 1500 chars
_MAX_WIKI_CHUNKS = 60         # cap Wikipedia sentences per query (full article, not just intro)
_MAX_PAGE_CHUNKS = 20         # cap full-page sentences per URL
_RAG_WALL_TIMEOUT = 8.0       # abort entire RAG pipeline after this many seconds
_SCORE_THRESHOLD = 1.5
_TOP_K_CHUNKS = 2             # return up to this many chunks if all pass threshold
_MAX_RERANK_CHUNKS = 80       # cap before cross-encoder to keep reranking under ~1.5s

# Wikipedia near-tie preference: only beats a web chunk when scores are close.
# Formula: bonus = min(_WIKI_BONUS_CAP, max(0, _SCORE_THRESHOLD - web_raw + 0.1))
# In practice this adds at most 0.6 to Wikipedia scores, not a flat 1.0.
_WIKI_BONUS_CAP = 0.6

# Penalty applied when a chunk does not mention any entity from the question.
# Prevents generic glossary chunks (e.g. "what is a contralto") from passing
# purely because they contain option keywords, while genuine subject-focused
# chunks (e.g. a Nolan biography that mentions "Nolan") are unaffected.
# Gate threshold for per-option score boosting.
# A chunk must score at least this value against the question OR the entity
# query before its option scores are considered. Chunks that fail the gate
# (i.e. genuinely unrelated to the question subject) fall back to their raw
# question score, preventing generic glossary chunks from gaming the reranker
# by matching option keywords (e.g. a voice-type glossary for a Whitney Houston
# vocal range question).
_OPTION_BOOST_GATE = -1.0

# User-generated / low-reliability domains
_BLOCKED_DOMAINS = {
    "quora.com", "reddit.com", "yahoo.com", "answers.yahoo.com",
    "stackexchange.com", "stackoverflow.com", "tripadvisor.com",
    "yelp.com", "facebook.com", "twitter.com", "x.com",
    "forums.com", "forum.", "discuss.", "community.",
}

# Loaded once at import time
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Question-word prefixes that hurt search-engine recall
_QUESTION_PREFIXES = re.compile(
    r"^\s*(which of the following|what is|what are|what was|what were|"
    r"who is|who was|who were|how many|how much|how did|when did|"
    r"where did|in which|name the)\s*",
    re.IGNORECASE,
)

# Sentence boundary splitter
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

# Signals that a question is mathematical/logical — RAG won't help
_MATHEMATICAL = re.compile(
    r"\b(calculate|compute|solve|equation|probability|"
    r"what is the value|what is the sum|what is the product|"
    r"if .{0,40} equals|standard deviation|mean|median|integral|"
    r"derivative|percent|ratio|proportion|expected value|"
    r"imaginary|complex number|modulus|argument|conjugate|"
    r"i\^[0-9]|\bi\b.*\bsum\b|eigenvalue|eigenvector|"
    r"normally distributed|p-value|hypothesis|confidence interval|"
    r"isomorphism|abelian|modulo|"
    # Abstract algebra theorem questions: testing pure knowledge of group/ring
    # axioms — web search returns irrelevant philosophy/logic articles.
    r"in a group|in a ring|in a field|subgroup|normal subgroup|"
    r"ring homomorphism|group homomorphism|kernel|ideal|coset|"
    r"cyclic group|quotient group|factor group)\b",
    re.IGNORECASE,
)

_AMBIGUOUS_NOUN = re.compile(
    r"\bthe (band|film|movie|show|series|book|song|artist|author|"
    r"character|company|team|group|album|episode|game|painting|play)\b",
    re.IGNORECASE,
)


def _has_ambiguous_ref(question: str) -> bool:
    for m in _AMBIGUOUS_NOUN.finditer(question):
        rest = question[m.end():]
        if re.match(r"\s+[A-Z]", rest):
            continue
        return True
    return False


def should_use_rag(question: str) -> bool:
    """Return True if RAG is likely to help for this question."""
    if _MATHEMATICAL.search(question):
        logging.info("RAG skipped: mathematical/logical question")
        return False
    return True


# ── internal helpers ──────────────────────────────────────────────────────────

def _entity_query(question: str) -> str:
    """Extract a short entity-focused query for Wikipedia.

    Picks capitalised words that appear mid-sentence (i.e. proper nouns).
    Falls back to the stripped base question if nothing is found.
    Example: "What is the fundamental principle that drives Mark Grayson's
              transformation into Invincible?" → "Mark Grayson Invincible"

    Special case — "Statement N | content" questions: strips the structural
    labels first so the entity scan runs on the mathematical content, not
    on "Statement" (which produces a useless Wikipedia search).
    """
    # Strip "Statement N |" labels before scanning for entities
    question = re.sub(r"[Ss]tatement\s+\d+\s*[|:]\s*", " ", question).strip()
    words = question.split()
    entities = []
    for i, word in enumerate(words):
        if i == 0:
            continue  # skip sentence-initial cap
        clean = re.sub(r"[^\w']", "", word, flags=re.UNICODE)  # keep letters + apostrophe
        clean = clean.lstrip("'")              # drop leading quote (e.g. 'Lemonade)
        clean = re.sub(r"'s?$", "", clean)     # drop possessive 's or trailing '
        if clean and clean[0].isupper() and len(clean) > 2:
            entities.append(clean)
    if entities:
        return " ".join(dict.fromkeys(entities))[:200]  # deduplicate, cap length
    return _QUESTION_PREFIXES.sub("", question).strip(" ,?")[:200]


def _make_queries(question: str, options: dict | None = None, speech_mode: bool = False) -> list[str]:
    """Return query variants: question-only, and (in text mode) option-enriched.

    Two queries double source diversity at zero extra time (run in parallel).
    The enriched query steers search toward the specific concept in the options.
    In speech mode the enriched query is skipped: option text is garbled and
    would pollute the DuckDuckGo search with noise.

    Special case — "Statement N | content" questions:
    The structural labels "Statement 1 |", "Statement 2 |" add noise without
    meaning. Strip them so the search is over the actual mathematical content.
    e.g. "Statement 1 | 4x-2 is irreducible over Z. Statement 2 | ..."
         → "4x-2 is irreducible over Z. 4x-2 is irreducible over Q."
    which hits Wikipedia's "Irreducible polynomial" article directly.
    """
    # Strip "Statement N |" / "Statement N:" labels if present
    stripped = re.sub(r"[Ss]tatement\s+\d+\s*[|:]\s*", " ", question).strip()
    base = _QUESTION_PREFIXES.sub("", stripped).strip(" ,?")
    queries = [base[:250]]  # query 1: question only

    if options and not speech_mode:
        option_text = " ".join(str(v) for v in options.values())
        enriched = f"{base} {option_text[:120]}"
        if enriched[:250] != queries[0]:
            queries.append(enriched[:250])
        if _has_ambiguous_ref(question):
            logging.info("RAG: ambiguous reference — enriched query with options")
        else:
            logging.info("RAG: running enriched query alongside base query")
    elif speech_mode:
        logging.info("RAG: speech mode — skipping enriched query (options are garbled)")

    return queries


def _search(query: str, k: int) -> list[dict]:
    """Return top-k DuckDuckGo results as list of {url, snippet}."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=k))
    return [{"url": r["href"], "snippet": r.get("body", "")} for r in results]


def _fetch_text(url: str) -> tuple[str, str] | None:
    """Fetch a URL and extract its main text. Returns (url, text) or None."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        text = trafilatura.extract(
            resp.text,
            include_comments=False,
            include_tables=False,
        )
        if text and len(text) > 100:
            return url, text
    except Exception as exc:
        logging.debug("Fetch failed for %s: %s", url, exc)
    return None


def _chunk(text: str, max_chunks: int | None = None) -> list[str]:
    """Split text into overlapping sentence windows, capped and truncated.

    Uses a 2-sentence sliding window with 1-sentence overlap so no key fact
    falls between chunks. Each window is truncated to _MAX_CHUNK_LEN chars.
    """
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if len(s.strip()) >= 15]
    if not sentences:
        # Fallback: paragraph split
        return [p.strip()[:_MAX_CHUNK_LEN]
                for p in text.split("\n")
                if len(p.strip()) >= _MIN_CHUNK_LEN][:max_chunks or 9999]

    window = 3   # sentences per chunk
    step   = 2   # overlap of 1 sentence
    chunks = []
    for i in range(0, max(1, len(sentences) - window + 1), step):
        piece = " ".join(sentences[i:i + window]).strip()
        if len(piece) >= _MIN_CHUNK_LEN:
            chunks.append(piece[:_MAX_CHUNK_LEN])
    if not chunks and sentences:
        chunks = [s[:_MAX_CHUNK_LEN] for s in sentences if len(s) >= _MIN_CHUNK_LEN]

    return chunks[:max_chunks] if max_chunks else chunks


def _is_low_quality(url: str) -> bool:
    url_lower = url.lower()
    return any(domain in url_lower for domain in _BLOCKED_DOMAINS)


def _wikipedia_chunks(query: str, n: int = 3) -> list[tuple[str, str]]:
    """Search Wikipedia and return up to n articles' intro chunks."""
    try:
        search = requests.get(_WIKI_API, headers=_HEADERS, params={
            "action": "query",
            "list": "search",
            "srsearch": query[:200],
            "srlimit": n,
            "format": "json",
        }, timeout=_TIMEOUT).json()

        titles = [r["title"] for r in search.get("query", {}).get("search", [])]
        if not titles:
            return []

        extract = requests.get(_WIKI_API, headers=_HEADERS, params={
            "action": "query",
            "prop": "extracts",
            "exlimit": n,
            "titles": "|".join(titles),
            "explaintext": True,
            "format": "json",
        }, timeout=_TIMEOUT).json()

        chunks = []
        per_page = max(1, _MAX_WIKI_CHUNKS // max(n, 1))  # e.g. 60 // 3 = 20 chunks per article
        for page in extract.get("query", {}).get("pages", {}).values():
            title = page.get("title", "")
            text  = page.get("extract", "").strip()
            if text:
                url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
                for c in _chunk(text, max_chunks=per_page):
                    chunks.append((url, c))
        return chunks

    except Exception as exc:
        logging.debug("Wikipedia chunks failed: %s", exc)
        return []


def _wiki_bonus(raw_score: float) -> float:
    """Near-tie Wikipedia preference bonus.

    Adds at most _WIKI_BONUS_CAP, only when the web score is near the
    threshold — so a highly relevant web chunk still wins over a mediocre
    Wikipedia one.
    """
    return min(_WIKI_BONUS_CAP, max(0.0, _SCORE_THRESHOLD - raw_score + 0.1))


# ── public API ────────────────────────────────────────────────────────────────

def fetch_rag_context(
    question: str,
    options: dict | None = None,
    k: int = 5,
    speech_mode: bool = False,
) -> str | None:
    """Search the web, re-rank all chunks, return top-1 or top-2 as context.

    Args:
        question:    Quiz question text.
        options:     Answer options dict {0: text, 1: text, ...}.
        k:           Number of pages to search and fetch per query.
        speech_mode: If True, skip garbled-option enriched query and per-option
                     reranking (both would add noise rather than signal).

    Returns:
        The most relevant paragraph(s) as a single string, or None.
    """
    import time as _time
    _wall_start = _time.time()

    def _timed_out() -> bool:
        return _time.time() - _wall_start >= _RAG_WALL_TIMEOUT

    # Compute once — used for both Wikipedia lookup and reranking gate.
    # In speech mode the options are garbled, but capitalized proper nouns
    # survive transcription well enough to enrich the Wikipedia entity lookup.
    # We extract them from the options and append to the entity query so that
    # anonymous questions ("the character's building") can still find the right
    # Wikipedia article (e.g. "Leonard" → Big Bang Theory).
    entity_query = _entity_query(question)
    if options:
        option_entities = []
        for v in options.values():
            for w in str(v).split():
                clean = re.sub(r"[^\w']", "", w)
                clean = re.sub(r"'s?$", "", clean).lstrip("'")
                if clean and len(clean) > 2 and clean[0].isupper() and clean.lower() not in {
                    # Common function words
                    "the", "this", "that", "they", "their", "there",
                    "and", "but", "for", "with", "from", "into", "onto",
                    # Structural True/False combo answer words — not content entities.
                    # Including these in the entity query causes Wikipedia to return
                    # articles on logic/philosophy ("False statement") instead of
                    # the subject matter (e.g. group theory).
                    "true", "false", "both", "neither", "none", "only",
                    "all", "some", "not", "just", "also", "either",
                    "above", "below", "only", "statement", "option",
                }:
                    option_entities.append(clean)
        if option_entities:
            extra = " ".join(dict.fromkeys(option_entities))[:150]
            combined = f"{entity_query} {extra}".strip()[:250]
            if combined != entity_query:
                logging.info("RAG: enriched entity query with option nouns: %s", combined)
                entity_query = combined

    queries = _make_queries(question, options, speech_mode=speech_mode)
    logging.info("RAG queries: %s", queries)

    # ── Step 1: Two parallel DuckDuckGo searches ──────────────────────────────
    all_results: list[dict] = []
    seen_urls: set[str] = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as pool:
        search_futures = [pool.submit(_search, q, k) for q in queries]
        for f in concurrent.futures.as_completed(search_futures):
            try:
                for r in f.result():
                    if r["url"] not in seen_urls:
                        seen_urls.add(r["url"])
                        all_results.append(r)
            except Exception as exc:
                logging.warning("DuckDuckGo search failed: %s", exc)

    if not all_results or _timed_out():
        logging.info("RAG: no search results or timed out after search")
        return None

    # ── Step 2: Collect chunks in parallel ───────────────────────────────────
    all_chunks: list[tuple[str, str]] = []
    seen_chunks: set[str] = set()
    urls = [r["url"] for r in all_results]

    def _add_chunk(url: str, text: str):
        key = text[:80]
        if key not in seen_chunks:
            seen_chunks.add(key)
            all_chunks.append((url, text))

    # entity_query already computed above — reuse for Wikipedia lookup.
    # E.g. "Mark Grayson Invincible" beats "fundamental principle drives transformation"

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls) + 1) as pool:
        wiki_future  = pool.submit(_wikipedia_chunks, entity_query)
        page_futures = [pool.submit(_fetch_text, url) for url in urls]

        # DDG snippets (instant)
        for r in all_results:
            if _is_low_quality(r["url"]):
                continue
            snippet = r["snippet"].strip()[:_MAX_CHUNK_LEN]
            if len(snippet) >= _MIN_CHUNK_LEN:
                _add_chunk(r["url"], snippet)

        # Wikipedia chunks — use a bounded timeout so a slow Wikipedia API
        # call cannot block the main thread past the wall budget.
        if not _timed_out():
            wiki_timeout = max(0.5, _RAG_WALL_TIMEOUT - (_time.time() - _wall_start))
            try:
                for url, c in wiki_future.result(timeout=wiki_timeout):
                    _add_chunk(url, c)
            except concurrent.futures.TimeoutError:
                logging.warning(
                    "RAG: Wikipedia timed out after %.1fs — skipping",
                    _time.time() - _wall_start,
                )

        # Full page text
        for future in concurrent.futures.as_completed(page_futures):
            if _timed_out():
                logging.warning("RAG: wall timeout — stopping page fetch early")
                break
            result = future.result()
            if result:
                url, text = result
                if _is_low_quality(url):
                    continue
                for c in _chunk(text, max_chunks=_MAX_PAGE_CHUNKS):
                    _add_chunk(url, c)

    logging.info("RAG: %d chunks to re-rank (%.1fs elapsed)",
                 len(all_chunks), _time.time() - _wall_start)

    if not all_chunks:
        logging.info("RAG: no chunks found")
        return None
    if _timed_out():
        # Don't discard already-collected chunks — reranking is fast (~1s).
        # The expensive work (fetching) already happened; throwing it away wastes time.
        logging.warning("RAG: past wall timeout but %d chunks ready — proceeding with rerank",
                        len(all_chunks))

    # Cap chunks before reranking: snippets and Wikipedia come first in all_chunks
    # (insertion order), so truncating keeps the highest-quality sources.
    if len(all_chunks) > _MAX_RERANK_CHUNKS:
        logging.info("RAG: capping %d chunks → %d before rerank", len(all_chunks), _MAX_RERANK_CHUNKS)
        all_chunks = all_chunks[:_MAX_RERANK_CHUNKS]

    # ── Step 3: Gated per-option reranking ───────────────────────────────────
    # Queries: [question, entity_query, opt0, opt1, opt2, opt3]
    #   - question   : full question text
    #   - entity     : proper-noun extract (e.g. "Whitney Houston") — acts as
    #                  a subject-relevance gate
    #   - options    : each answer option individually
    #
    # Gate logic:
    #   gate_score = max(question_score, entity_score)
    #   if gate_score >= _OPTION_BOOST_GATE:
    #       score = max(gate_score, max(option_scores))   ← option boost allowed
    #   else:
    #       score = question_score                         ← locked out
    #
    # This prevents generic glossary chunks (e.g. "what is a contralto") from
    # passing the threshold purely by matching option keywords, while still
    # allowing genuine subject chunks that don't mention the question framing
    # (e.g. "Nolan is renowned for...") to be boosted by option scoring.
    # entity_query already computed at function entry — reuse here.
    rerank_queries = [question.strip()[:400], entity_query[:200]]
    if options and not speech_mode:
        # In speech mode option text is garbled — including it in reranking adds
        # noise and misleads the cross-encoder. Skip per-option scoring entirely.
        for v in options.values():
            rerank_queries.append(str(v)[:200])
    elif speech_mode:
        logging.info("RAG: speech mode — skipping per-option reranking (garbled text)")

    n_q = len(rerank_queries)
    flat_pairs  = [[q, c] for (_, c) in all_chunks for q in rerank_queries]
    flat_scores = _reranker.predict(flat_pairs)

    scores = []
    for i in range(len(all_chunks)):
        chunk_scores = flat_scores[i * n_q : (i + 1) * n_q]
        q_score      = float(chunk_scores[0])   # question
        e_score      = float(chunk_scores[1])   # entity
        opt_scores   = chunk_scores[2:]          # one per option
        gate         = max(q_score, e_score)
        if gate >= _OPTION_BOOST_GATE:
            scores.append(max(gate, float(max(opt_scores))) if len(opt_scores) else gate)
        else:
            scores.append(q_score)              # option boost locked out

    # Apply near-tie Wikipedia bonus
    boosted = [
        s + (_wiki_bonus(s) if "wikipedia.org" in url else 0.0)
        for s, (url, _) in zip(scores, all_chunks)
    ]

    # Sort by boosted score descending
    ranked = sorted(
        zip(boosted, scores, all_chunks),
        key=lambda x: x[0],
        reverse=True,
    )

    best_boosted, best_raw, (best_url, best_chunk) = ranked[0]

    logging.info(
        "RAG: best score=%.3f (boosted=%.3f, threshold=%.1f, wiki=%s) | elapsed=%.1fs",
        best_raw, best_boosted, _SCORE_THRESHOLD,
        "wikipedia.org" in best_url, _time.time() - _wall_start,
    )

    if best_raw < _SCORE_THRESHOLD:
        logging.info("RAG: score too low — skipping context")
        print(f"[RAG] No useful context found (best score: {best_raw:.2f}) — skipping\n")
        return None

    # ── Step 4: Collect top-K chunks that pass threshold ──────────────────────
    top_chunks = []
    for _, raw, (url, chunk) in ranked[:_TOP_K_CHUNKS * 3]:  # look at up to 3x candidates
        if raw >= _SCORE_THRESHOLD and len(top_chunks) < _TOP_K_CHUNKS:
            # Skip near-duplicate of already accepted chunk
            if not any(chunk[:60] == existing[:60] for existing in top_chunks):
                top_chunks.append(chunk)
                logging.info("RAG: accepted chunk from %s (raw=%.3f)", url, raw)

    context = "\n\n".join(top_chunks)

    print(f"[RAG] Source : {best_url}")
    print(f"[RAG] Score  : {best_raw:.2f}")
    print(f"[RAG] Chunks : {len(top_chunks)}")
    print(f"[RAG] Context: {context[:300]}...\n" if len(context) > 300 else f"[RAG] Context: {context}\n")

    return context
