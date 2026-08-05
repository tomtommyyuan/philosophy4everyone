# Philosophy for Everyone

**让哲学说人话 — without letting it make things up.**

A retrieval-augmented philosophy companion, in your terminal or your browser. It
answers questions about philosophy using only passages retrieved from real primary
texts, shows you exactly which book and chapter each claim came from, and explains
everything twice: once in ordinary language, once with the argument laid out
properly.

```
philo ask "why should I not fear death?"
```

```
╭─ ◗ In plain words ───────────────────────────────────────────────────────────╮
│                                                                              │
│  Socrates' answer is not that death is fine. It is that nobody knows what    │
│  death is, so fearing it means being confident about something you have      │
│  never checked — "a pretence of knowing the unknown" [1].                    │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─ ◗ The argument ─────────────────────────────────────────────────────────────╮
│                                                                              │
│  The claim is epistemic rather than consolatory. To fear death is to treat   │
│  it as the greatest evil, which presumes knowledge of what death is; and     │
│  "no one knows whether death... may not be the greatest good" [1].           │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

 #   Philosopher       Work                     Section              Relevance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1   Plato             Apology                  —                    1.00 █████▉
 2   Epictetus         Discourses and the E…    That Confidence i…   0.96 █████▊
 3   Laozi             Tao Te Ching             Chapter 74           0.87 █████▏
```

<sub><i>The layout, citation markers, source table and retrieved passages above are
real output. The prose inside the two panels is illustrative — the exact wording
depends on which model you point it at. Run it yourself to see the actual answer;
offline mode quotes the retrieved sentences directly instead of writing around
them.</i></sub>

---

## The problem this solves

Ask a language model what Epictetus thought about anxiety and you will get a
fluent, confident, plausible answer. Some of it will be right. Some of it will be
a paraphrase of a self-help book about Stoicism. Occasionally it will be a
quotation he never wrote, attributed to a chapter that does not exist.

That failure is quiet, which is what makes it bad. There is no way to tell the
three cases apart by reading the output.

So this system is built the other way round:

1. **Nothing is asserted without a retrieved source.** The model is shown a set of
   real passages and instructed that any philosophical claim outside them is
   forbidden. If retrieval finds nothing relevant, the model is *never called* —
   you get an explicit "not in this library" instead of a confident guess.
2. **Every claim carries its citation.** `[1]`, `[2]` markers in the prose map to a
   numbered table of philosopher, work and chapter. Markers pointing at sources
   that do not exist are stripped, and you are told it happened.
3. **Two layers, always.** Plain language for someone who has never read
   philosophy; the reconstructed argument for someone who wants the real thing.
   Neither register is asked to do the other's job.

---

## Quick start

Offline, no API key, about thirty seconds:

```bash
make setup                                   # venv + install
.venv/bin/python -m philo fetch              # download 14 public-domain works
.venv/bin/python -m philo ingest             # chunk, embed, index (~3s offline)
.venv/bin/python -m philo ask "what is actually within my control?"
.venv/bin/python -m philo serve --open       # …or use it in a browser
```

Or install it as a normal command-line tool, no checkout required:

```bash
pipx install "git+https://github.com/tomtommyyuan/philosophy4everyone"
philo fetch && philo ingest        # texts and index land in ~/.philo
philo ask "why do we fear death?"
```

That runs entirely on the built-in **mock provider** — no network, no key, no cost.
Retrieval is real; generation is extractive rather than written. It exists so you
can verify the whole pipeline before a single API call (see *Offline first* below).

To use a real model, export **any one** of these and re-index:

```bash
export OPENAI_API_KEY=sk-...           # chat + embeddings
export ANTHROPIC_API_KEY=sk-ant-...    # chat only — see below
export GEMINI_API_KEY=...              # chat + embeddings
# or the four AZURE_OPENAI_* variables

.venv/bin/python -m philo doctor --probe   # verify the keys actually work
.venv/bin/python -m philo ingest --rebuild # re-embed with the real model
.venv/bin/python -m philo ask "why should I not fear death?"
```

An API key is now **required**: with nothing configured the CLI refuses to start
and prints the exact exports above, rather than quietly answering from the mock.
`PHILO_PROVIDER=mock` still selects offline mode explicitly.

`--rebuild` is required when you switch providers, and the index refuses to load
if you forget: vectors from different embedding models are not comparable, and
comparing them produces confident nonsense rather than an error.

---

## Providers

OpenAI, Azure OpenAI, Anthropic and Gemini, plus the offline mock. Export a key
and philo picks it up — nothing else to configure.

| Provider | Chat | Embeddings | Key |
|---|:---:|:---:|---|
| OpenAI | ✅ | ✅ | `OPENAI_API_KEY` |
| Azure OpenAI | ✅ | ✅ | `AZURE_OPENAI_*` (four variables) |
| Anthropic (Claude) | ✅ | ❌ | `ANTHROPIC_API_KEY` |
| Google Gemini | ✅ | ✅ | `GEMINI_API_KEY` |
| mock | ✅ | ✅ | none — fully offline |

### Chat and embeddings are resolved separately

**Anthropic publishes no embeddings endpoint.** The Messages API is its entire
surface, so "use Claude" cannot mean "use Claude for everything" — Claude writes
the answers and somebody else builds the index. Rather than paper over that with
an `embed()` that throws halfway through an ingest, the provider layer resolves
the two roles independently and pairs them:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...            # or GEMINI_API_KEY
export PHILO_CHAT_PROVIDER=anthropic    # → claude-opus-5 + text-embedding-3-small
```

`philo doctor` always names both halves, because only one of them constrains
your index: **vectors belong to the embedding provider.** Swapping the chat model
is free; swapping the embedding model requires `philo ingest --rebuild`, and the
store refuses to load a mismatched index rather than silently comparing vectors
from different spaces.

Left unset, providers are auto-detected from whichever keys exist, preferring
`azure → openai → anthropic → gemini` for chat and `azure → openai → gemini` for
embeddings. `PHILO_CHAT_PROVIDER` and `PHILO_EMBED_PROVIDER` override either half.

### Per-provider quirks, handled

Each vendor breaks a different assumption, and each is handled in its adapter
rather than leaking into the pipeline:

- **Claude rejects `temperature`.** The Claude 5 family removed sampling
  parameters — sending one is a 400, not a warning. Depth is set with
  `PHILO_ANTHROPIC_EFFORT` (`low`…`max`) instead.
- **Claude's `max_tokens` covers thinking *and* the answer.** Thinking is on by
  default, so a budget sized for the visible text truncates mid-sentence; the
  adapter raises a floor.
- **Gemini embeds documents and queries asymmetrically.** Passages go in as
  `RETRIEVAL_DOCUMENT`, questions as `RETRIEVAL_QUERY`; using the document mode
  for a query measurably hurts recall. This is why the embedding interface has
  had a separate `embed_query` from the start.
- **Refusals and safety blocks are HTTP 200.** Claude returns
  `stop_reason: "refusal"`, Gemini returns a candidate-less response. Both would
  render as a blank answer if unchecked, so both become readable text.

---

## Commands

| Command | What it does |
|---|---|
| `philo ask "question"` | Answer from the texts, with sources |
| `philo serve` | Web interface at `http://localhost:8000` |
| `philo fetch` | Download the public-domain texts into the library |
| `philo chat` | Conversation that remembers the last few turns |
| `philo daily` | Today's personalised "Daily Philosophy" |
| `philo search "query"` | Retrieval only — see exactly what the model would be sent |
| `philo sources` | What's in the library |
| `philo ingest` | Chunk, embed and index `library/` |
| `philo profile show\|list\|set` | Who the daily piece is written for |
| `philo doctor [--probe]` | Check configuration, index and connectivity |
| `philo init` | Create `.env`, a profile, and a first index |

Where things live: inside a checkout, the library and index sit beside the code.
Installed via pip there is no checkout, so both go to `~/.philo` (override with
`PHILO_HOME`, `PHILO_LIBRARY`, `PHILO_INDEX`).

Useful flags:

```bash
philo ask "what is virtue?" --philosopher Aristotle   # restrict to one thinker
philo ask "what is virtue?" --tradition Stoicism      # or one tradition
philo ask "what is virtue?" --show-sources            # print the full passages
philo ask "what is virtue?" --plain                   # everyday layer only
philo ask "what is virtue?" --json                    # machine-readable
philo search "wu wei" -k 10 --full                    # inspect retrieval directly
philo daily --profile yucheng --theme "grief"
```

`philo search` is the debugging tool worth knowing: it shows the blended score,
the embedding score and the BM25 score for each passage, so when an answer looks
wrong you can tell whether retrieval or generation is at fault.

---

## The web interface

```bash
philo serve --open          # http://localhost:8000
```

The same engine, same prompts, same grounding — rendered as a page instead of a
terminal. Two-layer answers, amber citation markers that jump to the passage
they cite, expandable source text so a claim is one click from its evidence, and
answers that stream token by token. It follows your system light/dark setting
and is one self-contained HTML file with no external requests, so it works
offline and inside a strict content-security policy.

Questions live in the URL (`/?q=why+do+we+fear+death`), so an answer is
shareable.

### Choosing a model

The filter row has a model picker, grouped by provider and remembered between
visits. It lists only providers whose keys are actually configured, discovered
live from each vendor's own model list (cached, with a curated shortlist when
that list is unreachable) so it never offers a model your key cannot reach.

**Only the chat model is selectable, deliberately.** Embeddings are fixed by
whatever built the index; putting them in a dropdown would let one click
invalidate the whole library. The picker changes who writes the answer — the
retrieval underneath is identical, which also makes it a fair way to compare
models on the same evidence.

The CLI takes the same flags:

```bash
philo ask "why do we fear death?" --model claude-sonnet-5 --chat-provider anthropic
```

On a public deployment, `PHILO_WEB_MODELS` caps what visitors may spend on:

```bash
export PHILO_WEB_MODELS="gpt-4o-mini,claude-haiku-4-5"
```

The API is a plain ASGI app (`philo.web.app:app`) and is documented at
`/api/docs`:

| Endpoint | |
|---|---|
| `GET /api/models` | chat models this installation can reach |
| `POST /api/ask` | grounded answer as JSON |
| `POST /api/ask/stream` | the same, as server-sent events |
| `GET /api/daily` | today's personalised piece |
| `GET /api/search` | retrieval only, with scores |
| `GET /api/sources` | what is indexed |
| `GET /api/health` | provider, index and passage counts |

## Deploying

Full guide in **[deploy/README.md](deploy/README.md)**. The short version:

```bash
docker build -t philo . && docker run --rm -p 8000:8000 philo
```

A container is the right shape for this app — it is a stateful index plus a
small server, and the image bakes the texts and an offline index in at build
time so a cold start answers immediately. That works on Fly.io, Render,
Railway and Cloud Run, all of which honour the `PORT` the image reads.

Vercel works too (`vercel.json` and `api/index.py` are included), but a
serverless function has no persistent disk, so the index must be built locally
and committed:

```bash
make deploy-index && git add -f deploy/index
vercel --prod
```

> **Before you put this on a public URL:** every request spends *your* API
> credits, and this app has no per-user metering. Set `PHILO_WEB_TOKEN` to a
> random string and the API will require it as an `X-Philo-Token` header — the
> page has a field for it. `philo serve --host 0.0.0.0` warns you when it is
> unset. For a demo that cannot cost anything, deploy with
> `PHILO_PROVIDER=mock`.

---

## Design decisions

### Offline first

The `mock` provider implements the same interface as OpenAI and Azure: hashed
bag-of-words embeddings and an extractive composer that arranges real retrieved
sentences into the two-layer format. It never invents philosophy.

This is not a toy — it is a debugging discipline. When something breaks in mock
mode it is a bug in this code; when it breaks *only* against a real API it is a
key, network or model problem. Keeping those two failure classes separable is
worth the extra interface.

*Limitation, stated plainly:* mock embeddings are lexical, so they cannot match
across languages. A Chinese question against an English library degrades to noise
in offline mode, and the CLI says so when it detects it. Real embedding models are
multilingual and do not have this problem.

### Chunking by argument, not by character count

Fixed-width chunking is the default everywhere and it is wrong for philosophy.
Cut at character 800 and you land between a premise and its conclusion. The
retrieved passage then says *"therefore the soul is immortal"* with the reasoning
stranded in a neighbouring chunk that never gets retrieved — and the model fills
the gap from memory. Fixed-width chunking manufactures the exact hallucination
this project exists to prevent.

So `philo/corpus/chunker.py` splits on the author's own boundaries: section
headings first, then whole paragraphs, and only when a paragraph exceeds the
ceiling does it split *between sentences*, carrying one sentence of overlap so a
conclusion keeps its premise. Fragments below the minimum are merged into a
neighbour, because a two-line chunk retrieves well and explains nothing. Length is
measured with a CJK-aware weight, since one Chinese character carries far more
than one Latin one.

### Hybrid retrieval with a floor

Three things beyond nearest-neighbour lookup, each because dense retrieval fails
philosophy in a specific way:

- **BM25 blending.** Terms of art — *eudaimonia*, *noumenon*, 無為 — are exactly
  where embeddings are weakest and exact matching is strongest.
- **MMR diversity + a per-work cap.** The top six by cosine are frequently six
  near-identical paragraphs from one chapter. That looks like six sources and is
  really one — a citation illusion.
- **A relevance floor.** Cosine always returns *something*. A ranked list is not
  evidence that the library contains an answer. Below `PHILO_MIN_SCORE` the system
  returns nothing and says so.

### Metadata is what makes citation possible

Every chunk carries philosopher, work, translator, section, tradition, era and
rights, denormalised so retrieval results are self-describing. The section label
comes from the text's own headings, which is why a citation can say
*"Meditations, The Ninth Book"* rather than just *"Meditations"*.

### The index is three inspectable files

```
.philo/index/
  manifest.json   what produced this index (provider, model, dimension)
  chunks.jsonl    one JSON object per chunk — greppable, diffable
  vectors.npy     float32 matrix, row i ↔ line i of chunks.jsonl
```

No server, no Docker, no opaque binary format. `grep` works.

---

## The library

`library/` holds fourteen complete works, all public domain, downloaded verbatim
from Project Gutenberg by `scripts/fetch_library.py`:

| | | |
|---|---|---|
| Marcus Aurelius · *Meditations* | Epictetus · *Discourses & Encheiridion* | Plato · *Apology* |
| Aristotle · *Nicomachean Ethics* | Kant · *Groundwork* | Mill · *Utilitarianism* |
| Mill · *On Liberty* | Descartes · *Discourse on the Method* | Hume · *Enquiry* |
| Nietzsche · *Beyond Good and Evil* | Laozi · *Tao Te Ching* | Confucius · *Analects* |
| Russell · *Problems of Philosophy* | Wollstonecraft · *Vindication of the Rights of Woman* | |

**Why a fetcher rather than committed excerpts.** This system's whole claim is that
its quotations are real. Text written from memory — human or model — cannot make
that claim. So nothing here is transcribed; it is downloaded from a canonical
edition and only *structurally* cleaned:

- Gutenberg's licence header and footer are removed.
- Editorial apparatus is removed — translators' introductions, tables of contents,
  appendices, glossaries, indexes. These are the editor's words, not the
  philosopher's, and keeping them would produce citations attributing a Victorian
  scholar's opinion to Marcus Aurelius.
- Footnote markers like `[12]` are stripped, since they are indistinguishable from
  this system's own citation markers once a passage reaches the model.
- Book and chapter headings become Markdown `##`, which is what makes the section
  label in a citation possible.

Not one word of the philosophical text itself is altered. Translator attributions
were verified against each downloaded file rather than assumed — PG #2680, for
instance, is Meric Casaubon's 1634 translation, which the file states in its own
notes, not the George Long translation it is often assumed to be.

### Adding your own texts

Drop a Markdown file in `library/` with front matter and run `philo ingest`:

```markdown
---
philosopher: Zhuangzi
philosopher_zh: 庄子
work: Inner Chapters
work_zh: 内篇
translator: James Legge
tradition: Daoism
era: Ancient
rights: public-domain
tags: [freedom, dreams, relativity, uselessness]
---

## Chapter 2

Once upon a time, I, Zhuangzi, dreamt I was a butterfly...
```

Plain `.txt` files work too — metadata is inferred from a
`Philosopher - Work.txt` filename. Re-ingesting is incremental: chunk ids embed a
content hash, so editing one file re-embeds only that file.

---

## Daily Philosophy

```bash
philo profile set --name yucheng --language zh --level curious \
  --interests "焦虑, 自由意志, 工作的意义" \
  --philosophers "Marcus Aurelius, Laozi" \
  --tone "平实、有具体例子、稍带一点幽默"

philo daily --profile yucheng
```

A profile is a small JSON file in `profiles/` — deliberately, so it is editable by
hand and obvious what the system knows about you. Theme selection is
*deterministic given (date, profile)*: the same person on the same day always gets
the same piece, re-running is free, and "why did I get this today" has an answer.
Variation comes from the date, not from randomness, which also makes it testable.

The profile shapes **examples, tone and length**. It never shapes what the sources
are taken to say. Personalisation that bends philosophy toward what the reader
wants to hear is the same failure as hallucination, wearing friendlier clothes —
so the guard against it is written into the prompt itself.

---

## Configuration

Everything is environment variables, read from `.env`. See
[.env.example](.env.example) for the annotated set. The ones worth knowing:

| Variable | Default | Meaning |
|---|---|---|
| `PHILO_PROVIDER` | auto | `openai` \| `azure` \| `anthropic` \| `gemini` \| `mock` |
| `PHILO_CHAT_PROVIDER` | auto | Override just the chat half |
| `PHILO_EMBED_PROVIDER` | auto | Override just the embedding half |
| `PHILO_CHAT_MODEL` | `gpt-5.5` | OpenAI model name |
| `PHILO_EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `PHILO_ANTHROPIC_MODEL` | `claude-opus-5` | Claude model |
| `PHILO_ANTHROPIC_EFFORT` | `medium` | Claude reasoning depth (no `temperature`) |
| `PHILO_GEMINI_MODEL` | `gemini-2.5-flash` | Gemini chat model |
| `PHILO_GEMINI_EMBED_MODEL` | `gemini-embedding-001` | Gemini embedding model |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | — | Azure **deployment** name, not model name |
| `PHILO_TOP_K` | `6` | Passages sent to the model |
| `PHILO_HYBRID_ALPHA` | `0.72` | 1.0 = pure embeddings, 0.0 = pure BM25 |
| `PHILO_MIN_SCORE` | `0.12` | Below this: "not in this library" |
| `PHILO_CHUNK_TARGET` | `900` | Target chunk size (CJK-weighted characters) |

Azure routes by *deployment name*, which is not the model name. That is the most
common first-run failure, so `philo doctor` checks for it by name and the provider
turns the resulting 404 into a message that says so.

---

## Development

```bash
make test                     # 120 tests, offline, no key required
.venv/bin/python -m pytest tests -q
```

The suite is hermetic: temporary library, temporary index, mock provider, no
network. The tests worth reading are the ones that encode the actual promise —
that an unanswerable question never reaches the model, and that invented citation
markers are stripped before a reader ever sees them.

```
philo/
  config.py            settings, provider auto-detection
  models.py            Work, Chunk, ScoredChunk, Answer, DailyPiece
  providers/           mock · openai · azure · anthropic · gemini
                       chat and embeddings are separate protocols
  corpus/              loader · argument-aware chunker · ingest pipeline
  store/               local vector store, three inspectable files
  retrieval/           hybrid dense + BM25, MMR diversity, relevance floor
  generation/          prompts, grounding, two-layer parsing, marker audit
  personalize/         profiles and the daily piece
  ui/                  theme, components, views (Rich)
  web/                 ASGI app + the single-page interface
  corpus/gutenberg.py  the Project Gutenberg fetcher (`philo fetch`)
  cli.py               the commands
Dockerfile             container image, index baked in
api/ + vercel.json     serverless adapter
```

`rich` is the only hard dependency; every provider SDK is an optional extra, so
you install only the vendors you use:

```bash
pip install "philo[openai]"     # or [anthropic], [gemini], [web], [fast], [all]
```

`numpy` is optional too (≈50× faster search; pure-Python fallback otherwise).

---

## Honest limitations

- **Retrieval quality is bounded by the library.** Fourteen works is a good
  starting shelf, not the history of philosophy. Nothing after 1929 is public
  domain, so contemporary philosophy is absent entirely.
- **Translations are old.** Public-domain means Victorian, mostly. Casaubon's
  Marcus Aurelius is beautiful and archaic; Legge's Laozi is a scholarly artefact
  of 1891. The system paraphrases them into modern language but the underlying
  text is of its period, with the biases that implies.
- **Offline mode cannot match across languages** (see *Offline first*).
- **Grounding is enforced structurally, not perfectly.** Retrieval floor, marker
  auditing and prompt constraints make unsourced claims hard; they do not make
  them impossible. `--show-sources` is one keystroke away for a reason.

## Licence

MIT for the code. Every text in `library/` is public domain, sourced from Project
Gutenberg; `philo sources` reports the rights status of everything indexed.
