# Philosophy for Everyone

**让哲学说人话 — without letting it make things up.**

A retrieval-augmented philosophy companion for the terminal. It answers questions
about philosophy using only passages retrieved from real primary texts, shows you
exactly which book and chapter each claim came from, and explains everything twice:
once in ordinary language, once with the argument laid out properly.

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
.venv/bin/python scripts/fetch_library.py    # download 14 public-domain works
.venv/bin/python -m philo ingest             # chunk, embed, index (~3s offline)
.venv/bin/python -m philo ask "what is actually within my control?"
```

That runs entirely on the built-in **mock provider** — no network, no key, no cost.
Retrieval is real; generation is extractive rather than written. It exists so you
can verify the whole pipeline before a single API call (see *Offline first* below).

To use a real model, add a key and re-index:

```bash
cp .env.example .env        # then set OPENAI_API_KEY, or the AZURE_OPENAI_* set
.venv/bin/python -m philo doctor --probe   # verify the key actually works
.venv/bin/python -m philo ingest --rebuild # re-embed with the real model
.venv/bin/python -m philo ask "why should I not fear death?"
```

`--rebuild` is required when you switch providers, and the index refuses to load
if you forget: vectors from different embedding models are not comparable, and
comparing them produces confident nonsense rather than an error.

---

## Commands

| Command | What it does |
|---|---|
| `philo ask "question"` | Answer from the texts, with sources |
| `philo chat` | Conversation that remembers the last few turns |
| `philo daily` | Today's personalised "Daily Philosophy" |
| `philo search "query"` | Retrieval only — see exactly what the model would be sent |
| `philo sources` | What's in the library |
| `philo ingest` | Chunk, embed and index `library/` |
| `philo profile show\|list\|set` | Who the daily piece is written for |
| `philo doctor [--probe]` | Check configuration, index and connectivity |
| `philo init` | Create `.env`, a profile, and a first index |

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
| `PHILO_PROVIDER` | auto | `mock` \| `openai` \| `azure` |
| `PHILO_CHAT_MODEL` | `gpt-4o` | OpenAI model name |
| `PHILO_EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
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
make test                     # 78 tests, offline, no key required
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
  providers/           mock (offline) · openai · azure — one interface
  corpus/              loader · argument-aware chunker · ingest pipeline
  store/               local vector store, three inspectable files
  retrieval/           hybrid dense + BM25, MMR diversity, relevance floor
  generation/          prompts, grounding, two-layer parsing, marker audit
  personalize/         profiles and the daily piece
  ui/                  theme, components, views (Rich)
  cli.py               the eight commands
scripts/fetch_library.py   builds library/ from Project Gutenberg
```

`rich` is the only hard dependency. `numpy` is optional (≈50× faster search;
pure-Python fallback otherwise) and `openai` is only needed for real providers.

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
