# langcheck — Implementation Plan

Oct 28, 2026 · Tyler Allen

## Overview

langcheck checks spelling and grammar with an LLM. It is a simpler sibling of
`contcheck`: same error taxonomy, same scope (spelling and grammar only, never style or
content), but no trained tagger. The LLM finds the errors and proposes fixes. A masked
language model ranks the fixes.

Three ideas shape the design:

1. **Edits, not rewrites.** The LLM never returns corrected text. It returns a list of
   issues, each anchored to a span of the original, with a few candidate replacements.
   The server finds each span in the original itself and rejects any edit it cannot
   place. A model that rewrites the paragraph cannot slip changes through.
2. **Document context, local chunks.** A short context summary of the whole document
   (register, dialect, names, intentional nonstandard spellings) is built once and sent
   with every chunk. Chunks are small, so calls are fast and cache well, but the model
   still knows that "Bilbo" is a name and that the narrator writes in UK English.
3. **Scored options.** Each issue offers 1 to 4 candidate replacements. A BERT-family
   scorer rates the original sentence and each candidate sentence and the options are
   ranked by how much they improve on it. The scorer may also drop issues, but only for
   error kinds where evaluation shows that helps (see *Scoring options*).

One shared FastAPI service does all the checking. Three clients use it: a `langcheck` CLI,
VS Code (through a thin LSP proxy) and LibreOffice (through a UNO proofreader extension).

## Decisions

Settled in a design review on Oct 28, 2026. The sections below give the details.

| # | Area | Decision |
| --- | --- | --- |
| 1 | Clients | VS Code, LibreOffice and the CLI are the first clients. |
| 2 | VS Code | A thin LSP server (`langcheck lsp`) forwards to the FastAPI service. The service holds the cache, models and sessions, so every client sees the same results. |
| 3 | LibreOffice | The proofreader returns cached results at once, checks changed paragraphs in the background and has LibreOffice re-check them when results arrive. A spike confirms this first. |
| 4 | LLM | Any OpenAI-compatible API. The reply schema goes in the prompt as well as `json_schema`; replies are validated with Pydantic and retried once. |
| 5 | Scorer | Always ranks options. Drops issues only for error kinds the M6 evaluation clears; filtering is off until then. |
| 6 | Deployment | A shared server, used by several people and machines. |
| 7 | Authentication | Configurable through an `Authenticator` protocol. Built in: `api_key` (default) and `none`. |
| 8 | State and scaling | One uvicorn process. Sessions and a per-user result cache in memory behind protocols; Redis or SQL adapters later. |
| 9 | CLI path | Calls the service by default; `--local` (or no server configured) runs the checker in process. |
| 10 | Dictionary | Per document, behind a `DictionaryStore` protocol with an HTTP API and several implementations. SQL (SQLAlchemy, SQLite or Postgres) by default. |
| 11 | Document identity | The client's URI, scoped by user. Unsaved documents get a temporary ID and nothing persisted. |
| 12 | Order | CLI → service → LLM first; then the scorer; then sessions, LSP and VS Code; then LibreOffice. |
| 13 | Python clients | Subpackages under `src/langcheck/`. The LibreOffice code imports only the standard library and `uno`. |
| 14 | VS Code extension | TypeScript in `editors/vscode/` at the repo root, built with `just vscode`. |
| 15 | Settings | `pydantic-settings` with nested models. Precedence: values passed to the settings object (CLI options), then environment variables, then a YAML file, then defaults. CLI with `cyclopts`, service with FastAPI. |

## Architecture

```text
 langcheck CLI ──────────────┐            (--local: Checker in process)
 VS Code ── langcheck lsp ───┼── HTTPS ──► reverse proxy (TLS) ──► langcheck serve
 LibreOffice ── .oxt UNO ────┘                                       │
                                                ┌────────────────────┼─────────────────┐
                                                │ Authenticator      │ DocumentStore   │
                                                │ Checker ── LLM ────┼─► OpenAI-compat │
                                                │ Scorer (thread)    │ ResultCache     │
                                                │ DictionaryStore ───┼─► SQLite / PG   │
                                                └────────────────────┴─────────────────┘
```

The service is the only place that talks to the LLM or loads scorer models (except the
CLI's `--local` mode and evaluation). Clients only translate between their editor and
the HTTP API.

## Pipeline

For each changed chunk of a document:

1. **Segment.** Split the text into paragraphs and sentences, keeping character offsets.
   Each sentence gets a stable ID within the request (`S1`, `S2`, ...) and a content hash.
2. **Context.** Use the document's `DocumentContext`, building or refreshing it if the
   document changed enough since the last build (see *Document context*).
3. **Chunk.** Group the changed sentences into chunks of a few sentences (about 150–300
   words). Each chunk carries the paragraph before and after as read-only context.
4. **Ask.** Send the context summary, the read-only neighbours and the numbered focus
   sentences to the LLM (see *LLM contract*).
5. **Resolve.** For each issue, find `original` inside the named sentence (with
   `occurrence` when it appears more than once) and turn it into document offsets. Reject
   issues that do not resolve, that change nothing, whose kind is not in the taxonomy, or
   whose edit is too large for a spelling or grammar fix (see *Validation*).
6. **Dictionary.** Drop issues on words in the document's dictionary.
7. **Score.** Build each candidate sentence, score the original and the candidates, and
   rank the options. Drop issues only for kinds with filtering turned on.
8. **Publish.** Store the suggestions against the document version and send them to
   the clients watching it.

Results are cached per user and sentence: key = hash(sentence, neighbouring sentences,
context version, model, prompt version, dictionary). The context version is a fingerprint
of its content. An unchanged sentence is never sent again. The cache is
not shared between users: hits across users would be rare, and a shared cache would hint
at what others have checked.

## LLM contract

The prompt (versioned, kept in `src/langcheck/prompts/` so it ships inside the package) tells the model to:

- fix only spelling and grammar, using the kinds below; leave style, word choice that is
  not wrong, and content alone;
- respect the document context: names, dialect, register, and nonstandard spellings the
  context marks as intentional;
- quote the smallest span that needs to change, exactly as written;
- give 1 to 4 replacements for that span, best first; give more than one only when they
  are really different fixes (for example "they're" and "they are").

Reply schema (Pydantic models at the boundary):

```json
{
  "issues": [
    {
      "sentence": "S3",
      "original": "their going",
      "occurrence": 1,
      "kind": "Apostrophe",
      "explanation": "'their' is possessive; the contraction of 'they are' is needed.",
      "options": ["they're going", "they are going"]
    }
  ]
}
```

langcheck works with any OpenAI-compatible server (Ollama, llama.cpp, vLLM, hosted APIs).
They differ in whether they honour `response_format` with a JSON schema, so the checker
never relies on it alone: the schema is sent as `json_schema` and also described in the
prompt, every reply is validated with Pydantic, and a reply that fails validation is
retried once with the validation error appended. A chunk that fails twice is reported as
unchecked, not as clean.

What the API returns (after resolution and scoring):

```json
{
  "document_version": 7,
  "suggestions": [
    {
      "id": "sg_9f2c",
      "start": 412,
      "end": 423,
      "original": "their going",
      "kind": "Apostrophe",
      "operation": "R",
      "explanation": "...",
      "options": [
        {"replacement": "they're going", "score": 0.91, "delta": 3.4},
        {"replacement": "they are going", "score": 0.88, "delta": 2.9}
      ]
    }
  ],
  "unchecked": []
}
```

Offsets are in Unicode code points of the text the client sent; clients convert to their
own units (LSP uses UTF-16 positions by default). `operation` (Missing, Unnecessary,
Replacement) is derived from the edit, not asked of the model.

### Taxonomy

Copy `contcheck.taxonomy.kinds` as is: the 19 `ErrorKind` values (Spelling, RealWord,
Capitalization, WordBoundary, Apostrophe, NounNumber, SubjectVerbAgreement, VerbTense,
VerbForm, AdjectiveAdverb, PronounForm, Determiner, Preposition, Conjunction, WordOrder,
MissingWord, ExtraWord, SentenceBoundary, Punctuation), `Operation` and `Formality`.
Keeping the two projects on one taxonomy lets their results be compared on the same
evaluation sets. `Formality` lets callers hide informal-only issues.

### Validation

An issue is rejected, and counted in metrics, when:

- `sentence` is unknown or `original` is not found in it (allowing for whitespace and
  quote-style normalisation only);
- a replacement equals the original, or the issue has no options left;
- the edit is too large to be a spelling or grammar fix: more than a few tokens changed,
  or a character edit ratio above a threshold, except for kinds such as WordOrder;
- it overlaps a span the user dismissed, or a term marked intentional in the context.

Overlapping issues in one chunk are merged, keeping the higher-scored one.

## Scoring options

`SentenceScorer` protocol: `score(sentences) -> list[float]`, batched, higher is better.
Two adapters, usable alone or combined:

- **Masked-LM pseudo-log-likelihood (PLL)** (Salazar et al., 2020). Mask each token in
  turn and sum the log probability of the true token. To keep it fast, mask only the
  tokens in a window around the edit and compare original and candidate over that same
  window. Port the model loading and batching from `contcheck.lm.hf_masked_lm`
  (`distilroberta-base` by default; DeBERTa-v3 as an option). PLL prefers frequent words
  and shorter text, so compare a candidate only against its own original, normalise by
  token count, and use the delta, never the raw score.
- **Acceptability classifier.** A RoBERTa or DeBERTa model fine-tuned on CoLA gives
  P(acceptable) for the whole sentence. It is cheap (one pass per sentence) and catches
  agreement and word-order errors that PLL scores weakly.

Ranking always applies: `delta = score(candidate) - score(original)`, options sorted by
delta. Filtering is per error kind: a `ScoringPolicy` maps each `ErrorKind` to
`filter: bool` and a `margin`, and drops an issue only when filtering is on for its kind
and no option beats the original by the margin. Every kind starts with filtering off.
M6 turns it on for the kinds where it raises precision without losing much recall; PLL
is expected to help least on Punctuation, Capitalization and SentenceBoundary.

The scorer is CPU or GPU bound. The service runs it in one worker thread behind a small
batching queue, so it never blocks the event loop. PyTorch builds (cpu, cuda, rocm) are
optional extras, as in contcheck, added in M3. Without a scorer, options keep the LLM's
order.

Scoring is off until `scorer.adapters` names `pll`, `cola` or both (their scores are
summed), so the service runs without PyTorch installed; `just setup` installs the cpu
build for development. Each option carries `score` and `delta` in the API (null when
unscored), and the scorer's name, models and filters are part of the result cache key.

## Document context

`DocumentContext` is a small value built by one LLM call over the document (or, for a
long document, its first N and last N paragraphs plus a sample):

- `summary`: two or three sentences on what the text is;
- `register`: formal, neutral, informal, fiction-dialogue-heavy;
- `dialect`: en-US, en-GB, en-AU, ... (spelling conventions);
- `glossary`: names, product terms, invented words and intentional nonstandard spellings
  (dialect speech, quoted errors) that must not be flagged;
- `notes`: tense and person of narration, other conventions.

It is versioned. It is rebuilt when a set share of the document (for example 20%) has
changed since the last build, or when the client asks. Words the user adds to the
document's dictionary are passed to the builder and the prompt too.

## Service

### Documents and identity

A document is keyed by **(user, URI)**. Clients send a stable URI: the file URI in
VS Code and the CLI, the document URL in LibreOffice. Opening the same URI again finds
the same document and its dictionary. Unsaved documents get a temporary
`untitled:<uuid>` URI from the client; nothing about them is persisted. A rename is an
explicit request that moves the key.

Sessions (current text, version, suggestions, dismissals, context) live in memory behind
a `DocumentStore` protocol and are lost on restart; clients re-send the text when they
reconnect. Only the dictionary is persisted.

### Endpoints

All endpoints except `/health` require authentication. `{uri}` is URL-encoded.

- `PUT /documents/{uri}` with the full text and, for an existing session, the base
  version: opens or updates the session. The server diffs paragraphs and sentences by
  hash, moves the offsets of existing suggestions in unchanged text, drops suggestions in
  changed text, and queues the changed chunks. Updates are debounced (for example 750 ms)
  so a typing user does not trigger a call per keystroke; a newer version cancels pending
  work for the old one. Text patches instead of the full text can come later.
- `DELETE /documents/{uri}`: closes the session.
- `POST /documents/{uri}/rename` with the new URI.
- `GET /documents/{uri}/suggestions?version=N`: the current suggestions.
- `GET /documents/{uri}/events` (Server-Sent Events): suggestions added and removed, and
  check progress.
- `POST /documents/{uri}/suggestions/{sid}/accept` and `/dismiss`: accepting applies the
  chosen option and bumps the version; dismissing records the span and text so the same
  issue is not suggested again in this session.
- `GET`, `POST`, `DELETE /documents/{uri}/dictionary`: list, add and remove words.
- `POST /check`: stateless one-shot check of a text (with an optional URI to use its
  dictionary), for the CLI and simple clients. Without a context in the request, the server builds one
  first and returns it with the suggestions.
- `POST /context`: the document context of a text, for `langcheck context` and prompt
  tuning.
- `GET /health`.

### Dictionary

`DictionaryStore` protocol: `words(user, uri)`, `add(user, uri, words)`,
`remove(user, uri, words)`, `rename(user, old_uri, new_uri)`. The server config picks an
implementation:

- `memory`: for tests and throwaway servers;
- `sql` (default): SQLAlchemy 2, SQLite file by default, Postgres by changing the URL.
  Migrations with Alembic once the schema first changes.

More implementations (for example a shared team word list layered under the per-document
one) plug in behind the same protocol.

### Authentication

`Authenticator` protocol: takes the request headers, returns a `User` or an
`Unauthenticated` result. The server config picks a mode:

- `api_key` (default): per-user bearer tokens. `langcheck admin add-user NAME` prints a
  new token; the server stores only its hash (in the same SQL database as the
  dictionary). `langcheck admin revoke` removes it.
- `none`: no check. Every request belongs to one anonymous user, or to the user named in
  a configured header when a reverse proxy has already authenticated it. Chosen
  explicitly, never by default, so an exposed server is not left open.

OIDC can be added later as another mode. TLS is handled by a reverse proxy (Caddy, nginx
or Traefik) in front of uvicorn, documented with an example config. Each user gets a
limit on LLM requests in flight and per minute, so one user cannot starve the others.

### Process and state

One uvicorn process. Request handling is async; LLM calls go through an async client
with a global concurrency limit; the scorer runs in one thread. Sessions and the result
cache are in memory behind `DocumentStore` and `ResultCache` protocols, so Redis or SQL
adapters can be added when one process is no longer enough. The dictionary and API keys
are in SQL from the start.

## Settings

`src/langcheck/settings/` holds every setting as `pydantic-settings` classes made of
nested Pydantic models. Two top-level classes, because the CLI and the server need
different things:

- `ClientSettings`: `server` (URL, API key, timeout), and `llm`, `checker` and `scorer`
  for `--local` runs.
- `ServerSettings`: `http` (host, port), `llm` (base URL, model, API key, timeout,
  retries, concurrency, reasoning switch), `scorer` (adapters, model names, device,
  `ScoringPolicy` per kind), `auth` (mode, proxy user header), `database` (the SQL URL
  shared by API keys and dictionaries), `dictionary` (backend), `limits` (per-user requests in flight and per minute), `checker` (chunk
  size, debounce, context refresh share).

Each nested model is the same frozen value the domain code takes, or converts to one at
the composition root (for example `llm` to the ported `ClientConfig`). Domain code never
sees a settings class.

### Sources and precedence

Highest first; a field missing from one source falls through to the next, field by
field, so nested values merge rather than replace whole sections:

1. **Values passed to the settings object.** The CLI collects its options with `cyclopts`
   and passes only the ones the user gave as keyword arguments, so an option left out
   does not mask the layers below.
2. **Environment variables.** Prefix `LANGCHECK_`, nested with `__`: for example
   `LANGCHECK_LLM__BASE_URL`, `LANGCHECK_AUTH__MODE`, `LANGCHECK_SERVER__API_KEY`.
3. **A YAML file**, mirroring the nested structure:

   ```yaml
   llm:
     base_url: http://localhost:11434/v1
     model: qwen3:8b
   auth:
     mode: api_key
   database:
     url: sqlite:///langcheck.db
   dictionary:
     backend: sql
   ```

   The path comes from `--config` or `LANGCHECK_CONFIG`; without either, `langcheck.yaml`
   in the working directory is used if it exists. A path given explicitly must exist.
4. **Defaults** in the models.

This order is set in `settings_customise_sources` (init, environment, YAML; `.env` files
and secrets directories are not used). Settings load only at startup, in the CLI and the
API factory, never at import time (`AGENTS.md`). Secrets (`api_key` fields) are
`SecretStr`, so they never reach logs or `repr`; `langcheck config show` prints the
merged settings with secrets masked, to see which value won.

Invalid settings stop startup with Pydantic's error, which names the field (for example
`llm.retries`) and the bad value but not the source; `config show` helps find which
layer set it.

## Clients

### CLI

Built with `cyclopts`, as in contcheck. Command-line arguments are the highest layer:
they override environment variables and the YAML file (see *Settings*). With
`llm.model: qwen3:8b` in the YAML and `LANGCHECK_LLM__MODEL=llama3` in the environment,
`langcheck check --local --llm-model gemma3 notes.md` uses `gemma3`; without the option
it uses `llama3`. Every command takes `--config PATH`.

- `langcheck check FILE [--json] [--local] [--no-score] [--apply]`: one-shot check,
  printed as `FILE:LINE:COL: Kind: 'original' -> options` with the line and explanation;
  `--apply` (not built yet) walks through suggestions interactively and writes the file.
  `--no-score` keeps the LLM's option order in a `--local` run.
  Exits 0 when clean, 1 with suggestions or unchecked spans, 2 when it cannot run.
- `langcheck watch FILE [--local]`: watches the file (`watchfiles`) and prints new and
  resolved suggestions as it changes.
- `langcheck dict FILE [add|remove|list] [WORDS...]`: edit the document's dictionary.
- `langcheck context FILE`: prints the document context, for tuning prompts.
- `langcheck serve`, `langcheck lsp`, `langcheck admin add-user|revoke|users`: run the
  service, the LSP proxy, and manage API keys.
- `langcheck config show [--server]`: print the merged client (or server) settings with
  secrets masked.

`check` and `watch` call the configured server. With `--local`, or when no server is
configured, they run the same `Checker` in process against the LLM URL directly. The
CLI and `api` startup are the composition roots.

### VS Code: LSP proxy and extension

`src/langcheck/lsp/` is a language server built on `pygls`, started with
`langcheck lsp --server URL`. It holds no checking logic:

- `didOpen` / `didChange` / `didClose` → `PUT` / `DELETE /documents/{uri}`; LSP document
  versions map to the service's versions.
- Suggestion events from SSE → `publishDiagnostics`, with offsets converted to UTF-16
  positions.
- Each option → a quick fix code action, in ranked order; plus "Add to dictionary" and
  "Dismiss" code actions.

`editors/vscode/` holds the TypeScript extension: it starts `langcheck lsp` with the
server URL and API key from VS Code settings (the key in VS Code's secret storage) and
registers it for Markdown, plain text and other configured languages. Built into a
`.vsix` with `just vscode`. Neovim, Helix and Zed can use the same LSP server directly.

### LibreOffice: proofreader extension

`src/langcheck/libreoffice/` is a UNO `XProofreader` implementation, packaged as an
`.oxt` by `just libreoffice`. LibreOffice runs it in its bundled Python, which cannot
see our virtualenv, so this subpackage imports only the standard library and `uno`: it
talks to the service with `urllib` and parses JSON by hand into its own small types. A
check in `scripts/check_layout.py` enforces the import rule. Unit tests run under our
Python with a fake `uno` module.

`doProofreading` is called per paragraph and must answer from that call. The extension:

1. returns the cached suggestions for the paragraph at once (none for unseen text);
2. sends the document's full text to the service in the background, so the service has
   the whole-document context;
3. when new suggestions arrive, asks LibreOffice to re-check the affected paragraphs.

Step 3 is the main unknown. M5 opens with a spike that confirms a Python proofreader can
trigger a re-check and can read the whole document from the paragraph call; if it
cannot, the fallback is a sidebar panel listing suggestions.

## Repo layout

Code follows the `src` layout in `AGENTS.md`, organised by domain. Every Python client
is a subpackage of `src/langcheck/`.

```text
langcheck/
  src/langcheck/
    taxonomy/     # ErrorKind, Operation, Formality (copied from contcheck)
    text/         # segmentation into paragraphs and sentences with offsets; diffing
    llm/          # LLMClient protocols, chat value types, OpenAI-compatible adapters
    check/        # prompts, chunking, reply schema, resolution and validation, Checker
    context/      # DocumentContext, builder, refresh policy
    scoring/      # SentenceScorer protocol, PLL and CoLA adapters, ScoringPolicy, ranker
    documents/    # sessions, suggestion lifecycle, DocumentStore, ResultCache, scheduler
    dictionary/   # DictionaryStore protocol, memory and SQL implementations
    auth/         # Authenticator protocol, api_key and none modes, User
    settings/     # ClientSettings, ServerSettings, nested models, YAML source
    api/          # FastAPI app, request and response schemas, SSE
    lsp/          # LSP proxy to the service (pygls)
    libreoffice/  # UNO proofreader; standard library and uno only
    cli/          # `langcheck` commands; composition root with api startup
    prompts/      # versioned prompt templates (package data)
  editors/
    vscode/       # TypeScript extension that starts `langcheck lsp`
  tests/
    unit/         # mirrors src/langcheck/, in process with fakes
    integration/  # real HTTP app, SQLite, models, LLM (markers)
  design-docs/
```

Fakes for `LLMClient` (scripted replies), `SentenceScorer` (lookup table),
`DictionaryStore`, `Authenticator` and the clock go in `tests/conftest.py`. Tests that
need a downloaded model or a running LLM carry the `models` or `llm` marker and are
skipped by default.

## Milestones

Each milestone starts only when asked (see `AGENTS.md`). Each one ends with `just check`
passing.

**M0 — Scaffold** (done). uv project, `AGENTS.md`, skills, `just` recipes, pre-commit,
CI, layout check, this plan.

**M1 — Core types.** `taxonomy` (copied), `text` segmentation with offsets (start with
`pysbd` or a regex splitter; spaCy only if they fall short), the `CheckReply` and
`Suggestion` schemas, and resolution: anchor → offsets, validation, operation
derivation, applying an accepted option. Pure code, fully unit tested.

**M2 — First slice: CLI → service → LLM.** Port `llm` (async and sync clients). Prompt
v1, chunker, `DocumentContext` builder, `Checker.check(text, context, dictionary)` with
options in LLM order and the per-user result cache. `settings` with the sources above
(adds `pydantic-settings` and `pyyaml`). Service: FastAPI app factory, `POST /check`,
`/health`, `Authenticator` with `api_key` and `none`, `langcheck admin`, SQL setup shared
with M4's dictionary. CLI (`cyclopts`): `check` (service or `--local`), `context`,
`serve`, `config show`. Integration test against an OpenAI-compatible server under the
`llm` marker.

**M3 — Scoring.** `SentenceScorer` protocol, PLL adapter (ported from contcheck's masked
LM), CoLA adapter, ranker and `ScoringPolicy` with every kind's filter off. Scorer thread
and batching queue in the service. PyTorch extras and `just setup <build>`.
`--no-score` flag.

**M4 — Sessions, dictionary, VS Code.** `DocumentStore` (memory), (user, URI) keys,
paragraph and sentence diffing, offset shifting, debounce and cancellation, accept and
dismiss, SSE events. `DictionaryStore` (memory and SQL) with its endpoints and
`langcheck dict`. Per-user rate limits. LSP proxy and the VS Code extension
(`editors/vscode/`, `just vscode`, a CI job). `langcheck watch`.

**M5 — LibreOffice.** Spike first (re-check trigger, whole-document read). Then the UNO
proofreader with cache-now-fill-later, the import-rule check, `just libreoffice` and a
CI job that builds the `.oxt`.

**M6 — Evaluation and tuning.** An `evaluate` command (local mode) that runs the checker
over annotated sets (BEA-2019 dev, JFLEG, CoNLL-2014; contcheck already fetches some of
these) and scores it with ERRANT span-level F0.5, per kind. Use it to compare models and
prompts, set `ScoringPolicy` per kind, measure how many LLM edits validation rejects,
and record latency per chunk. Compare against contcheck's tagger on the same sets.

## Features

Each milestone is built as one or more features (see the `git-workflow` skill): one
feature branch each, landing on `main` through one pull request. A feature should change
at most about 5,000 lines, tests included, so its pull request stays reviewable. Tasks
inside a feature run in their own worktrees and can run in parallel where they do not
depend on each other.

| Feature branch | Milestone | Scope | Tasks |
| --- | --- | --- | --- |
| `feat/core-types` | M1 | Pure types and text handling. | `taxonomy`; `text` segmentation with offsets; `check` reply and suggestion schemas, resolution, validation, operation derivation, applying an option |
| `feat/llm-checker` | M2 | Checking in process, no service. | `llm` port (async and sync clients, fakes); prompt v1 and chunker; `DocumentContext` builder; `Checker` and the per-user result cache |
| `feat/check-service` | M2 | Settings, service and CLI around the checker. | `settings` (YAML source, precedence); `auth` (`api_key`, `none`) with the SQL setup and `langcheck admin`; FastAPI app with `POST /check` and `/health`; CLI `check`, `context`, `serve`, `config show`; `llm` integration test |
| `feat/scoring` | M3 | Ranking options. | `SentenceScorer` protocol and PLL adapter; CoLA adapter; ranker and `ScoringPolicy`; scorer thread and batching queue; PyTorch extras and `--no-score` |
| `feat/document-sessions` | M4 | Stateful documents in the service. | `DocumentStore` and (user, URI) keys; paragraph and sentence diffing with offset shifting; debounce, cancellation and the document endpoints; accept, dismiss and SSE; per-user rate limits; `langcheck watch` |
| `feat/dictionary` | M4 | Per-document dictionaries. | `DictionaryStore` memory and SQL; endpoints; `langcheck dict`; dictionary filtering in the checker |
| `feat/vscode` | M4 | Editor client. | LSP proxy (`langcheck lsp`); TypeScript extension in `editors/vscode/`, `just vscode` and its CI job |
| `feat/libreoffice` | M5 | LibreOffice client. | Spike; UNO proofreader; import-rule check; `just libreoffice` and its CI job |
| `feat/evaluation` | M6 | Measuring and tuning. | `evaluate` command with ERRANT scoring; `ScoringPolicy` tuning; model and prompt comparison |

A feature that grows past the limit while being built is split at a task boundary into a
second feature that lands after it.

## Open questions

- **Default LLM model.** Any OpenAI-compatible server works; which model to recommend is
  decided by M6 on accuracy and latency (live editing needs about 1–3 s per chunk).
- **Sentence splitter.** M1 uses a dependency-free regex splitter behind the
  `SentenceSplitter` protocol: it keeps exact offsets and handles abbreviations,
  initials and dialogue tags. Swap in `pysbd` or spaCy if M6 finds its boundaries hurt
  accuracy.
- **Does PLL scoring pay for its latency?** M6 measures it per kind. If the CoLA
  classifier alone does as well, drop PLL.
- **Whole-sentence options vs span options.** The plan scores span replacements inside
  the original sentence. If the LLM often needs several linked edits (agreement across a
  clause), allow an issue to carry a multi-span edit.
- **LibreOffice re-check.** If the M5 spike shows LibreOffice cannot be made to re-check
  a paragraph from an extension, switch to a sidebar panel.
- **Scaling past one process.** Redis or SQL adapters for `DocumentStore` and
  `ResultCache`, and a separate scorer service, when one process is not enough.
