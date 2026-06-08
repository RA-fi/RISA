<div align="center">

<img src="assets/svg/RISA.svg" alt="RISA Logo" width="120" height="120" />

# RISA — AI Health & Mental Wellness Companion

**Responsive Intelligent Support Assistant**

*Medical Advisory · Emergency First Aid · Mental Health · Multilingual · Personalized · Graph-RAG Powered*

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA--3.3--70B-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Graph--RAG-FF6B35?style=for-the-badge)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **RISA is a production-ready, open-source AI health companion** — combining a 200-topic clinical knowledge base, full medical advisory with Bangladesh-specific drug brands and dosing, hardcoded emergency first-aid protocols, persistent Graph-RAG memory, passive user profiling, multilingual voice I/O, and mental health support — all on a completely free stack.

<br/>

[**Quick Start**](#quick-start) · [**Features**](#key-features) · [**Architecture**](#architecture) · [**API Reference**](#api-reference) · [**Contributing**](#contributing)

---

</div>

## What is RISA?

RISA (Responsive Intelligent Support Assistant) is an **open-source AI health and wellness companion** that acts like a knowledgeable doctor-friend. It provides:

- **Full medical advice** — symptoms → diagnosis → treatment plan → medication with correct dose for the patient's age
- **OTC medication guidance** — specific Bangladesh brand names, dosing by age group, serious side effects, prices in BDT
- **Emergency first-aid protocols** — CPR, choking, severe bleeding, anaphylaxis, stroke, seizure, drowning — hardcoded for maximum speed, no LLM latency
- **Mental health support** — anxiety, depression, trauma, burnout, crisis detection, panic support
- **Personalized care** — passively learns the user's name, age, conditions, allergies, and medications from conversation; applies them silently to every response

> **Clinical Disclaimer:** RISA is not a substitute for professional medical or mental health care. It does not replace licensed doctors or therapists. In emergencies, it directs users to local emergency services (Bangladesh: **999**).

---

## Key Features

### 🏥 Medical Advisory
- **200-topic clinical knowledge base** covering mental health, cardiology, respiratory, GI, endocrine, neurology, dermatology, women's health, pediatrics, geriatrics, infectious diseases, preventive health, and more
- **OTC medication recommendations** with generic + Bangladesh brand names (Napa, Brufen, Zetid, Alatrol, Seclo, etc.)
- **Age-based dosing** — asks for age once, applies correct infant/child/adult/elderly dose automatically; never asks again once known
- **BDT price guidance** when requested — approximate retail prices per strip/bottle from Bangladesh pharmacies
- **Serious side effect warnings** — every recommendation includes a `⚠️ Watch out for:` line covering critical risks
- **Drug interaction and contraindication checking** against the user's known medications and conditions

### 🚨 Emergency First Aid
Hardcoded response templates (bypass LLM entirely for speed) for 13 emergency types:

| Emergency | Key Protocol |
|---|---|
| Cardiac arrest / CPR | 100–120 compressions/min, 5–6cm depth, hands-only valid |
| Choking (adult) | 5 back blows + 5 Heimlich, alternate until cleared |
| Choking (infant <1yr) | Back blows + chest thrusts only — NO Heimlich |
| Severe bleeding | Direct pressure 10 min, tourniquet 5–7cm above wound |
| Anaphylaxis | EpiPen outer thigh, 10s hold, second dose at 5–15 min |
| Stroke (FAST) | Note exact onset time — 4.5-hour thrombolysis window |
| Seizure | Time it, protect, no restraint, no finger in mouth |
| Drowning | 5 rescue breaths FIRST (hypoxia priority) |
| Poisoning | No vomiting induction, collect packaging |
| Heat stroke | Ice packs to neck/armpits/groin + fan + evaporative cooling |
| Electric shock | Cut power first, do NOT touch victim |
| Burns | 10+ min cool running water, cling film, not ice/butter |
| Recovery position | 5-step technique for unconscious breathing person |

### 👤 Passive User Profiling
RISA silently learns and stores personal facts from what users naturally share — **never by asking directly**:

| Fact | Example trigger |
|---|---|
| Name | "My name is Rahim" / "I am Fatema," |
| Age | "I am 35 years old" / "I am 45" |
| Gender | "I am a woman / father / wife…" |
| Weight | "I weigh 65kg" / "weighing 58kg" |
| Blood group | "My blood group is O+" |
| Conditions | "I have diabetes / hypertension / asthma…" |
| Allergies | "I am allergic to penicillin" |
| Medications | "I take metformin / on lisinopril" |
| Location | Any Bangladesh city/district mentioned |

Once learned, this profile is injected into every LLM call — enabling correct age-based dosing, allergy conflict checking, and personalized condition-specific advice.

### 🧠 Mental Health Support
- **38-language support** with neural voice I/O
- Real-time **crisis pattern detection** (suicidal ideation, self-harm) in English and Bengali
- **Panic attack support mode** — guided 4-2-6 breathing, 5-4-3-2-1 grounding
- **Emotion detection** — anxiety, depression, anger, grief, burnout, trauma, loneliness
- **Muslim cultural alignment** — spiritually sensitive framing (sabr, dua, dhikr) where appropriate
- **Bangladesh crisis resources** — Kaan Pete Roi: 01779-554391, NIMH: 9676001

### 🔍 Graph-RAG Memory
- **Entity knowledge graph** (35 nodes, 148+ edges) — expands every query to surface related context
- **Per-device ChromaDB** — cross-session semantic memory using local ONNX embeddings (zero cost)
- **Dynamic device graph** — learns user's recurring themes, injects top patterns into every prompt
- **Token Optimizer** — automatically trims LLM input by 30–62% per request

### ⚡ Performance
- **Streaming SSE** — real-time sentence-by-sentence delivery via `/chat/stream`
- **Fire-and-forget** — RAG writes and profile extraction never block the response
- **Translation cache** — 24-hour TTL for language detection and translation
- **Emergency bypass** — first-aid responses skip the LLM entirely for maximum speed

---

## Architecture

### System Overview

```mermaid
flowchart TD
    User["👤 User\n(Browser)"]
    Voice["🎙️ Voice Input\nfaster-whisper STT"]
    TTS["🔊 Voice Output\nEdge TTS Neural"]

    User -->|text| API
    User -->|audio| Voice
    Voice -->|transcribed text| API
    API -->|streamed response| TTS

    subgraph API["FastAPI Backend — backend.py"]
        direction TB
        Dispatch["🚨 Safety & Dispatch Layer\npre-LLM • instant response"]
        Profile["👤 Passive Profile Extractor\nname · age · conditions · meds · allergies"]
        Knowledge["📚 Knowledge Retrieval\n200-topic clinical guidance"]
        GraphRAG["🔍 Graph-RAG\nentity graph + ChromaDB"]
        Optimizer["⚙️ Token Optimizer\n30–62% input reduction"]
        LLM["🧠 LLM\nLLaMA 3.3 70B via Groq"]
        Learn["💾 Adaptive Learning\nRAG store + graph update"]

        Dispatch -->|non-emergency| Profile
        Profile --> Knowledge
        Knowledge --> GraphRAG
        GraphRAG --> Optimizer
        Optimizer --> LLM
        LLM --> Learn
    end

    subgraph Store["Persistence Layer"]
        Chroma["ChromaDB\nvector store"]
        SQLite["SQLite\nsessions + profiles"]
        KBase["knowledge/\nclinical_guidance.json\n200 topics"]
    end

    GraphRAG <--> Chroma
    Profile <--> SQLite
    Knowledge <--> KBase
    Learn --> Chroma

    Dispatch -->|emergency detected| FirstAid["⚡ Hardcoded First-Aid\n13 templates\nzero LLM latency"]
```

---

### Request Processing Pipeline

```mermaid
flowchart LR
    Msg["User Message"] --> Trans["Translate\nto English"]
    Trans --> Extract["Extract Profile\nFacts background"]
    Trans --> Detect["Detect\nEmergency / Crisis / Panic"]

    Detect -->|"🚨 Medical Emergency"| FA["Return Hardcoded\nFirst-Aid Template"]
    Detect -->|"🆘 Crisis"| Crisis["Return Crisis\nResponse + 999"]
    Detect -->|"😰 Panic"| Panic["Return Breathing\n& Grounding Guide"]
    Detect -->|"✅ Normal"| Score["Score Knowledge\nTopics 200 topics"]

    Score --> GRAG["Graph-RAG\nExpand Query"]
    GRAG --> Embed["ChromaDB\nSemantic Search"]
    Embed --> Prof["Inject User\nProfile Context"]
    Prof --> Opt["Token\nOptimizer"]
    Opt --> Groq["LLaMA 3.3 70B\nGroq Inference"]
    Groq --> Clean["Clean &\nFormat Response"]
    Clean --> Back["Translate\nBack to User Lang"]
    Back --> Reply["💬 Reply to User"]

    Extract -.->|"stored in background"| DB[("SQLite\nProfile DB")]
```

---

### Knowledge Base Coverage — 200 Topics

```mermaid
pie title Clinical Knowledge Base (200 Topics)
    "Mental Health" : 22
    "Cardiovascular" : 12
    "Respiratory" : 6
    "Gastroenterology" : 12
    "Endocrinology" : 8
    "Neurology" : 9
    "Dermatology" : 10
    "Women's Health" : 7
    "Pediatrics" : 7
    "Emergency First Aid" : 18
    "Medications" : 12
    "Infectious Diseases" : 12
    "Preventive Health" : 10
    "Geriatrics" : 6
    "Other Specialties" : 49
```

---

### User Profile Lifecycle

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant R as RISA Backend
    participant E as Profile Extractor
    participant DB as SQLite Profile DB
    participant L as LLM

    U->>R: "I am Rahim, 45 years old, I have diabetes"
    R->>E: extract_profile_facts(english_text)
    E-->>DB: merge_profile(device_key, {name, age, conditions})
    Note over DB: Stored silently in background

    U->>R: "I have a headache, what should I take?"
    R->>DB: get_profile(device_key)
    DB-->>R: {name: Rahim, age: 45, conditions: [diabetes]}
    R->>L: [System] User profile: Rahim, 45yo, diabetic\n+ clinical knowledge context
    L-->>U: "Rahim, for headache at your age with diabetes,\nparacetamol (Napa) 500mg is safest —\nibuprofen/NSAIDs raise blood sugar, avoid them.\n⚠️ Max 4g/day..."

    Note over U,L: Age-appropriate dose applied automatically
    Note over U,L: Diabetes contraindication flagged automatically
    Note over U,L: Name used naturally — never asked again
```

---

### Graph-RAG Query Expansion

```mermaid
flowchart TD
    Q["User Query:\n'I'm anxious and can't sleep'"]

    Q --> EE["Entity Extraction"]
    EE --> E1["anxiety"]
    EE --> E2["insomnia"]

    subgraph KG["Static Knowledge Graph — 148+ edges"]
        E1 -->|CAUSES| E3["panic attack"]
        E1 -->|CO_OCCURS| E4["overthinking"]
        E1 -->|HELPED_BY| E5["breathing"]
        E1 -->|HELPED_BY| E6["grounding"]
        E2 -->|LEADS_TO| E7["fatigue"]
        E2 -->|CO_OCCURS| E8["stress"]
    end

    E1 & E2 & E3 & E4 & E5 & E6 & E7 & E8 --> AQ["Augmented Query\noriginal + 5 graph terms"]
    AQ --> Chroma["ChromaDB\nSemantic Search"]
    Chroma --> TopK["Top-K Relevant\nPast Turns"]
    TopK --> Context["Enriched Context\nfor LLM"]
```

---

### Age-Based Dosing Logic

```mermaid
flowchart TD
    Ask["Medication\nRequest"]
    Ask --> AgeKnown{"Age in\nUser Profile?"}

    AgeKnown -->|"✅ Yes"| UseAge["Use stored age\nsilently"]
    AgeKnown -->|"❌ No"| AskAge["Ask once:\nকত বছর বয়স?\nHow old is the patient?"]
    AskAge --> Store["Store age\nin profile"]
    Store --> UseAge

    UseAge --> AgeBand{"Age Band"}

    AgeBand -->|"< 1 month"| Neo["Neonate\n→ Refer to doctor\nNo OTC dosing"]
    AgeBand -->|"1–11 months"| Inf["Infant\n→ mg/kg weight-based\nSyrup/drops in ml"]
    AgeBand -->|"1–5 years"| Chld1["Young Child\n→ Age-band syrup dose\nState exact ml"]
    AgeBand -->|"6–11 years"| Chld2["Older Child\n→ Half adult dose\nor age-specific"]
    AgeBand -->|"12–17 years"| Teen["Teenager\n→ Adult dose\n(No aspirin <16)"]
    AgeBand -->|"18–59 years"| Adult["Adult\n→ Standard dose"]
    AgeBand -->|"60+ years"| Elder["Elderly\n→ Reduced dose\nRenal/hepatic caution"]
```

---

### Token Budget Optimization

```mermaid
pie title LLM Input Token Budget (≤ 3,200 tokens)
    "System Prompt (fixed)" : 1100
    "Clinical Knowledge (≤280)" : 280
    "RAG Context (≤360)" : 360
    "Session History (≤800)" : 800
    "User Message (≤250)" : 250
```

```
Before optimization (heavy session):     After optimization:
─────────────────────────────────────    ────────────────────
System prompt      ~1,100 tokens    →    ~1,100  (untouched)
Knowledge context    ~655 tokens    →      ~267  (capped)
RAG context          ~260 tokens    →      ~260  (within budget)
Session history    ~3,660 tokens    →      ~732  (smart-trimmed)
─────────────────────────────────────    ────────────────────
Total:             ~5,675 tokens    →    ~2,359  (58% reduction)
```

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| **Framework** | FastAPI + Uvicorn | Async, production-ready ASGI |
| **LLM** | LLaMA 3.3 70B via Groq | Sub-second inference, free tier |
| **Vector RAG** | ChromaDB + ONNX all-MiniLM-L6-v2 | Local embeddings, zero cost |
| **Graph RAG** | NetworkX + custom engine (`core/graph_rag.py`) | 35 entity nodes, 148+ knowledge edges |
| **Token Optimizer** | `core/token_optimizer.py` (custom) | 30–62% input reduction |
| **User Profiles** | SQLite via `core/user_profile_store.py` | Passive extraction, per-device |
| **Sessions** | SQLite via `core/session_store.py` | ChatGPT-style persistent sessions |
| **TTS** | Microsoft Edge TTS (`edge-tts`) | Free, 40+ neural voices |
| **STT** | faster-whisper (local) | No API key, no rate limits |
| **Translation** | deep-translator | Free, 38+ languages |
| **LangChain** | langchain + langchain-groq | LLM orchestration |
| **Frontend** | Vanilla HTML/CSS/JS | No framework, zero build step |

---

## Project Structure

```
risa/
├── backend.py              # FastAPI app — all endpoints, LLM, TTS, STT, RAG orchestration
├── settings.py             # Environment config (HOST, PORT, ALLOW_ORIGINS)
├── index.html              # Single-page frontend
├── requirements.txt
├── Makefile
│
├── core/                   # Python support modules
│   ├── __init__.py
│   ├── graph_rag.py        # Graph-RAG engine — entity graph, query expansion, device profile
│   ├── token_optimizer.py  # Token budget optimizer — 30–62% input reduction
│   ├── session_store.py    # SQLite-backed chat session persistence
│   └── user_profile_store.py  # Passive user profiling — extraction, storage, context injection
│
├── knowledge/
│   ├── clinical_guidance.json   # 200-topic evidence-based knowledge base
│   └── learned_guidance.json    # Auto-generated from conversations (gitignored)
│
├── assets/
│   ├── css/                # Styles + Material Icons font
│   ├── js/                 # Frontend logic, jQuery, runtime config
│   ├── svg/                # Logo and icons
│   └── audio/              # UI sound effects
│
├── scripts/
│   ├── generate_token.py   # Generate secure LEARNED_GUIDANCE_ADMIN_TOKEN
│   └── setup_venv.sh       # Virtual environment setup
│
├── risa_mcp/               # Model Context Protocol server
│   ├── __init__.py
│   └── server.py
│
└── chroma_db/              # Runtime data — ChromaDB vectors + SQLite (gitignored)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- `ffmpeg` in PATH (for audio transcription)
- A free [Groq API key](https://console.groq.com)

### 1. Clone & Install

```bash
git clone https://github.com/your-org/risa.git
cd risa

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — only required key:
#   GROQ_API_KEY=gsk_...
```

> TTS, STT, embeddings, Graph-RAG, and user profiling all run locally — no extra API keys needed.

### 3. Run

```bash
python backend.py
# or
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — RISA is ready.

### Docker

```bash
docker compose up -d --build
```

---

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | **Yes** | — | Groq API key for LLaMA 3.3 70B |
| `HOST` | No | `0.0.0.0` | Server bind address |
| `PORT` | No | `8000` | Server port |
| `ALLOW_ORIGINS` | No | `*` | CORS allowed origins (comma-separated) |
| `LEARNED_GUIDANCE_ADMIN_TOKEN` | No | — | Admin token for `/learned-topics` endpoint |
| `ENABLE_PREWARM` | No | `false` | Background warmup for embeddings, graph, Whisper |

Generate an admin token:
```bash
python scripts/generate_token.py
```

---

## Clinical Knowledge Base

200 topics organized across these domains:

| Domain | Count | Example Topics |
|---|---|---|
| Mental Health | 22 | anxiety, depression, PTSD, bipolar, ADHD, OCD, phobias, personality disorders |
| Emergency First Aid | 18 | CPR, choking (adult + infant), anaphylaxis, stroke, seizure, burns, drowning |
| Medications | 12 | antidepressants, antihypertensives, diabetes meds, mental health medications |
| Infectious Diseases | 12 | dengue, typhoid, malaria, HIV, STIs, sepsis, hepatitis B/C, worms |
| Gastroenterology | 12 | GERD, IBS, peptic ulcer, celiac, hepatitis, hemorrhoids, constipation |
| Cardiovascular | 12 | hypertension, heart failure, atrial fibrillation, DVT, PAD, Raynaud's |
| Preventive Health | 10 | screening schedules, adult vaccinations, cancer screening, travel health |
| Dermatology | 10 | acne, psoriasis, eczema, fungal infections, vitiligo, melasma |
| Neurology | 9 | migraine, epilepsy, Parkinson's, MS, Bell's palsy, dementia, memory |
| Endocrinology | 8 | type 1 & 2 diabetes, gestational diabetes, hypothyroidism, PCOS, gout |
| Women's Health | 7 | pregnancy, menopause, endometriosis, PMDD, miscarriage, contraception |
| Pediatrics | 7 | childhood vaccinations, fever management, autism, infant feeding |
| Respiratory | 6 | asthma, COPD, pneumonia, sleep apnea, TB, COVID-19 |
| Geriatrics | 6 | falls prevention, dementia care, polypharmacy, palliative care |
| Other Specialties | 49 | urology, ophthalmology, ENT, rheumatology, sexual health, nutrition, occupational health… |

---

## API Reference

### `POST /chat`
Standard chat. Returns full formatted HTML response.

```json
// Request
{
  "message": "I have a fever, what should I take?",
  "device_id": "uuid-v4-string",
  "skip_translation": false,
  "source_lang": "en"
}

// Response
{
  "reply": "<p>...</p>",
  "detectedLang": "en",
  "translatedQuery": "",
  "knowledgeUsed": ["fever-infection", "otc-pain-fever-meds"],
  "performanceMs": 1240
}
```

### `POST /chat/stream`
SSE streaming — sentences arrive in real-time as the LLM generates them.

```javascript
// Each SSE event: { t: "sentence text", lang: "en" }
// Final event:    { done: true, html: "...", lang: "en", ms: 1240 }
```

### `POST /tts`
Neural text-to-speech → streams MP3.
```json
{ "text": "Take paracetamol 500mg with water.", "lang": "bn-BD" }
```

### `POST /transcribe`
Local speech-to-text via faster-whisper.
```
multipart/form-data: file=<audio>, lang="bn"
→ { "text": "transcribed text", "lang": "bn" }
```

### `GET /health`
```json
{ "status": "ok", "app": "RISA" }
```

### `GET /stats`
```json
{
  "app": "RISA",
  "rag_documents": 412,
  "graph_rag": { "ready": true, "nodes": 35, "edges": 162 },
  "model": "llama-3.3-70b-versatile",
  "learned_topics": 24
}
```

### `POST /clear`
Clear in-memory session history for a device.

---

## Supported Languages

| Language | TTS Voice | STT Engine |
|---|---|---|
| Bengali (BD) | bn-BD-NabanitaNeural | faster-whisper ✦ |
| Bengali (IN) | bn-IN-TanishaaNeural | faster-whisper ✦ |
| English | en-US-JennyNeural | Web Speech API |
| Hindi | hi-IN-SwaraNeural | faster-whisper ✦ |
| Arabic | ar-SA-ZariyahNeural | faster-whisper ✦ |
| Urdu | ur-PK-UzmaNeural | faster-whisper ✦ |
| French | fr-FR-DeniseNeural | Web Speech API |
| Spanish | es-ES-ElviraNeural | Web Speech API |
| German | de-DE-KatjaNeural | Web Speech API |
| Chinese | zh-CN-XiaoxiaoNeural | faster-whisper ✦ |
| Japanese | ja-JP-NanamiNeural | faster-whisper ✦ |
| + 27 more | (see `_EDGE_VOICES` in backend.py) | — |

> **✦** faster-whisper provides superior accuracy for these languages vs browser Web Speech API.

---

## Deployment

### Railway
Set `GROQ_API_KEY` in Railway environment variables. The `Procfile` handles the rest.

```bash
# Update assets/js/config.js with your Railway URL before deploying:
# const defaultRemote = 'https://your-app.up.railway.app';
```

### Docker
```bash
docker build -t risa .
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... risa
```

### Manual (VPS)
```bash
export GROQ_API_KEY=gsk_...
uvicorn backend:app --host 0.0.0.0 --port 8000 --workers 1
```
> Use `--workers 1` — in-memory session state is not shared across workers.

---

## Security & Privacy

```mermaid
flowchart LR
    Raw["Raw Device ID\n(client UUID)"] -->|SHA-256 hash| Key["Device Key\n(40-char hex)"]
    Key --> Chroma["ChromaDB\nvector store"]
    Key --> SQLite["SQLite\nprofiles + sessions"]

    Msg["Conversation\nMessages"] -->|"embeddings only\nnot raw text"| Chroma
    Msg -->|"ephemeral\nin-process only"| Mem["In-Memory\nSession History"]
    Mem -->|"cleared on restart"| Gone["❌ Not persisted"]

    Note["No analytics\nNo external data sharing\nAll processing on-server"]
```

- **Device IDs are SHA-256 hashed** — raw identifiers never persisted
- **User profiles stored locally** in SQLite — never sent to any external service
- **Conversation memory is ephemeral** — only semantic embeddings persist, not raw messages
- **No third-party analytics** — fully self-contained
- **Admin endpoints** require `LEARNED_GUIDANCE_ADMIN_TOKEN`
- **CORS** configurable via `ALLOW_ORIGINS` — restrict in production

---

## Roadmap

```mermaid
gantt
    title RISA Development Roadmap
    dateFormat YYYY-MM
    section Completed
    200-topic knowledge base          :done, 2025-05, 1d
    OTC medical advisory + BD brands  :done, 2025-05, 1d
    Age-based dosing system           :done, 2025-05, 1d
    13 emergency first-aid protocols  :done, 2025-05, 1d
    Passive user profiling            :done, 2025-05, 1d
    Graph-RAG entity knowledge graph  :done, 2025-04, 1d
    Token optimizer 30–62% reduction  :done, 2025-04, 1d
    Persistent ChromaDB memory        :done, 2025-04, 1d
    Multilingual neural TTS + STT     :done, 2025-04, 1d
    Organized core/ module structure  :done, 2025-05, 1d

    section In Progress
    Streaming TTS (LLM → edge-tts)    :active, 2026-06, 2025-07
    Session export / GDPR endpoint    :active, 2026-06, 2025-07

    section Planned
    Redis session state multi-worker  :2026-07, 2026-08
    RAG memory summarization          :2026-07, 2026-08
    Mobile PWA support                :2026-08, 2026-09
    Sentiment trend dashboard         :2026-09, 2026-10
```

---

## Contributing

Contributions welcome — clinical guidance improvements, new language support, UI enhancements, and bug fixes.

```bash
git checkout -b feature/your-feature
# make changes
python -m py_compile backend.py core/*.py
git commit -m "feat: description"
# open PR
```

**Priority areas:**
- Expanding `knowledge/clinical_guidance.json` with more topics
- Adding more conditions/medications to `core/user_profile_store.py` extraction patterns
- Mobile PWA support
- Unit tests for crisis detection and RAG retrieval
- Redis-backed session state for multi-worker deployments

---

## License

MIT License — free to use, modify, and deploy. Please maintain the clinical disclaimer and do not present RISA as a replacement for professional medical or mental health services.

---

<div align="center">

**Built for accessible healthcare and mental wellness worldwide.**

</div>
