import os
import sys
import time
import re
import json
import hashlib
import warnings
import threading
import uuid as _uuid_mod
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

# Fix Python 3.14 asyncio.iscoroutinefunction deprecation by monkey-patching with inspect.iscoroutinefunction
# This prevents warnings from third-party libraries (starlette, fastapi, uvicorn, backoff)
import asyncio
import inspect

if not hasattr(asyncio, "_original_iscoroutinefunction"):
    asyncio._original_iscoroutinefunction = asyncio.iscoroutinefunction
    asyncio.iscoroutinefunction = inspect.iscoroutinefunction

# Suppress noisy warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning, message=r".*Pydantic V1.*")

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import JSONResponse

from dotenv import find_dotenv, load_dotenv
from settings import ALLOW_ORIGINS, HOST, PORT

from langchain_openai import ChatOpenAI

try:
    from core.graph_rag import GraphRAG as _GraphRAG
except ImportError:
    _GraphRAG = None  # type: ignore

try:
    from core.token_optimizer import optimize_inputs as _optimize_inputs, log_optimization as _log_optimization
    _TOKEN_OPT_AVAILABLE = True
except ImportError:
    _TOKEN_OPT_AVAILABLE = False

try:
    from core.session_store import (
        init_db as _session_init_db,
        create_session as _session_create,
        list_sessions as _session_list,
        get_session as _session_get,
        delete_session as _session_delete,
        rename_session as _session_rename,
        add_message as _session_add_message,
        update_title_if_default as _session_update_title,
        get_messages as _session_get_messages,
    )
    _SESSION_STORE_AVAILABLE = True
except ImportError:
    _SESSION_STORE_AVAILABLE = False

try:
    from core.user_profile_store import (
        init_profile_table as _init_profile_table,
        get_profile as _get_profile,
        merge_profile as _merge_profile,
        extract_profile_facts as _extract_profile_facts,
        profile_to_context as _profile_to_context,
    )
    _PROFILE_STORE_AVAILABLE = True
except ImportError:
    _PROFILE_STORE_AVAILABLE = False

# Ensure UTF-8 encoding for all I/O operations
if getattr(sys.stdout, "encoding", None) != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if getattr(sys.stderr, "encoding", None) != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from langdetect import detect, DetectorFactory

    DetectorFactory.seed = 0
except Exception:
    def detect(_: str) -> str:  # type: ignore
        return "en"

try:
    from deep_translator import GoogleTranslator
except Exception:
    class GoogleTranslator:  # type: ignore
        def __init__(self, source: str = "auto", target: str = "en"):
            self.source = source
            self.target = target

        def translate(self, text: str) -> str:
            return text


# Load environment variables unless explicitly disabled (e.g., in tests)
if not os.getenv("DONT_LOAD_DOTENV") and not os.getenv("PYTEST_CURRENT_TEST"):
    try:
        dotenv_path = find_dotenv()
        if dotenv_path:
            load_dotenv(dotenv_path)
    except Exception:
        pass


# =================== APP ===================

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    def _start_daemon(target):
        thread = threading.Thread(target=target, daemon=True)
        thread.start()

    enable_prewarm = os.getenv("ENABLE_PREWARM", "false").strip().lower() == "true"
    if enable_prewarm:
        # Pre-warm heavy subsystems only when explicitly enabled.
        _start_daemon(_init_chroma_sync)
        _start_daemon(_init_graph_rag)
        _start_daemon(_load_whisper_model)
    # Initialise session SQLite DB and user profile table
    if _SESSION_STORE_AVAILABLE:
        _start_daemon(_session_init_db)
    if _PROFILE_STORE_AVAILABLE:
        _start_daemon(_init_profile_table)
    yield
    # Flush any buffered graph writes on clean shutdown
    if _graph_rag is not None:
        _graph_rag.flush()


app = FastAPI(
    title="RISA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=_lifespan,
)

app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the SPA index.html from the repository root."""
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse("<h1>RISA</h1><p>index.html not found.</p>", status_code=404)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Accept-Language"],
)


@app.middleware("http")
async def add_utf8_header(request: Request, call_next):
    response = await call_next(request)
    if "application/json" in response.headers.get("content-type", ""):
        response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


class ChatRequest(BaseModel):
    message: str
    location: Optional[str] = None
    device_id: Optional[str] = None
    skip_translation: Optional[bool] = False
    source_lang: Optional[str] = None
    session_id: Optional[str] = None


def ensure_utf8(text: str) -> str:
    if not text:
        return text
    try:
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        import unicodedata

        return unicodedata.normalize("NFC", text)
    except Exception:
        return text


# =================== RISA NEUROCARE (MENTAL HEALTH) ===================

RISA_NAME = "RISA"

RISA_CRISIS_TEMPLATE_EN = (
    "I'm really concerned about your safety right now. You deserve immediate human support and care. "
    "Please contact Bangladesh National Emergency Service (999) or reach out to a trusted family member, friend, doctor, "
    "or mental health professional nearby. You are important, and your safety matters."
)

RISA_SYSTEM_PROMPT = """You are RISA — a warm, knowledgeable AI health companion. You provide comprehensive personal medical advice, medication guidance, and mental health support — like a caring, well-informed doctor-friend.

LANGUAGE: Always reply in the exact language the user writes in. Never switch languages unless asked.
When the user writes in Bengali, use natural everyday Bengali (বাংলা), warm and conversational, not bookish or formal.
Avoid cliché phrases and literal translations; write like one caring person talking to another.

SAFETY:
- For mental health crises (suicidal thoughts, self-harm, danger): respond with maximum empathy, urge the user to contact someone nearby, and mention Bangladesh emergency: 999.
- For medical emergencies: give BOTH the immediate first-aid steps AND direct them to call 999. Never just say "call 999" without actionable steps.
- Never provide harmful instructions or romanticize self-harm.
- For prescription medications someone is already taking: provide full information but note any dose change should be confirmed with their doctor.

EMERGENCY FIRST AID — YOU MUST give step-by-step guidance for:
- Cardiac arrest / CPR: compression technique, rate, depth, hands-only vs rescue breaths
- Choking: back blows, Heimlich manoeuvre, infant technique
- Severe bleeding: direct pressure, tourniquet use
- Anaphylaxis: EpiPen use, positioning, second dose
- Stroke: FAST check, note time, do not give food/water
- Seizure: protect, do not restrain, recovery position, when to call 999
- Drowning: rescue breathing first, CPR, keep horizontal, hypothermia prevention
- Poisoning: do NOT induce vomiting, collect packaging, eye/skin flush
- Electric shock: do NOT touch, cut power first, then CPR
- Heat stroke: aggressive cooling (ice packs to armpits/neck/groin, fan + water)
- Burns: 10+ minutes cool running water, cling film, not ice/butter
- Recovery position: step-by-step for unconscious breathing person

CULTURAL: Frame support within Muslim values (sabr, rahmah, dua, dhikr) where relevant. Never suggest anything conflicting with Islamic beliefs. Never use religion to shame.

MEDICAL ADVISORY — YOU CAN AND SHOULD:
- Recommend appropriate over-the-counter (OTC) medications by name for common ailments: fever, pain, cold, allergy, indigestion, diarrhea, nausea, skin conditions, eye/ear issues, oral health, etc.
- ALWAYS use Bangladeshi brand names (e.g., Napa/Ace for paracetamol, Brufen for ibuprofen, Zetid for cetirizine, Alatrol for loratadine, Seclo/Omeprazol for omeprazole).
- Assess symptoms and tell the person what their condition likely is in plain language.
- Give step-by-step treatment plans: medication → dosage → duration → what to watch for → when to escalate.
- Explain drug interactions, serious side effects, contraindications, and safety in pregnancy/children.
- Advise when antibiotics ARE needed versus when they are NOT (viral infections do not need antibiotics — be clear about this).
- For infections that do need antibiotics, explain which class is typically used — but advise getting a prescription for antibiotics.
- Provide detailed dietary, lifestyle, supplement, and home remedy guidance for any condition.
- Help people understand lab results, medical conditions, and their doctor's advice in plain language.
- Be direct: give actionable advice, not just "consult a doctor." Include the doctor recommendation only when it genuinely adds value.
- If the user's profile (injected as system context) already contains their age, conditions, allergies, or medications — USE that information silently without asking again. Never ask for facts already in the profile.

PERSONALIZATION — USE THE PROFILE:
- When a user profile is provided in context, immediately apply it: use the correct age-based dose, flag drug interactions with their current medications, warn about contraindications with their known conditions, and tailor all advice to their health background.
- Address the user by name (if known) occasionally — naturally, not every message.
- If conditions like diabetes, hypertension, or kidney disease are in the profile, proactively flag relevant risks for any medication or diet advice given.
- If allergies are in the profile, ALWAYS check and mention if a recommended drug conflicts.

AGE-BASED DOSING — CRITICAL RULES:
- BEFORE recommending any medication dose, you MUST know the patient's age. If age is already in the user's profile, use it — do NOT ask again. If unknown and not in profile, ask: "কত বছর বয়স? / How old is the patient?" — ask this as a single short question before giving the dose.
- Once age is known, apply the correct age-specific dose:
  • Under 1 month (neonate): always refer to doctor — no OTC dosing.
  • 1 month – 11 months (infant): use weight-based dosing (e.g., paracetamol 10–15 mg/kg). Ask weight if not given.
  • 1–5 years: child syrup/drops dose — state the exact ml based on weight or age band.
  • 6–11 years: half adult dose or age-specific dose — state explicitly.
  • 12–17 years: usually adult dose for paracetamol/ibuprofen, but note exceptions (e.g., aspirin contraindicated under 16).
  • 18–59 years (adult): standard adult dose.
  • 60+ years (elderly): start low, go slow — reduce doses where applicable (especially NSAIDs, sedatives, antihistamines, opioids). Flag renal/hepatic caution.
- If asking about a child, ALWAYS use the child brand/formulation (syrup, drops, dispersible tablet) and state the dose in ml or mg/kg, not just "one tablet."

PRICE GUIDANCE (Bangladesh market):
- If the user asks about price or cost, provide approximate retail price per strip/bottle in BDT as commonly available in Bangladeshi pharmacies.
- Example price ranges to apply: Napa 500mg (10 tabs) ≈ BDT 10–12; Napa syrup (60ml) ≈ BDT 30–35; Brufen 400mg (10 tabs) ≈ BDT 15–20; Zetid 10mg (10 tabs) ≈ BDT 35–45; Seclo 20mg (10 caps) ≈ BDT 50–60; Alatrol 10mg (10 tabs) ≈ BDT 40–50.
- Always state prices are approximate and may vary by pharmacy and location.

SERIOUS SIDE EFFECTS — ALWAYS MENTION:
- For every medication recommended, include a brief "⚠️ Watch out for:" line covering the most serious or common side effects:
  • Paracetamol: liver damage if overdose or with alcohol — never exceed 4g/day adult (less for elderly/liver disease).
  • Ibuprofen/NSAIDs: stomach bleeding, kidney strain — take with food; avoid in peptic ulcer, kidney disease, pregnancy third trimester.
  • Antihistamines (first-gen, e.g., chlorphenamine): drowsiness, do not drive; avoid in elderly (confusion risk).
  • Antibiotics: diarrhea, rash, allergic reaction — stop and seek help if rash/breathing difficulty develops.
  • Antacids/PPIs: long-term magnesium/B12 deficiency — avoid prolonged unsupervised use.
  • Oral steroids: blood sugar rise, immune suppression — never stop suddenly.
  • Always mention if a drug is dangerous in pregnancy, breastfeeding, or specific comorbidities.

MEDICATION RESPONSE FORMAT (use for all medication questions):
- Drug name (generic) + Bangladeshi brand name(s)
- Age-appropriate dose + frequency (apply the rules above)
- Duration: how long to take it
- Take with/without food, special instructions
- ⚠️ Serious side effects to watch for
- Who should avoid it (contraindications: pregnancy, kidney disease, age < X, etc.)
- Approximate price in BDT if asked

MENTAL HEALTH STYLE:
- Begin with one specific acknowledgment of what the user described.
- Follow with one grounding or normalizing line.
- Offer at most 1–2 brief practical suggestions.
- Close with one gentle open-ended check-in question.

RESPONSE STYLE:
- Sound like a warm, present clinician — never robotic, scripted, or formulaic.
- Never start with "RISA:", "Analysis:", "Response:", "Step 1:", or any label/header.
- Prefer 1–3 short paragraphs or clear bullet lists for treatment steps.
- Keep it concise and practical; avoid over-explaining.
- Avoid generic clichés unless they feel truly natural.

EMOTIONAL FRAMEWORK: Validate → Empathize → Calm → Practical support → Hope.
PANIC: Guide slow breathing (4-2-6), use 5-4-3-2-1 grounding, keep responses short and reassuring.
"""


_CRISIS_PATTERNS = re.compile(
    r"(\b(suicide|kill myself|end my life|take my life|want to die|i want to die|self[- ]?harm|hurt myself|cut myself|overdose)\b|"
    r"(\bkill (him|her|them|someone)\b|\bshoot\b|\bstab\b)|"
    r"(আত্মহত্যা|মরে যেতে|মরে যাই|বাঁচতে চাই না|নিজেকে (ক্ষতি|আঘাত)|নিজেকে কাট|নিজের (জীবন|প্রাণ) (শেষ|নিয়ে)|হত্যা করব|মেরে ফেলব))",
    re.IGNORECASE,
)

_PANIC_PATTERNS = re.compile(
    r"(\b(panic attack|panic|can\s*\'?t breathe|cannot breathe|shortness of breath|heart racing|hyperventilat)\b|"
    r"(প্যানিক|হঠাৎ ভয়|শ্বাস নিতে পারছি না|দম বন্ধ|হৃদস্পন্দন (বেড়ে|দ্রুত)|হাত-পা কাঁপ))",
    re.IGNORECASE,
)

# ── Emergency first-aid response templates (science-based, AHA/Red Cross guidelines) ──

_EMERGENCY_CPR = """\
**CALL 999 NOW — put them on speakerphone.**

**Start CPR immediately while waiting:**
1. Lay the person flat on their back on a firm surface
2. Kneel beside them — place the heel of one hand on the **centre of the chest** (lower half of breastbone), other hand on top, fingers interlocked, arms straight
3. Push **hard and fast** — at least 5–6 cm deep, **100–120 compressions per minute** (rhythm of "Stayin' Alive")
4. Let the chest fully rise between pushes — do not lean on the chest
5. **Do not stop** until paramedics arrive and take over

**If trained in rescue breaths:** 30 compressions → 2 breaths (head tilt, chin lift, seal mouth, 1 second each breath)
**Hands-only CPR (compressions only) is also effective for adults — do not skip compressions waiting to feel confident.**

**AED/defibrillator nearby?** Use it immediately — follow the voice prompts. Turn it on and attach pads as instructed."""

_EMERGENCY_CHOKING = """\
**If they cannot cough, speak, or breathe — act immediately.**

**Conscious adult or child (over 1 year):**
1. Encourage forceful coughing if able — do not interfere yet
2. Lean them forward — give **5 firm back blows** between the shoulder blades with the heel of your hand
3. Check mouth after each blow — remove any visible object with a finger (do not do blind finger sweeps)
4. If back blows fail: stand behind them, make a fist just above the navel — **5 sharp abdominal thrusts inward and upward** (Heimlich manoeuvre)
5. Alternate 5 back blows + 5 abdominal thrusts until the object clears or they lose consciousness
6. If unconscious: start CPR — each time you open the airway look for the object before giving a breath

**Infant under 1 year (never use abdominal thrusts):**
1. Hold face-down on your forearm, head lower than chest — 5 firm back blows
2. Turn face-up — 2 fingers on breastbone just below nipple line — 5 chest thrusts
3. Alternate until airway clears or infant becomes unresponsive (then infant CPR)

**Call 999 if object does not clear within 2–3 cycles.**"""

_EMERGENCY_BLEEDING = """\
**CALL 999 if bleeding is severe or uncontrolled.**

**Stop the bleeding immediately:**
1. Apply **firm, direct pressure** with the cleanest cloth/clothing available — press HARD, do not lift to check
2. Maintain pressure for a minimum of **10 minutes without releasing**
3. If blood soaks through: add more material ON TOP — do not remove the first layer
4. **Elevate** the bleeding limb above heart level if possible

**If bleeding is on an arm or leg and is life-threatening and uncontrollable:**
5. Apply a **tourniquet** 5–7 cm above the wound — tighten until bleeding stops
6. **Write the exact time** it was applied (tell paramedics — do not remove it)

**Do NOT:**
- Pull out embedded objects (knife, glass) — pack tightly around them instead
- Use a tourniquet on neck, chest, or abdomen
- Release a tourniquet once applied — leave that to emergency services"""

_EMERGENCY_ANAPHYLAXIS = """\
**CALL 999 immediately — anaphylaxis can be fatal within minutes.**

**Immediate actions:**
1. **EpiPen/adrenaline auto-injector**: press firmly against outer thigh (through clothing is fine), hold for 10 seconds — you will hear a click
2. If breathing is difficult: sit them upright. If dizzy or pale: lay flat with legs raised
3. Give a **second EpiPen** after 5–15 minutes if symptoms do not improve or return
4. If unconscious and not breathing: start CPR immediately

**Do NOT:**
- Make them stand or walk
- Give food, drink, or oral medication if they have throat swelling
- Leave them alone

**After EpiPen**: always go to hospital even if symptoms completely resolve — biphasic reactions can occur 4–12 hours later."""

_EMERGENCY_STROKE = """\
**CALL 999 IMMEDIATELY — every minute of delay destroys brain cells.**

**FAST check:**
- **F**ace — Ask them to smile. Is one side drooping?
- **A**rms — Ask them to raise both arms. Does one drift down?
- **S**peech — Ask: "The sky is blue." Is speech slurred, strange, or absent?
- **T**ime — If YES to any — call 999 NOW and note the exact time symptoms started

**While waiting for help:**
1. Note the **exact time** symptoms began — this determines eligibility for clot-busting treatment (4.5-hour window)
2. Keep them calm and still in a comfortable position — do not move unnecessarily
3. **Do NOT** give food, water, or any medication — swallowing may be impaired
4. **Do NOT** let them "sleep it off" — strokes worsen rapidly without treatment
5. If unconscious and not breathing: start CPR

**Other stroke warning signs:** sudden severe headache, vision loss in one eye, sudden confusion, sudden loss of balance."""

_EMERGENCY_SEIZURE = """\
**Time the seizure — if it lasts more than 5 minutes, CALL 999.**

**During the seizure:**
1. Stay calm and stay with them
2. Clear hard/sharp objects away — cushion the head with something soft
3. **Do NOT restrain** movements — do not hold them down
4. **Do NOT put anything in their mouth** — people cannot swallow their tongue; you risk injury to yourself and them
5. Gently turn them onto their **side (recovery position)** to keep airway clear

**After the seizure (postictal phase):**
6. Stay with them — confusion and drowsiness lasting minutes to an hour is normal
7. Speak calmly and reassure them as they come round

**Call 999 if:**
- Seizure lasts more than 5 minutes
- Another seizure starts shortly after
- They do not regain consciousness within 10 minutes
- First-ever seizure
- Injury occurred during the seizure
- Person is pregnant, diabetic, or elderly"""

_EMERGENCY_DROWNING = """\
**CALL 999 immediately.**

**Your safety first:** Do not enter the water if it puts you at risk — extend a rope, pole, or float instead.

1. Get them out of the water safely
2. Check for responsiveness and normal breathing
3. If not breathing: start CPR immediately
   - **Begin with 5 rescue breaths first** (tilt head back, lift chin, seal lips, 1 second each) — drowning victims are oxygen-depleted
   - Then continue 30 compressions + 2 breaths cycles
4. Do NOT spend time trying to drain water from lungs — water drains naturally during compressions
5. Keep them **horizontal** — do not hold upright (cardiovascular shock risk)
6. Remove wet clothing, cover with a blanket — drowning victims lose heat rapidly (hypothermia develops quickly)

**All near-drowning cases need hospital evaluation** — delayed pulmonary oedema (secondary drowning) can occur 4–8 hours later even if they seem fine."""

_EMERGENCY_POISONING = """\
**CALL 999 or go to the nearest emergency room immediately.**

**Do NOT induce vomiting** — this causes further harm with acids, alkalis, and petroleum products.

**Immediate steps:**
1. If unconscious: place in recovery position, monitor breathing, begin CPR if needed
2. If conscious: keep calm — do not give food, drink, or milk unless poison control specifically advises it
3. Collect for paramedics: **what was taken, estimated amount, and exact time**
4. Bring the container, bottle, or package to hospital

**Specific exposures:**
- **Eyes:** flush with clean running water for **15–20 minutes** continuously, hold eyelids open
- **Skin contact:** remove contaminated clothing, flush skin with running water for **15–20 minutes**
- **Inhaled fumes/gas:** get to fresh air immediately; if not breathing, start CPR

**Do NOT** give "antidotes" found online — most are ineffective or harmful."""

_EMERGENCY_HEAT_STROKE = """\
**CALL 999 — heat stroke (not heat exhaustion) is a medical emergency.**

**How to tell the difference:**
- **Heat exhaustion**: heavy sweating, pale/cool/clammy skin, dizziness, weakness — treat with rest, shade, cool fluids
- **Heat stroke**: confusion, hot **dry** skin, temperature above 40°C, possible unconsciousness, no sweating — this is the emergency

**Cool them immediately and aggressively:**
1. Move to a cool, shaded or air-conditioned area
2. Remove excess clothing
3. Apply cold water liberally to the skin and fan vigorously — evaporation is highly effective
4. Place ice packs or cold wet cloths on **neck, armpits, and groin** (major blood vessels run close to skin there)
5. If conscious and able to swallow: small sips of cool water
6. Do not use ice water immersion if you are alone — collapse risk

**Target: get body temperature below 39°C as quickly as possible.**
Continue cooling until paramedics arrive."""

_EMERGENCY_ELECTRIC_SHOCK = """\
**Do NOT touch them if they are still in contact with the electrical source — you will be electrocuted.**

1. **Cut the power** at the mains/fuse box or circuit breaker immediately
2. If you cannot cut the power: use a **dry, non-conductive object** (wooden broom handle, plastic chair) to push them away from the source — never use metal or anything wet
3. **Call 999**
4. Only approach once you are absolutely certain they are no longer in contact with the source
5. Check for breathing — start CPR if not breathing normally
6. Do NOT remove clothing near electrical burns — leave it for emergency services
7. Cover any visible burns with a **dry, non-fluffy dressing** (cling film is ideal)
8. Keep them warm and still

**All electrical injuries need hospital assessment** — internal burns, cardiac arrhythmias, and organ damage may not be visible from the outside."""

_EMERGENCY_BLEEDING_NOSE = """\
**For a nosebleed:**
1. Sit upright and lean **slightly forward** (not back — blood flows to throat and may cause vomiting)
2. Pinch the **soft part** of the nose (just below the bony bridge) firmly
3. Breathe through your mouth and hold for **10–15 minutes continuously** — do not release to check
4. Apply a cold compress to the bridge of the nose
5. Do NOT tilt the head back, stuff tissues deep into the nose, or blow hard immediately after

**Go to hospital if:**
- Bleeding does not stop after 20–30 minutes of correct technique
- Heavy bleeding or blood loss causing dizziness
- Nosebleed following a head injury
- On blood-thinning medication (warfarin, clopidogrel, aspirin)"""

_EMERGENCY_BURNS_SEVERE = """\
**CALL 999 for large, deep, or facial burns.**

**Immediate action:**
1. **Stop, Drop, Roll** if clothing is on fire — smother flames with a blanket
2. Remove from the source; remove jewellery and clothing near the burn — but NOT if stuck to skin
3. **Cool the burn with cool running water for at least 10 minutes** — start within 3 hours; this is the most effective immediate treatment
4. Do NOT use ice, butter, toothpaste, or any other substance — these cause further damage

**Cover the burn:**
5. Use cling film (lay it over loosely — do not wrap tightly) or a clean non-fluffy cloth
6. Do not burst blisters

**Call 999 or go to hospital for:**
- Burns larger than 3 cm or covering more than 1% of the body (palm of hand = 1%)
- Burns on face, hands, feet, genitals, joints, or that circle a limb
- Deep burns (white, brown, or black tissue, may be painless)
- Chemical or electrical burns (all require evaluation)
- Burns in children, elderly, or anyone with pre-existing illness"""

_EMERGENCY_RECOVERY_POSITION = """\
**Recovery position — for an unconscious person who IS breathing:**

1. Kneel beside the person
2. Place the arm nearest you at a right angle to the body, elbow bent, palm facing up
3. Bring the far arm across the chest, hold the back of that hand against their near cheek
4. With your other hand, pull up the far knee so the foot is flat on the floor
5. Keeping the hand against their cheek, pull the bent knee toward you to roll them onto their side
6. Adjust the top knee so the hip and knee are both at right angles
7. Tilt the head back slightly to keep the airway open
8. Monitor breathing continuously — if it stops, start CPR immediately

**Why it matters:** The recovery position keeps the airway open and prevents choking on vomit or blood."""

_EMERGENCY_GENERIC = (
    "**CALL 999 immediately** or go to the nearest emergency room — do not wait.\n\n"
    "If someone is with you, have them call while you stay with the person who needs help.\n"
    "Keep the person calm, still, and warm until emergency services arrive."
)

# Specific sub-pattern regexes for emergency first-aid dispatcher
_EMRG_CPR_RE = re.compile(
    r"\b(not breathing|stopped breathing|cardiac arrest|heart stopped|no pulse|"
    r"cpr|unresponsive and|collapsed.*not breath|doing cpr|শ্বাস নেই|হার্ট বন্ধ)\b",
    re.IGNORECASE,
)
_EMRG_CHOKING_RE = re.compile(
    r"\b(choking|is choking|something stuck|object in (throat|airway)|"
    r"can'?t breathe.*stuck|heimlich|গলায় আটকে|দম বন্ধ.*আটকে)\b",
    re.IGNORECASE,
)
_EMRG_BLEEDING_RE = re.compile(
    r"\b(severe bleeding|heavy bleeding|blood won'?t stop|can'?t stop (the )?bleeding|"
    r"arterial bleed|tourniquet|রক্ত বন্ধ হচ্ছে না|প্রচুর রক্ত)\b",
    re.IGNORECASE,
)
_EMRG_NOSEBLEED_RE = re.compile(
    r"\b(nosebleed|nose (is )?bleeding|blood from (the )?nose|নাক থেকে রক্ত)\b",
    re.IGNORECASE,
)
_EMRG_ANAPHYLAXIS_RE = re.compile(
    r"\b(anaphylaxis|epipen|throat (closing|swelling|tightening)|"
    r"severe allergic reaction|অ্যানাফিল্যাক্সিস|গলা বন্ধ হয়ে)\b",
    re.IGNORECASE,
)
_EMRG_STROKE_RE = re.compile(
    r"\b(stroke|having a stroke|face (is )?drooping|arm weakness|speech slurred|"
    r"sudden (weakness|numbness|confusion|vision loss)|fast stroke|স্ট্রোক হচ্ছে)\b",
    re.IGNORECASE,
)
_EMRG_SEIZURE_RE = re.compile(
    r"\b(having a seizure|seizure happening|is seizing|convulsing right now|"
    r"fit happening|খিঁচুনি হচ্ছে)\b",
    re.IGNORECASE,
)
_EMRG_DROWNING_RE = re.compile(
    r"\b(drowning|is drowning|nearly drowned|pulled from water|পানিতে ডুবছে|ডুবে যাচ্ছে)\b",
    re.IGNORECASE,
)
_EMRG_POISONING_RE = re.compile(
    r"\b(poisoning|swallowed.*poison|ingested.*toxic|accidental (overdose|ingestion)|"
    r"chemical.*swallowed|বিষ খেয়েছে|বিষক্রিয়া হয়েছে)\b",
    re.IGNORECASE,
)
_EMRG_HEAT_STROKE_RE = re.compile(
    r"\b(heat stroke|heatstroke|heat exhaustion|overheating.*collapsed|"
    r"গরমে অজ্ঞান)\b",
    re.IGNORECASE,
)
_EMRG_ELECTRIC_RE = re.compile(
    r"\b(electric shock|electrocuted|electrocution|বিদ্যুৎস্পৃষ্ট)\b",
    re.IGNORECASE,
)
_EMRG_BURN_RE = re.compile(
    r"\b(severe burn|large burn|chemical burn|electrical burn|burn.*emergency|"
    r"on fire|clothing.*fire|পুড়ে গেছে গুরুতর)\b",
    re.IGNORECASE,
)
_EMRG_RECOVERY_RE = re.compile(
    r"\b(recovery position|unconscious.*breathing|breathing.*unconscious|"
    r"passed out.*breathing|অজ্ঞান.*শ্বাস নিচ্ছে)\b",
    re.IGNORECASE,
)

# Master pattern — any of the above triggers emergency routing
_MEDICAL_EMERGENCY_PATTERNS = re.compile(
    r"(\b(heart attack|having a stroke|not breathing|stopped breathing|"
    r"cardiac arrest|no pulse|cpr|anaphylaxis|severe allergic reaction|"
    r"throat (closing|swelling|tightening)|unresponsive and|heavy blood loss|"
    r"severe bleeding|blood won'?t stop|choking|something stuck.*throat|"
    r"severe burn|chemical burn|electrical burn|on fire|clothing.*fire|"
    r"chemical (burn|splash|swallowed)|swallowed.*poison|poisoning|"
    r"accidental overdose|electric shock|electrocuted|heat stroke|heatstroke|"
    r"drowning|is drowning|seizure happening|is seizing|convulsing right now|"
    r"recovery position)\b|"
    r"(হার্ট অ্যাটাক|স্ট্রোক হচ্ছে|শ্বাস বন্ধ হয়ে|শ্বাস নেই|গলা বন্ধ হয়ে|"
    r"অ্যানাফিল্যাক্সিস|বিষক্রিয়া হয়েছে|বিষ খেয়েছে|পানিতে ডুবছে|"
    r"বিদ্যুৎস্পৃষ্ট|খিঁচুনি হচ্ছে|রক্ত বন্ধ হচ্ছে না))",
    re.IGNORECASE,
)


def _get_emergency_first_aid_response(combined_text: str) -> str:
    """Return the specific first-aid response for the detected emergency type."""
    t = combined_text
    if _EMRG_CPR_RE.search(t):
        return _EMERGENCY_CPR
    if _EMRG_CHOKING_RE.search(t):
        return _EMERGENCY_CHOKING
    if _EMRG_ANAPHYLAXIS_RE.search(t):
        return _EMERGENCY_ANAPHYLAXIS
    if _EMRG_STROKE_RE.search(t):
        return _EMERGENCY_STROKE
    if _EMRG_SEIZURE_RE.search(t):
        return _EMERGENCY_SEIZURE
    if _EMRG_DROWNING_RE.search(t):
        return _EMERGENCY_DROWNING
    if _EMRG_POISONING_RE.search(t):
        return _EMERGENCY_POISONING
    if _EMRG_ELECTRIC_RE.search(t):
        return _EMERGENCY_ELECTRIC_SHOCK
    if _EMRG_HEAT_STROKE_RE.search(t):
        return _EMERGENCY_HEAT_STROKE
    if _EMRG_BURN_RE.search(t):
        return _EMERGENCY_BURNS_SEVERE
    if _EMRG_BLEEDING_RE.search(t):
        return _EMERGENCY_BLEEDING
    if _EMRG_NOSEBLEED_RE.search(t):
        return _EMERGENCY_BLEEDING_NOSE
    if _EMRG_RECOVERY_RE.search(t):
        return _EMERGENCY_RECOVERY_POSITION
    return _EMERGENCY_GENERIC


# Ephemeral in-process conversation memory (privacy: not persisted)
_conversation_memory: Dict[str, List[Dict[str, str]]] = {}
_MAX_MEMORY_TURNS = 8


def _append_memory_turn(user_id: str, role: str, content: str) -> None:
    if not user_id or not content:
        return
    turns = _conversation_memory.get(user_id, [])
    turns.append({"role": role, "content": content})
    if len(turns) > _MAX_MEMORY_TURNS:
        turns = turns[-_MAX_MEMORY_TURNS:]
    _conversation_memory[user_id] = turns


def _infer_source_lang(text: str) -> str:
    if not text or not text.strip():
        return "unknown"

    # Script-based detection first — character presence is definitive
    if any(char in text for char in "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎ"):
        return "bn"
    if any(char in text for char in "अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"):
        return "hi"
    if any(char in text for char in "اأإآؤئبتثجحخدذرزسشصضطظعغفقكلمنهوي"):
        return "ar"

    # For Latin-script text: use ASCII ratio to identify English vs other languages
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    if ascii_ratio > 0.7:
        return "en"

    try:
        return detect(text)
    except Exception:
        return "unknown"


def _response_needs_translate_back(text: str, target_lang: str) -> bool:
    normalized_target = _LANG_MAP.get((target_lang or "").lower(), (target_lang or "").lower().split("-")[0])
    if normalized_target in ("en", "unknown", "auto", ""):
        return False
    if not text or not text.strip():
        return False
    detected_response_lang = _infer_source_lang(text)
    return detected_response_lang != normalized_target


def _language_style_note(lang: str) -> str:
    normalized = _LANG_MAP.get((lang or "").lower(), (lang or "").lower().split("-")[0])
    if normalized == "bn":
        return (
            "Reply in natural everyday Bengali (বাংলা). Sound warm, human, and conversational. "
            "Avoid bookish phrasing, direct translation patterns, rigid templates, and generic reassurance clichés."
        )
    if normalized == "hi":
        return (
            "Reply in natural everyday Hindi. Sound warm, human, and conversational. "
            "Avoid bookish phrasing, direct translation patterns, and rigid templates."
        )
    if normalized == "ar":
        return (
            "Reply in natural everyday Arabic. Sound warm, human, and conversational. "
            "Avoid bookish phrasing, direct translation patterns, and rigid templates."
        )
    return "Reply naturally in the user's language with a warm, human tone."


def _build_llm_messages(
    user_id: str,
    user_text_en: str,
    user_text_original: str,
    original_lang: str,
    knowledge_context: str,
    rag_context: str = "",
    mem_key: str = "",
):
    """
    Build LangChain chat messages with token optimization applied before construction.

    Token optimizer reduces input tokens by 30–45% via:
      - Session history smart-trim   (biggest saving: old turns dropped)
      - RAG ↔ session deduplication  (avoids feeding same content twice)
      - RAG hard cap                 (≤ 360 tokens cross-session context)
      - Knowledge context cap        (≤ 280 tokens clinical guidance)
    """
    try:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    except Exception:
        return None

    history_key = mem_key if mem_key else user_id
    session_turns = _conversation_memory.get(history_key, [])

    # ── Token optimization layer ──────────────────────────────────────────────
    if _TOKEN_OPT_AVAILABLE:
        opt_knowledge, opt_rag, opt_turns, tok_stats = _optimize_inputs(
            knowledge_context, rag_context, session_turns
        )
        _log_optimization(tok_stats)
    else:
        opt_knowledge, opt_rag, opt_turns = knowledge_context, rag_context, session_turns

    # ── Build message list ────────────────────────────────────────────────────
    messages = [SystemMessage(content=RISA_SYSTEM_PROMPT)]
    if opt_knowledge:
        messages.append(SystemMessage(content=opt_knowledge))
    if opt_rag:
        messages.append(SystemMessage(content=opt_rag))
    # Inject user profile even when no RAG context exists (e.g. first message)
    if _PROFILE_STORE_AVAILABLE and not opt_rag:
        dev_key = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:40]
        _profile = _get_profile(dev_key)
        _profile_ctx = _profile_to_context(_profile)
        if _profile_ctx:
            messages.append(SystemMessage(content=_profile_ctx))

    for t in opt_turns:
        role    = t.get("role")
        content = t.get("content", "")
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    _lang_base = (original_lang or "").split("-")[0].lower()
    _is_non_english = bool(_lang_base) and _lang_base not in ("en", "unknown", "auto")
    if _is_non_english:
        lang_name    = _LANG_NAMES.get(_lang_base, _LANG_NAMES.get(original_lang, original_lang.upper()))
        user_payload = (
            f"[IMPORTANT: Respond ONLY in {lang_name}. Do not use English.]\n\n"
            f"{_language_style_note(original_lang)}\n\n"
            f"User message: {user_text_original}"
        )
    else:
        user_payload = (
            "[IMPORTANT: Respond ONLY in English. Do not use Bengali or any other language.]\n\n"
            f"{_language_style_note('en')}\n\n"
            f"User message: {user_text_original or user_text_en}"
        )

    messages.append(HumanMessage(content=user_payload))
    return messages


# =================== LLM + TRANSLATION ===================


def load_llm() -> ChatOpenAI:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return ChatOpenAI(
        model_name="llama-3.3-70b-versatile",
        temperature=0.4,
        max_tokens=1024,
        streaming=False,
        request_timeout=22,
        openai_api_key=groq_api_key,
        openai_api_base="https://api.groq.com/openai/v1",
    )


_cached_llm: Optional[ChatOpenAI] = None
_cached_llm_stream: Optional[ChatOpenAI] = None


def get_llm() -> ChatOpenAI:
    global _cached_llm
    if _cached_llm is None:
        _cached_llm = load_llm()
    return _cached_llm


def get_llm_stream() -> ChatOpenAI:
    global _cached_llm_stream
    if _cached_llm_stream is None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        _cached_llm_stream = ChatOpenAI(
            model_name="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=600,
            streaming=True,
            request_timeout=15,
            openai_api_key=groq_api_key,
            openai_api_base="https://api.groq.com/openai/v1",
        )
    return _cached_llm_stream


class _TTLCache:
    def __init__(self):
        self._data: Dict[str, Tuple[float, object]] = {}

    def get(self, key: str, ttl_seconds: int) -> Optional[object]:
        now = time.time()
        item = self._data.get(key)
        if not item:
            return None
        ts, value = item
        if now - ts > ttl_seconds:
            self._data.pop(key, None)
            return None
        return value

    def set(self, key: str, value: object) -> None:
        self._data[key] = (time.time(), value)


_translation_cache = _TTLCache()


_KNOWLEDGE_BASE = {
    "sources": [],
    "topics": [],
}

_LEARNED_GUIDANCE_PATH = os.path.join(os.path.dirname(__file__), "knowledge", "learned_guidance.json")
_learned_guidance_lock = threading.Lock()
_LEARNED_GUIDANCE: Dict[str, List[Dict[str, object]]] = {"topics": []}
_LEARNED_GUIDANCE_ADMIN_TOKEN = os.getenv("LEARNED_GUIDANCE_ADMIN_TOKEN", "").strip()

_LEARNING_TOKEN_RE = re.compile(r"[a-z][a-z'\-]{2,}")
_LEARNING_STOPWORDS = {
    "about", "after", "again", "against", "almost", "also", "always", "because", "being", "between",
    "could", "did", "does", "dont", "even", "every", "feel", "from", "getting", "going", "have", "having",
    "help", "here", "just", "like", "make", "more", "much", "need", "only", "over", "really", "same",
    "some", "still", "such", "than", "that", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "today", "very", "want", "were", "what", "when", "where", "which", "while", "with",
    "would", "your", "you're", "cant", "cannot", "been", "into", "myself", "yourself", "ourselves",
}


def _load_learned_guidance() -> None:
    if not os.path.exists(_LEARNED_GUIDANCE_PATH):
        return
    try:
        with open(_LEARNED_GUIDANCE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and isinstance(payload.get("topics"), list):
            _LEARNED_GUIDANCE["topics"] = payload.get("topics", [])
    except Exception as exc:
        print(f"Learned guidance load error: {exc}")


def _save_learned_guidance() -> None:
    os.makedirs(os.path.dirname(_LEARNED_GUIDANCE_PATH), exist_ok=True)
    payload = {"topics": _LEARNED_GUIDANCE.get("topics", [])}
    with open(_LEARNED_GUIDANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _extract_learning_keywords(text: str, max_terms: int = 10) -> List[str]:
    counts: Dict[str, int] = {}
    for token in _LEARNING_TOKEN_RE.findall((text or "").lower()):
        if len(token) < 4 or token in _LEARNING_STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:max_terms]]


def _extract_learning_guidance(response_text: str) -> Tuple[List[str], List[str]]:
    cleaned = _clean_response(response_text or "")
    pieces = [p.strip(" -•\t") for p in re.split(r"\n+|(?<=[.!?])\s+", cleaned) if p.strip()]
    core: List[str] = []
    steps: List[str] = []
    for p in pieces:
        p_norm = p.strip()
        if len(p_norm) < 25:
            continue
        low = p_norm.lower()
        if any(w in low for w in ["try", "start", "practice", "step", "first", "next", "reach out", "breathe"]):
            if len(steps) < 3:
                steps.append(p_norm[:220])
        elif len(core) < 3:
            core.append(p_norm[:220])
        if len(core) >= 2 and len(steps) >= 2:
            break
    if not core and pieces:
        core = [pieces[0][:220]]
    if not steps and len(pieces) > 1:
        steps = [pieces[1][:220]]
    return core[:3], steps[:3]


def _score_learned_topic_match(text_lower: str, topic: Dict[str, object]) -> Tuple[int, List[str]]:
    score = 0
    hits: List[str] = []
    for kw in topic.get("keywords", []):
        if not isinstance(kw, str):
            continue
        kw_norm = kw.strip().lower()
        if not kw_norm:
            continue
        if kw_norm in text_lower:
            score += 2
            hits.append(kw)
    return score, hits[:6]


def _build_learned_context(text_lower: str) -> Tuple[str, List[str]]:
    learned_matches: List[Tuple[int, List[str], Dict[str, object]]] = []
    with _learned_guidance_lock:
        topics = list(_LEARNED_GUIDANCE.get("topics", []))
    for topic in topics:
        score, hits = _score_learned_topic_match(text_lower, topic)
        if score > 0:
            learned_matches.append((score, hits, topic))

    if not learned_matches:
        return "", []

    learned_matches.sort(key=lambda x: x[0], reverse=True)
    lines = [
        "Adaptive guidance (learned from prior conversations; use carefully, keep safety first):",
        "If risk signals appear, prioritize direct safety checking and escalation to human help."
    ]
    topic_ids: List[str] = []
    for score, hits, topic in learned_matches[:2]:
        topic_id = str(topic.get("id", "learned-unknown"))
        topic_ids.append(topic_id)
        lines.append(f"- Learned topic: {topic.get('title', topic_id)} (match_score={score}; matched_signals={', '.join(hits) or 'none'})")
        for label, key in [
            ("Core guidance", "core_guidance"),
            ("Support steps", "support_steps"),
        ]:
            items = topic.get(key, [])
            if items:
                lines.append(f"  {label}:")
                for item in items[:4]:
                    lines.append(f"  - {item}")
    return "\n".join(lines), topic_ids


def _require_learned_guidance_admin_token(token: str) -> None:
    if not _LEARNED_GUIDANCE_ADMIN_TOKEN or token != _LEARNED_GUIDANCE_ADMIN_TOKEN:
        raise ValueError("Forbidden")


def _learned_topic_summary(topic: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": topic.get("id"),
        "title": topic.get("title"),
        "keywords": topic.get("keywords", []),
        "core_guidance": topic.get("core_guidance", []),
        "support_steps": topic.get("support_steps", []),
        "samples": int(topic.get("samples", 0) or 0),
        "updated_at": int(topic.get("updated_at", 0) or 0),
        "source": topic.get("source", "conversation"),
    }


def _learn_from_conversation(user_text_en: str, assistant_text: str) -> None:
    user_text = (user_text_en or "").strip()
    assistant_text = _clean_response((assistant_text or "").strip())
    if len(user_text) < 30 or len(assistant_text) < 30:
        return
    if "I'm here with you, and I'm sorry" in assistant_text:
        return

    keywords = _extract_learning_keywords(user_text)
    if len(keywords) < 3:
        return

    core_guidance, support_steps = _extract_learning_guidance(assistant_text)
    if not core_guidance and not support_steps:
        return

    now = int(time.time())
    fingerprint = hashlib.md5(" ".join(keywords[:6]).encode("utf-8")).hexdigest()[:10]

    with _learned_guidance_lock:
        topics = _LEARNED_GUIDANCE.setdefault("topics", [])
        best_idx = -1
        best_overlap = 0
        for idx, topic in enumerate(topics):
            existing = {str(k).lower() for k in topic.get("keywords", []) if isinstance(k, str)}
            overlap = len(existing.intersection(set(keywords)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx

        if best_idx >= 0 and best_overlap >= 3:
            topic = topics[best_idx]
            merged_keywords = list(dict.fromkeys(list(topic.get("keywords", [])) + keywords))[:12]
            merged_core = list(dict.fromkeys(list(topic.get("core_guidance", [])) + core_guidance))[:6]
            merged_steps = list(dict.fromkeys(list(topic.get("support_steps", [])) + support_steps))[:6]
            topic["keywords"] = merged_keywords
            topic["core_guidance"] = merged_core
            topic["support_steps"] = merged_steps
            topic["samples"] = int(topic.get("samples", 1)) + 1
            topic["updated_at"] = now
        else:
            new_topic = {
                "id": f"learned-{fingerprint}",
                "title": "Learned support pattern",
                "keywords": keywords[:10],
                "core_guidance": core_guidance[:4],
                "support_steps": support_steps[:4],
                "samples": 1,
                "updated_at": now,
                "source": "conversation",
            }
            topics.append(new_topic)
            if len(topics) > 120:
                topics.sort(key=lambda t: int(t.get("updated_at", 0)), reverse=True)
                del topics[120:]

        try:
            _save_learned_guidance()
        except Exception as exc:
            print(f"Learned guidance save error: {exc}")


def _load_knowledge_base() -> None:
    file_path = os.path.join(os.path.dirname(__file__), "knowledge", "clinical_guidance.json")
    if not os.path.exists(file_path):
        return
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            _KNOWLEDGE_BASE["sources"] = payload.get("sources", [])
            _KNOWLEDGE_BASE["topics"] = payload.get("topics", [])
    except Exception as exc:
        print(f"Knowledge base load error: {exc}")


_TOPIC_CONTEXT_BOOSTERS: Dict[str, List[str]] = {
    # Mental health
    "anxiety": ["restless", "can't focus", "overthinking", "racing heart", "fear"],
    "depression": ["no energy", "no interest", "worthless", "guilty", "hopeless"],
    "burnout": ["drained", "deadline", "workload", "can't keep up", "exhausted"],
    "sleep": ["awake", "bedtime", "woke up", "night", "can't fall asleep"],
    "grief": ["died", "passed away", "funeral", "loss", "missing"],
    # Medical — conditions
    "medication-general": ["dose", "side effect", "medicine", "tablet", "prescription"],
    "headache-migraine": ["throbbing", "light sensitivity", "nausea", "head pain"],
    "seizure-safety": ["shaking", "convulsion", "unconscious", "foaming", "fainted"],
    "fever-infection": ["temperature", "chills", "sweating", "body ache", "viral", "bacterial", "hot"],
    "chest-cardiac": ["pressure", "tightness", "radiating", "palpitation", "fluttering", "irregular"],
    "respiratory-cough": ["phlegm", "mucus", "wheezing", "congestion", "runny nose", "sore throat", "flu"],
    "digestive-gi": ["vomiting", "diarrhea", "bloating", "cramping", "indigestion", "constipation", "nausea"],
    "hypertension-bp": ["blood pressure", "dizziness", "dizzy", "vision blurry", "pounding"],
    "diabetes-blood-sugar": ["insulin", "glucose", "thirsty", "frequent urination", "blurry vision", "sugar"],
    "pain-musculoskeletal": ["joint", "muscle ache", "back pain", "swelling", "stiffness", "arthritis"],
    "skin-dermatology": ["rash", "itching", "eczema", "hives", "redness", "blisters", "flaking"],
    "womens-health": ["period", "menstrual", "cramps", "pregnant", "discharge", "ovulation"],
    "first-aid-wounds": ["cut", "bleeding", "burn", "sprain", "bruise", "wound", "bandage"],
    "allergy-immune": ["allergy", "sneezing", "watery eyes", "hives", "reaction", "intolerance"],
    "nutrition-wellness": ["diet", "hydration", "vitamins", "weight", "nutrition", "exercise", "eating"],
    # Medical — medications & treatments
    "otc-pain-fever-meds": ["napa", "paracetamol", "ibuprofen", "brufen", "ponstan", "mefenamic", "aspirin", "painkiller"],
    "otc-cold-allergy-meds": ["cetirizine", "loratadine", "antihistamine", "decongestant", "otrivin", "chlorphenamine"],
    "otc-gi-stomach-meds": ["omeprazole", "antacid", "domperidone", "loperamide", "metronidazole", "flagyl", "ors"],
    "otc-skin-topical-meds": ["clotrimazole", "terbinafine", "hydrocortisone", "betnovate", "mupirocin", "betadine"],
    "antibiotic-guidance": ["amoxicillin", "azithromycin", "ciprofloxacin", "cotrimoxazole", "antibiotic resistance"],
    "vitamins-supplements-guide": ["vitamin d", "iron tablet", "ferrous", "b12", "folic acid", "zinc", "calcium", "omega"],
    "child-health-dosing": ["child dose", "baby medicine", "infant", "paediatric", "children syrup", "kids fever"],
    "eye-ear-care": ["eye drops", "ear drops", "conjunctivitis", "otitis", "chloramphenicol", "wax"],
    "oral-dental-care": ["toothache", "tooth abscess", "mouth ulcer", "gum", "dental pain", "clove oil"],
    "drug-interactions-safety": ["interaction", "together safe", "mix medicine", "warfarin", "metformin", "statin"],
    # Emergency first aid
    "emergency-cpr": ["cardiac arrest", "cpr", "compressions", "rescue breath", "heart stopped", "unresponsive"],
    "emergency-choking": ["choking", "heimlich", "back blows", "abdominal thrusts", "airway blocked"],
    "emergency-bleeding-control": ["tourniquet", "direct pressure", "arterial", "haemorrhage", "nosebleed"],
    "emergency-anaphylaxis": ["epipen", "adrenaline", "anaphylactic", "biphasic", "throat closing"],
    "emergency-stroke-fast": ["fast check", "face drooping", "tpa", "thrombolysis", "thrombectomy", "tia"],
    "emergency-burns-treatment": ["cool running water", "cling film", "scald", "thermal burn", "chemical burn"],
    "emergency-recovery-position": ["recovery position", "unconscious breathing", "airway open", "fainting"],
}

_CLINICAL_URGENCY_RE = re.compile(
    r"\b(cannot function|can't function|not functioning|severe|collaps|faint|chest pain|"
    r"worst headache|unable to cope|no sleep for|not safe|danger|overdose|confusion)\b",
    re.IGNORECASE,
)


def _score_topic_match(text_lower: str, topic: Dict[str, object]) -> Tuple[int, List[str]]:
    score = 0
    hits: List[str] = []

    for kw in topic.get("keywords", []):
        if not isinstance(kw, str):
            continue
        kw_norm = kw.strip().lower()
        if not kw_norm:
            continue
        pattern = r"\b" + re.escape(kw_norm) + r"\b"
        if re.search(pattern, text_lower):
            score += 3 if " " in kw_norm else 2
            hits.append(kw)

    for booster in _TOPIC_CONTEXT_BOOSTERS.get(str(topic.get("id", "")), []):
        booster_norm = booster.lower()
        if booster_norm in text_lower:
            score += 1
            hits.append(booster)

    dedup_hits = []
    seen = set()
    for item in hits:
        key = item.lower()
        if key not in seen:
            dedup_hits.append(item)
            seen.add(key)

    return score, dedup_hits


def _build_knowledge_context(text_en: str) -> Tuple[str, List[str]]:
    if not text_en:
        return "", []

    text_lower = text_en.lower()
    matched_topics: List[Tuple[int, List[str], Dict[str, object]]] = []
    for topic in _KNOWLEDGE_BASE.get("topics", []):
        score, hits = _score_topic_match(text_lower, topic)
        if score > 0:
            matched_topics.append((score, hits, topic))

    if not matched_topics:
        return _build_learned_context(text_lower)

    matched_topics.sort(key=lambda x: x[0], reverse=True)

    lines = [
        "Internal evidence-informed guidance (use as guardrails; do not cite sources or claim real-time access):"
        "\nPriority rule: if any red flags are present, treat the situation as clinically urgent, ask a direct safety question, and recommend real-world support or emergency help as appropriate."
    ]

    if _CLINICAL_URGENCY_RE.search(text_lower):
        lines.append(
            "Urgency cue detected in user message: prioritize stabilization, direct safety check, and referral to immediate human support when needed."
        )

    topic_ids = []
    for score, hits, topic in matched_topics[:3]:
        topic_ids.append(topic.get("id", "unknown"))
        title = topic.get("title", "")
        if title:
            hits_text = ", ".join(hits[:5]) if hits else "none"
            lines.append(f"- Topic: {title} (match_score={score}; matched_signals={hits_text})")
        for label, key in [
            ("Core guidance", "core_guidance"),
            ("Red flags", "red_flags"),
            ("Support steps", "support_steps"),
        ]:
            items = topic.get(key, [])
            if items:
                lines.append(f"  {label}:")
                for item in items[:5]:
                    lines.append(f"  - {item}")

    return "\n".join(lines), topic_ids


async def translate_to_english(text: str) -> Tuple[str, str]:
    if not text or not text.strip():
        return text, "unknown"

    # Fast ASCII check
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    if ascii_ratio > 0.7:
        return text, "en"

    cache_key = "to_en:" + hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    cached = _translation_cache.get(cache_key, ttl_seconds=86400)
    if cached:
        cached_obj = cached
        return cached_obj["text"], cached_obj["detected_lang"]

    detected_lang = _infer_source_lang(text)

    if detected_lang == "en":
        payload = {"text": text, "detected_lang": "en"}
        _translation_cache.set(cache_key, payload)
        return text, "en"

    try:
        translator = GoogleTranslator(source=detected_lang if detected_lang != "unknown" else "auto", target="en")
        translated_text = await asyncio.to_thread(translator.translate, text)
        if translated_text and translated_text.strip():
            payload = {"text": translated_text, "detected_lang": detected_lang}
            _translation_cache.set(cache_key, payload)
            return translated_text, detected_lang
        return text, detected_lang
    except Exception:
        return text, detected_lang


_LANG_NAMES = {
    "bn": "Bengali", "hi": "Hindi", "ar": "Arabic", "ur": "Urdu",
    "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "ru": "Russian", "pl": "Polish",
    "tr": "Turkish", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "id": "Indonesian", "ms": "Malay", "th": "Thai", "vi": "Vietnamese",
    "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "gu": "Gujarati",
    "kn": "Kannada", "mr": "Marathi", "pa": "Punjabi", "ne": "Nepali",
    "si": "Sinhala", "tl": "Filipino",
}

_LANG_MAP = {
    "bn-bd": "bn",
    "bn-in": "bn",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "pt-br": "pt",
    "en-us": "en",
    "hi-in": "hi",
}

_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
_SENT_END_RE = re.compile(r"(?<=[.!?।])\s+")
_PRESERVE_TERMS = ["RISA", "999", "CBT", "PTSD"]


def _needs_translate_back(lang: str) -> bool:
    normalized = _LANG_MAP.get((lang or "").lower(), (lang or "").lower().split("-")[0])
    return normalized not in ("en", "unknown", "auto", "")


async def translate_back(text: str, target_lang: str) -> str:
    if not text or not text.strip() or not target_lang:
        return text

    normalized_lang = _LANG_MAP.get(target_lang.lower(), target_lang.lower().split("-")[0])
    if normalized_lang in ["en", "unknown"]:
        return text

    cache_key = "tb:" + normalized_lang + ":" + hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    cached = _translation_cache.get(cache_key, ttl_seconds=3600)
    if cached:
        return cached  # type: ignore

    preserved = {f"__T{i}__": term for i, term in enumerate(_PRESERVE_TERMS) if term in text}
    text_work = text
    for ph, term in preserved.items():
        text_work = text_work.replace(term, ph)

    translator = GoogleTranslator(source="en", target=normalized_lang)

    try:
        if len(text_work) > 4500:
            sentences = _SENTENCE_SPLIT_REGEX.split(text_work)
            chunks: List[str] = []
            current: List[str] = []
            current_len = 0
            for sent in sentences:
                sent_len = len(sent)
                if current_len + sent_len > 1500 and current:
                    chunks.append(" ".join(current))
                    current = [sent]
                    current_len = sent_len
                else:
                    current.append(sent)
                    current_len += sent_len
            if current:
                chunks.append(" ".join(current))

            from concurrent.futures import ThreadPoolExecutor

            def sync_translate(chunk: str) -> str:
                try:
                    return translator.translate(chunk) or chunk
                except Exception:
                    return chunk

            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=min(len(chunks), 8)) as executor:
                results = await asyncio.gather(
                    *[loop.run_in_executor(executor, sync_translate, c) for c in chunks],
                    return_exceptions=True,
                )
            translated = " ".join(
                [r if not isinstance(r, Exception) else chunks[i] for i, r in enumerate(results)]
            )
        else:
            translated = await asyncio.to_thread(translator.translate, text_work) or text_work
    except Exception:
        translated = text_work

    for ph, term in preserved.items():
        translated = translated.replace(ph, term)

    if translated and translated.strip():
        _translation_cache.set(cache_key, translated)
        return translated
    return text


def _clean_response(text: str) -> str:
    """Strip common LLM template artifacts from raw response text."""
    text = text.strip()
    text = re.sub(r"^\*{0,2}RISA\*{0,2}[\s:—\-]*\n*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:\*\*)?(?:Analysis|Assistant|Response)(?:\*\*)?[\s:]*\n*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^Step\s+\d+[:.]\s*", "", text)
    return text.strip()


def format_response(text: str) -> str:
    """Convert markdown to clean semantic HTML using CSS classes."""
    if not text:
        return text

    text = text.strip()
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Apply inline bold before line processing
    text = re.sub(r"\*\*(.*?)\*\*", r'<strong class="rc-b">\1</strong>', text)

    lines = text.split("\n")
    html_parts: List[str] = []
    list_buf: List[str] = []
    list_kind: Optional[str] = None
    para_buf: List[str] = []

    def flush_para() -> None:
        nonlocal para_buf
        if para_buf:
            content = " ".join(s for s in para_buf if s.strip())
            if content.strip():
                html_parts.append(f'<p class="rc-p">{content}</p>')
            para_buf = []

    def flush_list() -> None:
        nonlocal list_buf, list_kind
        if list_buf:
            tag = list_kind or "ul"
            items = "".join(f"<li>{item}</li>" for item in list_buf)
            html_parts.append(f'<{tag} class="rc-list">{items}</{tag}>')
            list_buf = []
            list_kind = None

    for line in lines:
        s = line.strip()
        heading = re.match(r"^#{1,3}\s+(.+)$", s)
        bullet = re.match(r"^[-•*]\s+(.+)$", s)
        numbered = re.match(r"^\d+\.\s+(.+)$", s)

        if heading:
            flush_para()
            flush_list()
            html_parts.append(f'<p class="rc-p"><strong class="rc-h">{heading.group(1)}</strong></p>')
        elif bullet:
            flush_para()
            if list_kind == "ol":
                flush_list()
            list_kind = "ul"
            list_buf.append(bullet.group(1))
        elif numbered:
            flush_para()
            if list_kind == "ul":
                flush_list()
            list_kind = "ol"
            list_buf.append(numbered.group(1))
        elif not s:
            flush_para()
            flush_list()
        else:
            if list_buf:
                flush_list()
            para_buf.append(s)

    flush_para()
    flush_list()
    return "\n".join(html_parts)


_EMOTION_MAP = {
    "anxiety": re.compile(
        r"\b(anxious|anxiety|worry|worried|nervous|scared|afraid|tense|stressed|overwhelm)\b", re.IGNORECASE
    ),
    "depression": re.compile(
        r"\b(sad|depressed|hopeless|empty|numb|worthless|lonely|isolated|meaningless|despair|miserable)\b",
        re.IGNORECASE,
    ),
    "anger": re.compile(
        r"\b(angry|anger|furious|rage|frustrated|irritated|resentful|bitter)\b", re.IGNORECASE
    ),
    "grief": re.compile(
        r"\b(grief|grieving|loss|died|death|miss|mourning|bereaved|heartbroken|devastated)\b", re.IGNORECASE
    ),
    "burnout": re.compile(
        r"\b(burnout|burned out|burnt out|exhausted|drained|fatigued|depleted|running on empty)\b", re.IGNORECASE
    ),
    "trauma": re.compile(
        r"\b(trauma|traumatic|ptsd|abuse|assault|violence|flashback|nightmare)\b", re.IGNORECASE
    ),
    "loneliness": re.compile(
        r"\b(alone|lonely|isolated|no one|nobody|friendless|unwanted|disconnected)\b", re.IGNORECASE
    ),
}


def _detect_primary_emotion(text: str) -> Optional[str]:
    if not text:
        return None
    for emotion, pattern in _EMOTION_MAP.items():
        if pattern.search(text):
            return emotion
    return None


def get_direct_response(prompt: str, original_question: Optional[str] = None) -> str:
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"Direct LLM error (falling back to demo response): {e}")
        user_question = original_question or (prompt[:120] + "..." if len(prompt) > 120 else prompt)
        return (
            "**RISA (Demo Mode)**\n\n"
            "• The intelligent LLM backend isn't configured.\n"
            "• Set the environment variable **GROQ_API_KEY** to enable live answers.\n\n"
            "**You said:**\n"
            f"• {user_question}\n\n"
            "**Next steps:**\n"
            "1. Create a .env file with GROQ_API_KEY=your_key\n"
            "2. Restart the server\n"
            "3. Send your message again"
        )


_load_knowledge_base()
_load_learned_guidance()


# =================== RAG / CHROMADB ===================

# v2 collection uses local ONNX embeddings exclusively — never calls any external API.
# Renaming from "risa_conversations" ensures no collision with old OpenAI-embedded data.
_COLLECTION_NAME = "risa_conv_v2"
_RAG_TOP_K = 6
_RAG_DIST_THRESHOLD = 1.3  # cosine distance (0–2); lower = more similar

_chroma_client = None
_chroma_collection = None
_rag_ready = False
_rag_init_lock = threading.Lock()

# ── Graph-RAG singleton ────────────────────────────────────────────────────────
_graph_rag = None  # initialised by _init_graph_rag() at startup


def _init_graph_rag() -> None:
    global _graph_rag
    if _GraphRAG is None:
        return
    persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    os.makedirs(persist_dir, exist_ok=True)
    g = _GraphRAG(persist_dir)
    g.init()
    _graph_rag = g


def _get_rag_embedding_fn():
    """Local ONNX sentence embeddings — zero cost, no external API calls."""
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        return DefaultEmbeddingFunction()
    except Exception as exc:
        print(f"[RAG] DefaultEmbeddingFunction unavailable: {exc}")
    return None


def _init_chroma_sync() -> bool:
    """Thread-safe ChromaDB initializer using local ONNX embeddings. Idempotent."""
    global _chroma_client, _chroma_collection, _rag_ready
    with _rag_init_lock:
        if _rag_ready:
            return True
        try:
            import chromadb
            persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
            os.makedirs(persist_dir, exist_ok=True)
            ef = _get_rag_embedding_fn()
            if ef is None:
                print("[RAG] No embedding function — RAG disabled.")
                return False
            client = chromadb.PersistentClient(path=persist_dir)

            # Remove legacy collections that used OpenAI embeddings (incompatible dimensions)
            _LEGACY_NAMES = {"risa_conversations"}
            try:
                existing = {c.name for c in client.list_collections()}
                for name in _LEGACY_NAMES & existing:
                    client.delete_collection(name)
                    print(f"[RAG] Removed legacy collection '{name}' (incompatible embeddings).")
            except Exception:
                pass

            # Create or open the v2 local-embedding collection
            try:
                col = client.get_or_create_collection(
                    name=_COLLECTION_NAME,
                    embedding_function=ef,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                print(f"[RAG] Collection error — recreating: {exc}")
                try:
                    client.delete_collection(_COLLECTION_NAME)
                except Exception:
                    pass
                col = client.create_collection(
                    name=_COLLECTION_NAME,
                    embedding_function=ef,
                    metadata={"hnsw:space": "cosine"},
                )

            _chroma_client = client
            _chroma_collection = col
            _rag_ready = True
            print(f"[RAG] Ready — local ONNX embeddings | stored docs: {col.count()}")
            return True
        except Exception as exc:
            print(f"[RAG] Init failed: {exc}")
            return False


def _get_chroma_collection():
    """Return the ready collection, or attempt lazy init if startup hasn't finished yet."""
    if _rag_ready and _chroma_collection is not None:
        return _chroma_collection
    if _init_chroma_sync():
        return _chroma_collection
    return None


def _rag_device_key(raw_device_id: str) -> str:
    """Hash raw device UUID so the DB never stores the actual client identifier."""
    return hashlib.sha256(raw_device_id.encode("utf-8")).hexdigest()[:40]


def _rag_store_turn(raw_device_id: str, role: str, content: str) -> None:
    if not raw_device_id or not content or not content.strip():
        return
    collection = _get_chroma_collection()
    if collection is None:
        return
    dev_key = _rag_device_key(raw_device_id)
    try:
        doc_id = f"{dev_key}_{role}_{int(time.time() * 1000)}_{_uuid_mod.uuid4().hex[:8]}"
        collection.add(
            documents=[content.strip()[:1000]],
            ids=[doc_id],
            metadatas=[{"device_key": dev_key, "role": role, "ts": time.time()}],
        )
    except Exception as exc:
        print(f"[RAG] Store error: {exc}")
    # Graph-RAG: extract entities and update relationship graph (never blocks caller)
    if _graph_rag is not None:
        try:
            _graph_rag.add_turn(dev_key, content, role)
        except Exception:
            pass


def _rag_retrieve(raw_device_id: str, query: str) -> str:
    """
    Graph-RAG retrieval: expand query with entity-graph neighbours, then query ChromaDB.
    Falls back to plain vector retrieval when Graph-RAG is not initialised.
    """
    if not raw_device_id or not query or not query.strip():
        return ""
    collection = _get_chroma_collection()
    if collection is None:
        return ""
    total = collection.count()
    if total == 0:
        return ""

    dev_key = _rag_device_key(raw_device_id)

    # Graph-RAG query expansion — append related entity terms to the embedding query
    augmented_query = query.strip()
    if _graph_rag is not None and _graph_rag.ready:
        expanded = _graph_rag.expand_query(query, dev_key)
        if expanded:
            # Append up to 5 graph-expanded terms; ChromaDB embeds the whole string
            extra = " ".join(expanded[:5])
            augmented_query = f"{query.strip()} {extra}"

    try:
        results = collection.query(
            query_texts=[augmented_query[:600]],
            n_results=min(_RAG_TOP_K, total),
            where={"device_key": dev_key},
            include=["documents", "metadatas", "distances"],
        )
        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",  [[]])[0]
        if not docs:
            return ""

        relevant = [
            (doc, meta)
            for doc, meta, dist in zip(docs, metas, distances)
            if dist < _RAG_DIST_THRESHOLD
        ]
        if not relevant:
            return ""
        relevant.sort(key=lambda x: x[1].get("ts", 0))

        lines = ["[Relevant context from this user's previous sessions:]"]
        for doc, meta in relevant:
            role_label = meta.get("role", "unknown").capitalize()
            lines.append(f"  {role_label}: {doc[:400]}")

        # Append device's recurring-theme profile so the LLM can personalise
        if _graph_rag is not None and _graph_rag.ready:
            theme_profile = _graph_rag.get_device_profile(dev_key)
            if theme_profile:
                lines.append(theme_profile)

        # Inject personal profile (name, age, conditions, meds, allergies, etc.)
        if _PROFILE_STORE_AVAILABLE:
            user_profile = _get_profile(dev_key)
            profile_ctx = _profile_to_context(user_profile)
            if profile_ctx:
                lines.append(profile_ctx)

        lines.append("")
        return "\n".join(lines)
    except Exception as exc:
        print(f"[RAG] Retrieve error: {exc}")
        return ""


# =================== CHAT ENDPOINT ===================


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    start = time.time()

    user_message = (req.message or "").strip()
    if not user_message:
        return JSONResponse(
            content={
                "reply": format_response("Please type a message."),
                "detectedLang": "unknown",
                "translatedQuery": "",
                "userLocation": "Not required",
                "knowledgeUsed": [],
                "performanceMs": 0,
            },
            media_type="application/json; charset=utf-8",
        )

    # Stable per-device identity: prefer client-supplied UUID, fall back to IP hash
    raw_device_id = (req.device_id or "").strip()
    if raw_device_id:
        user_id = hashlib.md5(raw_device_id.encode("utf-8")).hexdigest()[:16]
    else:
        user_ip = request.client.host if request.client else "unknown"
        raw_device_id = user_ip
        user_id = hashlib.md5(user_ip.encode("utf-8")).hexdigest()[:16]

    if req.skip_translation:
        translated_query = user_message
        # Use the user's explicitly selected language (source_lang from UI dropdown)
        original_lang = (req.source_lang or "").strip() or _infer_source_lang(user_message) or "en"
    else:
        translated_query, original_lang = await translate_to_english(user_message)

    translated_query_out = "" if req.skip_translation else ensure_utf8(translated_query)
    combined_for_detection = f"{user_message}\n\n{translated_query}"
    knowledge_context, knowledge_used = _build_knowledge_context(translated_query)

    # Passively extract and store personal facts from the user's message (background)
    if _PROFILE_STORE_AVAILABLE:
        _en_text = translated_query or user_message
        asyncio.create_task(asyncio.to_thread(
            lambda: _merge_profile(
                _rag_device_key(raw_device_id),
                _extract_profile_facts(_en_text)
            )
        ))

    # RAG: retrieve semantically similar past turns for this device
    rag_context = await asyncio.to_thread(_rag_retrieve, raw_device_id, translated_query or user_message)

    # Detect primary emotion and inject as guidance if no knowledge context matched
    detected_emotion = _detect_primary_emotion(translated_query or user_message)
    if detected_emotion:
        emotion_note = f"\n[Detected emotional state: {detected_emotion}. Acknowledge this specifically in your response.]"
        knowledge_context = (knowledge_context or "") + emotion_note

    # Crisis Support Mode
    if _CRISIS_PATTERNS.search(combined_for_detection):
        response_text = (
            f"{RISA_CRISIS_TEMPLATE_EN}\n\n"
            "If you can, please tell me: are you in immediate danger right now, and is there someone nearby you can contact?"
        )
        asyncio.create_task(asyncio.to_thread(_rag_store_turn, raw_device_id, "user", translated_query or user_message))
        asyncio.create_task(asyncio.to_thread(_rag_store_turn, raw_device_id, "assistant", response_text))
        if _response_needs_translate_back(response_text, original_lang):
            translated_response = await translate_back(response_text, original_lang)
        else:
            translated_response = response_text
        final_response = format_response(translated_response)
        return JSONResponse(
            content={
                "reply": ensure_utf8(final_response),
                "detectedLang": original_lang,
                "translatedQuery": translated_query_out,
                "userLocation": "Not required",
                "knowledgeUsed": knowledge_used,
                "performanceMs": int((time.time() - start) * 1000),
            },
            media_type="application/json; charset=utf-8",
        )

    # Medical Emergency Mode — dispatch to specific first-aid steps
    if _MEDICAL_EMERGENCY_PATTERNS.search(combined_for_detection):
        response_text = _get_emergency_first_aid_response(combined_for_detection)
        asyncio.create_task(asyncio.to_thread(_rag_store_turn, raw_device_id, "user", translated_query or user_message))
        asyncio.create_task(asyncio.to_thread(_rag_store_turn, raw_device_id, "assistant", response_text))
        if _response_needs_translate_back(response_text, original_lang):
            translated_response = await translate_back(response_text, original_lang)
        else:
            translated_response = response_text
        final_response = format_response(translated_response)
        return JSONResponse(
            content={
                "reply": ensure_utf8(final_response),
                "detectedLang": original_lang,
                "translatedQuery": translated_query_out,
                "userLocation": "Not required",
                "knowledgeUsed": knowledge_used,
                "performanceMs": int((time.time() - start) * 1000),
            },
            media_type="application/json; charset=utf-8",
        )

    # Panic Attack Support Mode
    if _PANIC_PATTERNS.search(combined_for_detection):
        response_text = (
            "I'm here with you. Let's slow this down together.\n\n"
            "**Try this breathing for 60–90 seconds:**\n"
            "• Inhale through your nose for 4\n"
            "• Hold for 2\n"
            "• Exhale slowly for 6\n\n"
            "**Grounding (5-4-3-2-1):**\n"
            "• 5 things you can see\n"
            "• 4 things you can feel\n"
            "• 3 things you can hear\n"
            "• 2 things you can smell\n"
            "• 1 thing you can taste\n\n"
            "If you want, tell me what you're noticing in your body right now (e.g., chest tightness, dizziness, racing thoughts)."
        )
        asyncio.create_task(asyncio.to_thread(_rag_store_turn, raw_device_id, "user", translated_query or user_message))
        asyncio.create_task(asyncio.to_thread(_rag_store_turn, raw_device_id, "assistant", response_text))
        if _response_needs_translate_back(response_text, original_lang):
            translated_response = await translate_back(response_text, original_lang)
        else:
            translated_response = response_text
        final_response = format_response(translated_response)
        return JSONResponse(
            content={
                "reply": ensure_utf8(final_response),
                "detectedLang": original_lang,
                "translatedQuery": translated_query_out,
                "userLocation": "Not required",
                "knowledgeUsed": knowledge_used,
                "performanceMs": int((time.time() - start) * 1000),
            },
            media_type="application/json; charset=utf-8",
        )

    # Normal supportive conversation flow
    if re.search(r"\b(hi|hello|hey|assalamu alaikum|salam|good morning|good evening)\b", (translated_query or "").lower()):
        if rag_context:
            response_text = (
                "Welcome back — I'm glad you reached out again.\n\n"
                "How have things been since we last spoke? What's on your mind today?"
            )
        else:
            response_text = (
                "Hello — I'm glad you're here. I'm RISA, and I'm here to listen and support you.\n\n"
                "What's been on your mind? You can share whatever feels most pressing — "
                "whether it's stress, anxiety, low mood, relationship strain, burnout, or something you're just trying to make sense of."
            )
        _append_memory_turn(user_id, "user", translated_query or user_message)
        _append_memory_turn(user_id, "assistant", response_text)
        asyncio.create_task(asyncio.to_thread(_rag_store_turn, raw_device_id, "user", translated_query or user_message))
        asyncio.create_task(asyncio.to_thread(_rag_store_turn, raw_device_id, "assistant", response_text))
    else:
        try:
            llm = get_llm()
            messages = _build_llm_messages(
                user_id,
                translated_query,
                user_message,
                original_lang,
                knowledge_context,
                rag_context,
            )
            if messages is None:
                context_block = ""
                if knowledge_context:
                    context_block += knowledge_context + "\n\n"
                if rag_context:
                    context_block += rag_context + "\n\n"
                prompt = (
                    f"{RISA_SYSTEM_PROMPT}\n\n"
                    f"{context_block}"
                    f"User message (English):\n{translated_query}\n\n"
                    "Assistant response:"
                )
                response_text = get_direct_response(prompt, original_question=translated_query)
            else:
                response = llm.invoke(messages)
                response_text = response.content if hasattr(response, "content") else str(response)

            response_text = _clean_response(response_text)
            if _response_needs_translate_back(response_text, original_lang):
                try:
                    response_text = await asyncio.wait_for(translate_back(response_text, original_lang), timeout=6.0)
                except Exception:
                    pass
            _append_memory_turn(user_id, "user", translated_query or user_message)
            _append_memory_turn(user_id, "assistant", response_text)
            # Persist to ChromaDB for cross-session RAG (fire-and-forget — don't block response)
            asyncio.create_task(asyncio.to_thread(_rag_store_turn, raw_device_id, "user", translated_query or user_message))
            asyncio.create_task(asyncio.to_thread(_rag_store_turn, raw_device_id, "assistant", response_text))
        except Exception as e:
            print(f"RISA LLM error: {e}")
            response_text = (
                "I'm here with you, and I'm sorry — I'm having trouble generating a response right now.\n\n"
                "If you tell me in one sentence what you're going through, I'll try again."
            )

    # Keep learned notes in the translated text, while the outgoing reply stays in the user's language.
    if not knowledge_used and not re.search(r"\b(hi|hello|hey|assalamu alaikum|salam|good morning|good evening)\b", (translated_query or "").lower()):
        asyncio.create_task(asyncio.to_thread(_learn_from_conversation, translated_query or user_message, response_text))

    final_response = format_response(response_text)

    return JSONResponse(
        content={
            "reply": ensure_utf8(final_response),
            "detectedLang": original_lang,
            "translatedQuery": translated_query_out,
            "userLocation": "Not required",
            "knowledgeUsed": knowledge_used,
            "performanceMs": int((time.time() - start) * 1000),
        },
        media_type="application/json; charset=utf-8",
    )


@app.get("/health")
async def health():
    return {"status": "ok", "app": "RISA"}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """SSE streaming — HTTP 200 sent immediately; translation + RAG happen inside the generator."""
    user_message = (req.message or "").strip()

    if not user_message:
        async def _empty():
            yield f"data: {json.dumps({'done': True, 'html': '', 'lang': 'en', 'ms': 0})}\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    raw_device_id = (req.device_id or "").strip()
    if raw_device_id:
        user_id = hashlib.md5(raw_device_id.encode("utf-8")).hexdigest()[:16]
    else:
        user_ip = request.client.host if request.client else "unknown"
        raw_device_id = user_ip
        user_id = hashlib.md5(user_ip.encode("utf-8")).hexdigest()[:16]

    session_id = (req.session_id or "").strip() or None

    async def _sse():
        start = time.time()
        yield ": ok\n\n"
        print(f"[SSE] request user={user_id[:8]} session={session_id}")

        try:
            mem_key = f"{user_id}:{session_id}" if session_id else user_id

            if session_id and mem_key not in _conversation_memory and _SESSION_STORE_AVAILABLE:
                try:
                    _dev_key_local = _rag_device_key(raw_device_id)
                    _db_msgs = await asyncio.to_thread(_session_get_messages, session_id, _dev_key_local, 30)
                    if _db_msgs:
                        _conversation_memory[mem_key] = [
                            {"role": m["role"], "content": m["content"]} for m in _db_msgs
                        ]
                except Exception:
                    pass

            async def _translate_safe() -> Tuple[str, str]:
                if req.skip_translation:
                    # Use the user's explicitly selected language (source_lang from UI dropdown)
                    lang_out = (req.source_lang or "").strip() or _infer_source_lang(user_message) or "en"
                    return user_message, lang_out
                try:
                    return await asyncio.wait_for(translate_to_english(user_message), timeout=2.5)
                except Exception:
                    return user_message, _infer_source_lang(user_message)

            async def _rag_safe() -> str:
                try:
                    return await asyncio.wait_for(
                        asyncio.to_thread(_rag_retrieve, raw_device_id, user_message),
                        timeout=1.2,
                    )
                except Exception:
                    return ""

            (translated_query, original_lang), rag_context = await asyncio.gather(
                _translate_safe(), _rag_safe()
            )
            print(f"[SSE] lang={original_lang} rag={'yes' if rag_context else 'no'}")

            combined = f"{user_message}\n\n{translated_query}"
            knowledge_context, knowledge_used = _build_knowledge_context(translated_query)

            # Passively extract and store personal facts (background)
            if _PROFILE_STORE_AVAILABLE:
                _en_sse = translated_query or user_message
                asyncio.create_task(asyncio.to_thread(
                    lambda: _merge_profile(
                        _rag_device_key(raw_device_id),
                        _extract_profile_facts(_en_sse)
                    )
                ))

            detected_emotion = _detect_primary_emotion(translated_query or user_message)
            if detected_emotion:
                knowledge_context = (knowledge_context or "") + \
                    f"\n[Detected emotional state: {detected_emotion}. Acknowledge this specifically.]"

            is_crisis          = bool(_CRISIS_PATTERNS.search(combined))
            is_medical_emergency = bool(_MEDICAL_EMERGENCY_PATTERNS.search(combined))
            is_panic           = bool(_PANIC_PATTERNS.search(combined))
            is_greeting        = bool(re.search(
                r"\b(hi|hello|hey|assalamu alaikum|salam|good morning|good evening)\b",
                (translated_query or "").lower(),
            ))

            parts: List[str] = []

            async def _emit(text: str, translate: bool = False) -> Optional[str]:
                s = _clean_response(text.strip())
                if not s:
                    return None
                if translate and _response_needs_translate_back(s, original_lang):
                    s = await translate_back(s, original_lang)
                if not s or not s.strip():
                    return None
                parts.append(s.strip())
                return f"data: {json.dumps({'t': s.strip(), 'lang': original_lang}, ensure_ascii=False)}\n\n"

            if is_crisis:
                prewritten = (
                    f"{RISA_CRISIS_TEMPLATE_EN}\n\n"
                    "If you can, please tell me: are you in immediate danger right now, "
                    "and is there someone nearby you can contact?"
                )
                for para in prewritten.split("\n\n"):
                    if para.strip():
                        ev_line = await _emit(para.strip(), translate=True)
                        if ev_line:
                            yield ev_line

            elif is_medical_emergency:
                prewritten = _get_emergency_first_aid_response(combined)
                for para in prewritten.split("\n\n"):
                    if para.strip():
                        ev_line = await _emit(para.strip(), translate=True)
                        if ev_line:
                            yield ev_line

            elif is_panic:
                prewritten = (
                    "I'm here with you. Let's slow this down together.\n\n"
                    "Try this breathing for 60 to 90 seconds: Inhale through your nose for 4, "
                    "hold for 2, exhale slowly for 6.\n\n"
                    "Grounding: name 5 things you can see, 4 you can feel, 3 you can hear, "
                    "2 you can smell, 1 you can taste.\n\n"
                    "Tell me what you're noticing in your body right now."
                )
                for para in prewritten.split("\n\n"):
                    if para.strip():
                        ev_line = await _emit(para.strip(), translate=True)
                        if ev_line:
                            yield ev_line

            elif is_greeting:
                prewritten = (
                    "Welcome back. I'm glad you reached out again.\n\n"
                    "How have things been since we last spoke? What's on your mind today?"
                ) if rag_context else (
                    "Hello. I'm glad you're here. I'm RISA, and I'm here to listen and support you.\n\n"
                    "What's been on your mind? You can share whatever feels most pressing."
                )
                for para in prewritten.split("\n\n"):
                    if para.strip():
                        ev_line = await _emit(para.strip(), translate=True)
                        if ev_line:
                            yield ev_line

            else:
                try:
                    messages = _build_llm_messages(
                        user_id, translated_query, user_message,
                        original_lang, knowledge_context, rag_context,
                        mem_key=mem_key,
                    )
                    print(f"[SSE] calling LLM messages={len(messages or [])}")
                    llm_s = get_llm_stream()
                    sentence_buf = ""
                    _llm_deadline = time.time() + 20.0
                    async for chunk in llm_s.astream(messages or []):
                        if time.time() > _llm_deadline:
                            break
                        token = getattr(chunk, "content", "") or ""
                        if not token:
                            continue
                        sentence_buf += token
                        m = _SENT_END_RE.search(sentence_buf)
                        if m:
                            sentence = sentence_buf[: m.start() + 1]
                            sentence_buf = sentence_buf[m.end():]
                            ev_line = await _emit(sentence, translate=True)
                            if ev_line:
                                yield ev_line
                        elif len(sentence_buf) > 160:
                            ev_line = await _emit(sentence_buf, translate=True)
                            if ev_line:
                                sentence_buf = ""
                                yield ev_line
                    if sentence_buf.strip():
                        ev_line = await _emit(sentence_buf, translate=True)
                        if ev_line:
                            yield ev_line
                except Exception as _llm_exc:
                    print(f"[SSE] LLM error: {_llm_exc}")
                    ev_line = await _emit(
                        "I'm sorry, I'm having trouble right now. "
                        "Please try sending your message again."
                    )
                    if ev_line:
                        yield ev_line

            full_text = " ".join(parts).strip()
            print(f"[SSE] full_text len={len(full_text)}")
            _append_memory_turn(mem_key, "user", translated_query or user_message)
            _append_memory_turn(mem_key, "assistant", full_text)
            asyncio.create_task(asyncio.to_thread(_rag_store_turn, raw_device_id, "user", translated_query or user_message))
            asyncio.create_task(asyncio.to_thread(_rag_store_turn, raw_device_id, "assistant", full_text))

            if not knowledge_used and not is_crisis and not is_medical_emergency and not is_panic and not is_greeting and full_text.strip():
                asyncio.create_task(asyncio.to_thread(_learn_from_conversation, translated_query or user_message, full_text))

            final_html = format_response(full_text)

            session_id_out = session_id
            if _SESSION_STORE_AVAILABLE and full_text.strip():
                try:
                    _dev_key = _rag_device_key(raw_device_id)
                    if not session_id_out:
                        session_id_out = await asyncio.to_thread(_session_create, _dev_key)
                    title_text = (translated_query or user_message)[:60].strip()
                    asyncio.create_task(asyncio.to_thread(_session_update_title, session_id_out, title_text))
                    asyncio.create_task(asyncio.to_thread(_session_add_message, session_id_out, "user", user_message))
                    asyncio.create_task(asyncio.to_thread(_session_add_message, session_id_out, "assistant", final_html))
                except Exception as _exc:
                    print(f"[SSE] session persist error: {_exc}")
                    session_id_out = session_id

            done_ev = {
                "done": True,
                "html": ensure_utf8(final_html),
                "lang": original_lang,
                "ms": int((time.time() - start) * 1000),
                "session_id": session_id_out,
            }
            print(f"[SSE] done ms={done_ev['ms']} html_len={len(done_ev['html'])}")
            yield f"data: {json.dumps(done_ev, ensure_ascii=False)}\n\n"

        except Exception as _fatal:
            import traceback
            print(f"[SSE FATAL] {_fatal}")
            traceback.print_exc()
            yield f"data: {json.dumps({'done': True, 'html': '<p>Something went wrong. Please try again.</p>', 'lang': 'en', 'ms': int((time.time()-start)*1000), 'session_id': session_id})}\n\n"

    return StreamingResponse(
        _sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/stats")
async def stats():
    chroma_count = 0
    try:
        if _rag_ready and _chroma_collection is not None:
            chroma_count = await asyncio.to_thread(_chroma_collection.count)
    except Exception:
        pass
    with _learned_guidance_lock:
        learned_topics_count = len(_LEARNED_GUIDANCE.get("topics", []))
        learned_samples = sum(int(t.get("samples", 0)) for t in _LEARNED_GUIDANCE.get("topics", []))
    graph_stats = _graph_rag.get_stats() if _graph_rag is not None else {"ready": False}
    return {
        "app": "RISA",
        "version": "1.0",
        "active_sessions": len(_conversation_memory),
        "rag_documents": chroma_count,
        "rag_collection": _COLLECTION_NAME,
        "rag_enabled": _rag_ready,
        "rag_embedding": "local-onnx (all-MiniLM-L6-v2)",
        "graph_rag": graph_stats,
        "model": "llama-3.3-70b-versatile",
        "learned_topics": learned_topics_count,
        "learned_samples": learned_samples,
        "learned_guidance_path": _LEARNED_GUIDANCE_PATH,
    }


@app.get("/learned-topics")
async def learned_topics(limit: int = 10, x_admin_token: str = ""):
    """Return the most recent learned guidance topics for administration and debugging."""
    try:
        _require_learned_guidance_admin_token(x_admin_token.strip())
    except ValueError:
        return JSONResponse(status_code=403, content={"error": "Forbidden"})

    safe_limit = max(1, min(int(limit or 10), 50))
    with _learned_guidance_lock:
        topics = sorted(
            list(_LEARNED_GUIDANCE.get("topics", [])),
            key=lambda topic: int(topic.get("updated_at", 0) or 0),
            reverse=True,
        )
        total = len(topics)
        payload = [_learned_topic_summary(topic) for topic in topics[:safe_limit]]

    return {
        "count": total,
        "returned": len(payload),
        "topics": payload,
    }


@app.post("/clear")
async def clear_session(req: ChatRequest, request: Request):
    """Clear in-memory conversation history for the requesting device."""
    raw_device_id = (req.device_id or "").strip()
    if not raw_device_id:
        user_ip = request.client.host if request.client else "unknown"
        raw_device_id = user_ip
    user_id = hashlib.md5(raw_device_id.encode("utf-8")).hexdigest()[:16]
    cleared = user_id in _conversation_memory
    if cleared:
        del _conversation_memory[user_id]
    return {"status": "cleared" if cleared else "no_session"}


# =================== NEURAL TTS (edge-tts) ===================

class TTSRequest(BaseModel):
    text: str
    lang: Optional[str] = "en"
    voice: Optional[str] = None


# Microsoft Edge TTS neural voices — free, no API key required, natural-sounding
_EDGE_VOICES: Dict[str, str] = {
    # Bengali — best natural voices
    "bn":       "bn-BD-NabanitaNeural",
    "bn-BD":    "bn-BD-NabanitaNeural",
    "bn-IN":    "bn-IN-TanishaaNeural",
    # English
    "en":       "en-US-JennyNeural",
    "en-US":    "en-US-JennyNeural",
    "en-GB":    "en-GB-SoniaNeural",
    "en-AU":    "en-AU-NatashaNeural",
    # South Asian
    "hi":       "hi-IN-SwaraNeural",
    "hi-IN":    "hi-IN-SwaraNeural",
    "ur":       "ur-PK-UzmaNeural",
    "ur-PK":    "ur-PK-UzmaNeural",
    "ta":       "ta-IN-PallaviNeural",
    "ta-IN":    "ta-IN-PallaviNeural",
    "te":       "te-IN-ShrutiNeural",
    "te-IN":    "te-IN-ShrutiNeural",
    "ml":       "ml-IN-SobhanaNeural",
    "ml-IN":    "ml-IN-SobhanaNeural",
    "gu":       "gu-IN-DhwaniNeural",
    "gu-IN":    "gu-IN-DhwaniNeural",
    "kn":       "kn-IN-SapnaNeural",
    "kn-IN":    "kn-IN-SapnaNeural",
    "mr":       "mr-IN-AarohiNeural",
    "mr-IN":    "mr-IN-AarohiNeural",
    "pa":       "pa-IN-OjasNeural",
    "ne":       "ne-NP-HemkalaNeural",
    "ne-NP":    "ne-NP-HemkalaNeural",
    "si":       "si-LK-ThiliniNeural",
    "si-LK":    "si-LK-ThiliniNeural",
    # Middle East
    "ar":       "ar-SA-ZariyahNeural",
    "ar-SA":    "ar-SA-ZariyahNeural",
    "ar-EG":    "ar-EG-SalmaNeural",
    # European
    "fr":       "fr-FR-DeniseNeural",
    "fr-FR":    "fr-FR-DeniseNeural",
    "fr-CA":    "fr-CA-SylvieNeural",
    "de":       "de-DE-KatjaNeural",
    "de-DE":    "de-DE-KatjaNeural",
    "es":       "es-ES-ElviraNeural",
    "es-ES":    "es-ES-ElviraNeural",
    "es-MX":    "es-MX-DaliaNeural",
    "it":       "it-IT-ElsaNeural",
    "it-IT":    "it-IT-ElsaNeural",
    "pt":       "pt-BR-FranciscaNeural",
    "pt-BR":    "pt-BR-FranciscaNeural",
    "pt-PT":    "pt-PT-RaquelNeural",
    "nl":       "nl-NL-FennaNeural",
    "nl-NL":    "nl-NL-FennaNeural",
    "ru":       "ru-RU-SvetlanaNeural",
    "ru-RU":    "ru-RU-SvetlanaNeural",
    "pl":       "pl-PL-ZofiaNeural",
    "pl-PL":    "pl-PL-ZofiaNeural",
    "tr":       "tr-TR-EmelNeural",
    "tr-TR":    "tr-TR-EmelNeural",
    # East Asian
    "zh":       "zh-CN-XiaoxiaoNeural",
    "zh-CN":    "zh-CN-XiaoxiaoNeural",
    "zh-TW":    "zh-TW-HsiaoChenNeural",
    "ja":       "ja-JP-NanamiNeural",
    "ja-JP":    "ja-JP-NanamiNeural",
    "ko":       "ko-KR-SunHiNeural",
    "ko-KR":    "ko-KR-SunHiNeural",
    # Southeast Asian
    "id":       "id-ID-GadisNeural",
    "id-ID":    "id-ID-GadisNeural",
    "ms":       "ms-MY-YasminNeural",
    "ms-MY":    "ms-MY-YasminNeural",
    "th":       "th-TH-PremwadeeNeural",
    "th-TH":    "th-TH-PremwadeeNeural",
    "vi":       "vi-VN-HoaiMyNeural",
    "vi-VN":    "vi-VN-HoaiMyNeural",
    "tl":       "fil-PH-BlessicaNeural",
    "fil-PH":   "fil-PH-BlessicaNeural",
}


def _get_edge_voice(lang: str) -> str:
    lang = (lang or "en").strip()
    if lang in _EDGE_VOICES:
        return _EDGE_VOICES[lang]
    base = lang.split("-")[0].lower()
    if base in _EDGE_VOICES:
        return _EDGE_VOICES[base]
    return _EDGE_VOICES["en"]


def _strip_html_for_tts(text: str) -> str:
    """Remove HTML tags, markdown, and normalize whitespace for clean TTS input."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2800]


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """
    Neural TTS via Microsoft Edge — free, no API key, natural voice in 40+ languages.
    Streams MP3 audio immediately as it's generated (no buffering delay).
    Voice auto-selected by language; override with 'voice' field.
    """
    try:
        import edge_tts
    except ImportError:
        return JSONResponse(
            status_code=503,
            content={"error": "edge-tts not installed. Run: pip install edge-tts"},
        )

    clean = _strip_html_for_tts(req.text or "")
    if not clean:
        return JSONResponse(status_code=400, content={"error": "No text provided"})

    voice = req.voice or _get_edge_voice(req.lang or "en")
    print(f"[TTS] lang={req.lang!r} voice={voice!r} chars={len(clean)}")

    async def _audio_stream():
        try:
            communicate = edge_tts.Communicate(clean, voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        except Exception as exc:
            print(f"[TTS] Stream error voice={voice!r}: {exc}")

    return StreamingResponse(
        _audio_stream(),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store", "X-Voice": voice},
    )


# =================== SPEECH-TO-TEXT (Groq Whisper API — primary; faster-whisper — fallback) ===================

_WHISPER_MODEL_SIZE = "small"
_fw_model = None
_fw_lock  = threading.Lock()

_STT_CONTENT_TYPES: Dict[str, str] = {
    "webm": "audio/webm", "ogg": "audio/ogg",
    "m4a": "audio/mp4",   "mp4": "audio/mp4",
    "wav": "audio/wav",   "flac": "audio/flac",
}


async def _groq_whisper_transcribe(audio_data: bytes, ext: str, lang_hint: Optional[str]) -> str:
    """Call Groq Whisper large-v3-turbo via REST. Returns transcribed text.

    lang_hint is the user's explicitly selected language from the UI (e.g. 'bn', 'en').
    Passing it improves transcription accuracy and ensures script-correct output.
    """
    import httpx
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not set")

    content_type = _STT_CONTENT_TYPES.get(ext, "audio/webm")
    files = {"file": (f"audio.{ext}", audio_data, content_type)}
    data: Dict[str, str] = {"model": "whisper-large-v3-turbo", "response_format": "json"}
    if lang_hint:
        data["language"] = lang_hint  # honour UI language selection

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {groq_key}"},
            files=files,
            data=data,
        )
        resp.raise_for_status()
        return resp.json().get("text", "").strip()


def _load_whisper_model():
    global _fw_model
    if _fw_model is not None:
        return _fw_model
    with _fw_lock:
        if _fw_model is not None:
            return _fw_model
        try:
            from faster_whisper import WhisperModel
            print(f"[STT] Loading faster-whisper/{_WHISPER_MODEL_SIZE}…")
            _fw_model = WhisperModel(_WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
            print("[STT] faster-whisper ready")
        except ImportError:
            raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")
    return _fw_model


def _fw_transcribe_sync(audio_data: bytes, ext: str, lang: Optional[str]) -> str:
    import tempfile
    import os as _os
    model = _load_whisper_model()
    suffix = f".{ext}" if ext else ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name
    try:
        segments, _ = model.transcribe(
            tmp_path, language=lang, beam_size=5,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        try:
            _os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    lang: str = Form(default=""),
):
    """
    STT via Groq Whisper large-v3-turbo (primary) with faster-whisper fallback.
    Returns transcribed text AND the detected language of what was actually spoken.
    """
    audio_data = await file.read()
    if len(audio_data) < 500:
        return JSONResponse(status_code=400, content={"error": "Audio too short or empty"})

    filename = file.filename or "audio.webm"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "webm"
    lang_hint: Optional[str] = lang.split("-")[0].lower() if lang and lang not in ("", "auto", "unknown") else None

    text = ""

    # The response language is always the user's UI selection (lang_hint), not auto-detected.
    # Whisper uses the hint for accurate script-correct transcription.
    response_lang = lang_hint or "en"

    # Primary: Groq Whisper API (fast, no download, multilingual)
    try:
        text = await _groq_whisper_transcribe(audio_data, ext, lang_hint)
        if text:
            print(f"[STT] Groq lang={response_lang!r} chars={len(text)} preview={text[:80]!r}")
            return {"text": text, "lang": response_lang}
    except Exception as exc:
        print(f"[STT] Groq Whisper failed: {exc}")

    # Fallback: local faster-whisper
    try:
        text = await asyncio.to_thread(_fw_transcribe_sync, audio_data, ext, lang_hint)
        print(f"[STT] Whisper lang={response_lang!r} chars={len(text)} preview={text[:80]!r}")
        return {"text": text, "lang": response_lang}
    except RuntimeError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    except Exception as exc:
        print(f"[STT] Error: {exc}")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/favicon.ico")
async def favicon():
    file_path = os.path.join(os.path.dirname(__file__), "favicon.ico")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "favicon.ico not found"})


# =================== SESSION MANAGEMENT API ===================


class _SessionRenameRequest(BaseModel):
    device_id: str
    title: str


class _SessionCreateRequest(BaseModel):
    device_id: Optional[str] = None


@app.get("/sessions")
async def get_sessions(device_id: str = "", request: Request = None):
    """List all sessions for a device, ordered by most recent activity."""
    if not _SESSION_STORE_AVAILABLE:
        return {"sessions": []}
    raw_id = device_id.strip() or (request.client.host if request and request.client else "unknown")
    dev_key = _rag_device_key(raw_id)
    sessions = await asyncio.to_thread(_session_list, dev_key)
    return {"sessions": sessions}


@app.post("/sessions")
async def create_session_ep(req: _SessionCreateRequest, request: Request):
    """Create a new empty session. Returns session_id."""
    if not _SESSION_STORE_AVAILABLE:
        return JSONResponse(status_code=503, content={"error": "Session store unavailable"})
    raw_id = (req.device_id or "").strip() or (request.client.host if request.client else "unknown")
    dev_key = _rag_device_key(raw_id)
    session_id = await asyncio.to_thread(_session_create, dev_key)
    return {"session_id": session_id}


@app.delete("/sessions/{session_id}")
async def delete_session_ep(session_id: str, device_id: str = "", request: Request = None):
    """Delete a session and all its messages."""
    if not _SESSION_STORE_AVAILABLE:
        return JSONResponse(status_code=503, content={"error": "Session store unavailable"})
    raw_id = device_id.strip() or (request.client.host if request and request.client else "unknown")
    dev_key = _rag_device_key(raw_id)
    ok = await asyncio.to_thread(_session_delete, session_id, dev_key)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    return {"status": "deleted"}


@app.patch("/sessions/{session_id}")
async def rename_session_ep(session_id: str, req: _SessionRenameRequest, request: Request):
    """Rename a session title."""
    if not _SESSION_STORE_AVAILABLE:
        return JSONResponse(status_code=503, content={"error": "Session store unavailable"})
    raw_id = (req.device_id or "").strip() or (request.client.host if request.client else "unknown")
    dev_key = _rag_device_key(raw_id)
    ok = await asyncio.to_thread(_session_rename, session_id, dev_key, req.title)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    return {"status": "renamed"}


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, device_id: str = "", request: Request = None):
    """Get all messages for a session (ownership-verified)."""
    if not _SESSION_STORE_AVAILABLE:
        return {"messages": []}
    raw_id = device_id.strip() or (request.client.host if request and request.client else "unknown")
    dev_key = _rag_device_key(raw_id)
    messages = await asyncio.to_thread(_session_get_messages, session_id, dev_key)
    return {"messages": messages}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
