"""
Graph-RAG for RISA Mental Health Chatbot.

Architecture:
  Static layer  — pre-wired mental-health knowledge edges (anxiety→insomnia, etc.)
  Dynamic layer — per-device co-occurrence edges built from live conversations
  Retrieval     — 2-hop BFS entity expansion → richer ChromaDB query text
  Profile       — device-level recurring-theme summary injected into system prompt

100% local, zero-cost, no external API.
Requires: pip install networkx>=3.0
"""

import json
import os
import re
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
    _NX = True
except ImportError:
    _NX = False
    nx = None  # type: ignore

# ── Canonical entity names + aliases ─────────────────────────────────────────
_MH_ENTITIES: Dict[str, List[str]] = {
    # Emotions
    "anxiety":        ["anxious", "anxiousness", "worry", "worried", "nervous", "nervousness", "apprehensive"],
    "depression":     ["depressed", "depressive", "hopeless", "hopelessness", "worthless", "worthlessness"],
    "stress":         ["stressed", "overwhelmed", "pressure", "overloaded", "burden", "stressful"],
    "grief":          ["grieving", "mourning", "bereavement", "bereaved", "mourn"],
    "anger":          ["angry", "rage", "furious", "irritable", "frustrated", "frustration", "irritated"],
    "loneliness":     ["lonely", "isolated", "alone", "disconnected", "no one cares", "no one understands"],
    "burnout":        ["burnt out", "burned out", "drained", "depleted", "exhausted mentally"],
    "trauma":         ["traumatic", "ptsd", "flashback", "flashbacks", "traumatized"],
    "guilt":          ["guilty", "shame", "shameful", "regret", "regrets", "remorse"],
    "fear":           ["fearful", "scared", "afraid", "phobia", "terrified", "frightened"],
    # Symptoms
    "insomnia":       ["can't sleep", "cannot sleep", "sleep problems", "sleepless", "woke up early", "sleep disorder", "trouble sleeping", "hard to sleep"],
    "fatigue":        ["tired", "no energy", "weak", "low energy", "constant tiredness"],
    "panic attack":   ["heart racing", "heart pounding", "can't breathe", "chest tightening", "panic attacks"],
    "overthinking":   ["racing thoughts", "intrusive thoughts", "rumination", "overthink", "overthinks", "can't stop thinking"],
    "crying":         ["cry", "tears", "sobbing", "weeping", "burst into tears", "feel like crying"],
    "numbness":       ["numb", "emptiness", "feeling nothing", "emotionally numb", "feel empty", "detached from reality"],
    "appetite loss":  ["not eating", "lost appetite", "no appetite", "skipping meals"],
    "concentration":  ["can't focus", "can't concentrate", "brain fog", "memory problems", "forgetful"],
    # Life domains
    "work":           ["job", "career", "office", "boss", "colleague", "workplace", "coworker", "work life"],
    "family":         ["parents", "mother", "father", "sibling", "children", "parent", "mom", "dad", "in-laws"],
    "relationship":   ["partner", "spouse", "marriage", "divorce", "breakup", "boyfriend", "girlfriend", "ex"],
    "finances":       ["money", "debt", "financial", "bills", "salary", "afford", "broke", "financial stress"],
    "health":         ["illness", "sick", "medical", "doctor", "hospital", "disease", "chronic pain"],
    "spirituality":   ["prayer", "dua", "dhikr", "salah", "faith", "religion", "allah", "god", "quran", "iman"],
    "studies":        ["school", "university", "exam", "assignment", "student", "grade", "fail", "academic"],
    "social media":   ["instagram", "facebook", "tiktok", "comparison", "likes", "followers", "online"],
    "loss":           ["death", "died", "passed away", "deceased", "funeral", "losing someone"],
    # Coping strategies
    "breathing":      ["deep breath", "breathing exercise", "breathe slowly", "inhale", "exhale", "box breathing"],
    "meditation":     ["mindfulness", "grounding", "relaxation", "body scan", "guided meditation"],
    "exercise":       ["walk", "walking", "running", "yoga", "physical activity", "workout", "gym"],
    "therapy":        ["therapist", "counselor", "counseling", "psychologist", "psychiatrist", "mental health professional"],
    "social support": ["talk to someone", "friend", "support system", "open up", "reach out", "trusted person"],
    "journaling":     ["write down", "writing feelings", "diary", "journal", "expressing feelings"],
    "grounding":      ["5-4-3-2-1", "grounding technique", "sensory", "present moment"],
}

# Build reverse lookup: alias/canonical → canonical (longest-first for greedy matching)
_ALIAS_TO_CANON: Dict[str, str] = {}
for _canon, _aliases in _MH_ENTITIES.items():
    _ALIAS_TO_CANON[_canon] = _canon
    for _alias in _aliases:
        _ALIAS_TO_CANON[_alias.lower()] = _canon

# Pre-compile regex patterns: longest alias first to avoid sub-string shadowing
_ENTITY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE), canon)
    for alias, canon in sorted(_ALIAS_TO_CANON.items(), key=lambda x: -len(x[0]))
]

# ── Static knowledge graph edges (source, target, relation, weight) ───────────
_STATIC_EDGES: List[Tuple[str, str, str, float]] = [
    # Triggers / causes
    ("work",           "stress",         "TRIGGERS",  2.0),
    ("work",           "burnout",        "TRIGGERS",  1.8),
    ("family",         "stress",         "TRIGGERS",  1.5),
    ("relationship",   "grief",          "TRIGGERS",  1.5),
    ("relationship",   "loneliness",     "TRIGGERS",  1.5),
    ("relationship",   "anger",          "TRIGGERS",  1.2),
    ("finances",       "anxiety",        "TRIGGERS",  2.0),
    ("finances",       "stress",         "TRIGGERS",  2.0),
    ("studies",        "anxiety",        "TRIGGERS",  1.5),
    ("studies",        "stress",         "TRIGGERS",  1.5),
    ("social media",   "anxiety",        "TRIGGERS",  1.2),
    ("social media",   "depression",     "TRIGGERS",  1.2),
    ("loss",           "grief",          "TRIGGERS",  2.5),
    ("loss",           "depression",     "TRIGGERS",  2.0),
    # Leads to
    ("stress",         "anxiety",        "LEADS_TO",  2.0),
    ("stress",         "burnout",        "LEADS_TO",  1.8),
    ("stress",         "depression",     "LEADS_TO",  1.5),
    ("loneliness",     "depression",     "LEADS_TO",  2.0),
    ("burnout",        "depression",     "LEADS_TO",  2.0),
    ("burnout",        "anxiety",        "LEADS_TO",  1.5),
    ("trauma",         "anxiety",        "CAUSES",    2.0),
    ("trauma",         "depression",     "CAUSES",    2.0),
    ("trauma",         "overthinking",   "CAUSES",    1.8),
    ("grief",          "depression",     "LEADS_TO",  1.8),
    ("grief",          "loneliness",     "CO_OCCURS", 1.5),
    # Anxiety symptoms
    ("anxiety",        "insomnia",       "CAUSES",    2.0),
    ("anxiety",        "panic attack",   "CAUSES",    2.0),
    ("anxiety",        "overthinking",   "CAUSES",    2.0),
    ("anxiety",        "fatigue",        "CAUSES",    1.5),
    ("anxiety",        "concentration",  "CAUSES",    1.5),
    # Depression symptoms
    ("depression",     "fatigue",        "CAUSES",    2.0),
    ("depression",     "insomnia",       "CAUSES",    1.8),
    ("depression",     "numbness",       "CAUSES",    2.0),
    ("depression",     "crying",         "CAUSES",    1.8),
    ("depression",     "appetite loss",  "CAUSES",    1.5),
    ("depression",     "concentration",  "CAUSES",    1.5),
    # Other symptom chains
    ("burnout",        "fatigue",        "CAUSES",    2.0),
    ("burnout",        "numbness",       "CAUSES",    1.5),
    ("grief",          "crying",         "CAUSES",    1.8),
    ("fear",           "panic attack",   "CAUSES",    1.5),
    ("guilt",          "depression",     "LEADS_TO",  1.5),
    ("anger",          "stress",         "CO_OCCURS", 1.2),
    # Co-occurrence
    ("anxiety",        "depression",     "CO_OCCURS", 2.0),
    ("anxiety",        "stress",         "CO_OCCURS", 2.0),
    ("anxiety",        "overthinking",   "CO_OCCURS", 1.8),
    ("loneliness",     "grief",          "CO_OCCURS", 1.5),
    ("guilt",          "anger",          "CO_OCCURS", 1.2),
    # Coping / helped by
    ("anxiety",        "breathing",      "HELPED_BY", 2.5),
    ("anxiety",        "meditation",     "HELPED_BY", 2.0),
    ("anxiety",        "grounding",      "HELPED_BY", 2.5),
    ("anxiety",        "therapy",        "HELPED_BY", 2.0),
    ("anxiety",        "spirituality",   "HELPED_BY", 1.5),
    ("anxiety",        "journaling",     "HELPED_BY", 1.5),
    ("panic attack",   "breathing",      "HELPED_BY", 3.0),
    ("panic attack",   "grounding",      "HELPED_BY", 3.0),
    ("stress",         "breathing",      "HELPED_BY", 2.0),
    ("stress",         "meditation",     "HELPED_BY", 2.0),
    ("stress",         "exercise",       "HELPED_BY", 2.0),
    ("stress",         "spirituality",   "HELPED_BY", 1.5),
    ("depression",     "exercise",       "HELPED_BY", 2.5),
    ("depression",     "social support", "HELPED_BY", 2.0),
    ("depression",     "therapy",        "HELPED_BY", 2.5),
    ("depression",     "journaling",     "HELPED_BY", 1.8),
    ("loneliness",     "social support", "HELPED_BY", 2.5),
    ("loneliness",     "therapy",        "HELPED_BY", 2.0),
    ("grief",          "therapy",        "HELPED_BY", 2.0),
    ("grief",          "social support", "HELPED_BY", 2.0),
    ("burnout",        "meditation",     "HELPED_BY", 2.0),
    ("burnout",        "exercise",       "HELPED_BY", 1.8),
    ("trauma",         "therapy",        "HELPED_BY", 3.0),
    ("anger",          "breathing",      "HELPED_BY", 2.0),
    ("anger",          "exercise",       "HELPED_BY", 2.0),
    ("overthinking",   "meditation",     "HELPED_BY", 2.0),
    ("overthinking",   "grounding",      "HELPED_BY", 2.5),
    ("overthinking",   "journaling",     "HELPED_BY", 2.0),
    ("insomnia",       "meditation",     "HELPED_BY", 2.0),
    ("insomnia",       "breathing",      "HELPED_BY", 1.8),
]


class GraphRAG:
    """
    Entity-relation knowledge graph layered on top of ChromaDB.

    Call sequence:
        g = GraphRAG("./chroma_db")
        g.init()                              # once at startup (background thread ok)
        g.add_turn(device_key, text, role)    # fire-and-forget after each turn
        terms = g.expand_query(query, device_key)   # before ChromaDB query
        profile = g.get_device_profile(device_key)  # inject into system prompt
    """

    def __init__(self, persist_dir: str) -> None:
        self.persist_dir = persist_dir
        self._graph_path = os.path.join(persist_dir, "graph_rag.json")
        self._lock = threading.RLock()
        self._graph = None          # nx.DiGraph once init'd
        self._device_entities: Dict[str, Dict[str, int]] = {}  # device_key → {entity: count}
        self._ready = False
        self._dirty = 0             # turns since last disk write

    @property
    def ready(self) -> bool:
        return self._ready

    # ── Initialisation ─────────────────────────────────────────────────────────

    def init(self) -> bool:
        if not _NX:
            print("[Graph-RAG] networkx unavailable — graph features disabled. Run: pip install networkx")
            return False
        with self._lock:
            if self._ready:
                return True
            try:
                g = nx.DiGraph()
                for src, dst, rel, w in _STATIC_EDGES:
                    g.add_edge(src, dst, relation=rel, weight=w, static=True)
                    # Weak reverse edge allows traversal in both directions
                    if not g.has_edge(dst, src):
                        g.add_edge(dst, src, relation=f"REV_{rel}", weight=round(w * 0.35, 2), static=True)
                self._graph = g
                self._load_from_disk()
                self._ready = True
                print(
                    f"[Graph-RAG] Ready — "
                    f"nodes:{g.number_of_nodes()}  edges:{g.number_of_edges()}  "
                    f"devices:{len(self._device_entities)}"
                )
                return True
            except Exception as exc:
                print(f"[Graph-RAG] Init error: {exc}")
                return False

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        if not os.path.exists(self._graph_path):
            return
        try:
            with open(self._graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded = 0
            for e in data.get("edges", []):
                self._graph.add_edge(
                    e["src"], e["dst"],
                    relation=e.get("rel", "CO_OCCURS"),
                    weight=float(e.get("w", 1.0)),
                    static=False,
                    device=e.get("dev", ""),
                )
                loaded += 1
            for dev, ents in data.get("device_entities", {}).items():
                self._device_entities[dev] = ents
            if loaded:
                print(f"[Graph-RAG] Loaded {loaded} dynamic edges from disk")
        except Exception as exc:
            print(f"[Graph-RAG] Disk load error: {exc}")

    def _save_to_disk(self) -> None:
        if self._graph is None:
            return
        try:
            edges = [
                {
                    "src": s, "dst": d,
                    "rel": a.get("relation", "CO_OCCURS"),
                    "w":   round(float(a.get("weight", 1.0)), 3),
                    "dev": a.get("device", ""),
                }
                for s, d, a in self._graph.edges(data=True)
                if not a.get("static", False)
            ]
            data = {
                "edges": edges,
                "device_entities": self._device_entities,
                "saved_at": time.time(),
            }
            tmp = self._graph_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self._graph_path)
        except Exception as exc:
            print(f"[Graph-RAG] Disk save error: {exc}")

    # ── Entity extraction ──────────────────────────────────────────────────────

    def extract_entities(self, text: str) -> List[str]:
        """Extract canonical mental-health entity names from free text."""
        found: Set[str] = set()
        for pattern, canon in _ENTITY_PATTERNS:
            if pattern.search(text):
                found.add(canon)
        return list(found)

    # ── Dynamic graph update ───────────────────────────────────────────────────

    def add_turn(self, device_key: str, text: str, role: str = "user") -> None:
        """
        Extract entities from a conversation turn and strengthen the device graph.
        Safe to call from a background thread.
        """
        if not self._ready or not text.strip():
            return
        entities = self.extract_entities(text)
        if not entities:
            return

        with self._lock:
            # Update per-device frequency counters
            dev_ents = self._device_entities.setdefault(device_key, {})
            for ent in entities:
                dev_ents[ent] = dev_ents.get(ent, 0) + 1

            # Add / strengthen co-occurrence edges between entities in the same turn
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1:]:
                    if e1 == e2:
                        continue
                    for src, dst in ((e1, e2), (e2, e1)):
                        if self._graph.has_edge(src, dst):
                            old = self._graph[src][dst].get("weight", 1.0)
                            self._graph[src][dst]["weight"] = min(old + 0.3, 8.0)
                        else:
                            self._graph.add_edge(
                                src, dst,
                                relation="CO_OCCURS",
                                weight=1.0,
                                static=False,
                                device=device_key,
                            )

            self._dirty += 1
            if self._dirty % 5 == 0:
                self._save_to_disk()

    # ── Query expansion ────────────────────────────────────────────────────────

    def expand_query(self, query: str, device_key: str = "", hops: int = 2) -> List[str]:
        """
        Return a ranked list of entity terms (seed + graph neighbours) to append to
        the ChromaDB query text for context-aware retrieval.

        Returns seed entities first, then hop-1/hop-2 neighbours by score.
        """
        if not self._ready or self._graph is None:
            return []
        seed = self.extract_entities(query)
        if not seed:
            return []

        scored: Dict[str, float] = {e: 3.0 for e in seed}
        frontier: Set[str] = set(seed)

        for hop in range(1, hops + 1):
            decay = 1.0 / hop
            next_frontier: Set[str] = set()
            for node in frontier:
                if node not in self._graph:
                    continue
                for nbr in self._graph.neighbors(node):
                    edge_w = self._graph[node][nbr].get("weight", 1.0)
                    score = edge_w * decay
                    if scored.get(nbr, 0.0) < score:
                        scored[nbr] = score
                    next_frontier.add(nbr)
            frontier = next_frontier - set(seed)

        # Boost entities this device mentions frequently (personalisation)
        if device_key and device_key in self._device_entities:
            for ent, freq in self._device_entities[device_key].items():
                if ent in scored:
                    scored[ent] += 0.15 * min(freq, 10)

        return [e for e, _ in sorted(scored.items(), key=lambda x: -x[1])]

    # ── Device profile ─────────────────────────────────────────────────────────

    def get_device_profile(self, device_key: str) -> str:
        """
        Return a one-line recurring-theme summary for this device, or '' if none.
        Injected into the system context so the LLM can personalise its response.
        """
        ents = self._device_entities.get(device_key, {})
        top = [(e, c) for e, c in sorted(ents.items(), key=lambda x: -x[1]) if c >= 2][:5]
        if not top:
            return ""
        themes = ", ".join(f"{e} (×{c})" for e, c in top)
        return f"[User's recurring themes: {themes}]"

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            "ready": self._ready,
            "nodes": self._graph.number_of_nodes() if self._graph else 0,
            "edges": self._graph.number_of_edges() if self._graph else 0,
            "devices": len(self._device_entities),
            "total_entity_mentions": sum(
                sum(v.values()) for v in self._device_entities.values()
            ),
        }

    def flush(self) -> None:
        """Force an immediate disk write (call at shutdown if needed)."""
        with self._lock:
            self._save_to_disk()
