"""
Convexity Terminal — Theme Engine

Auto-assigns tickers to investment themes based on industry classification,
business description keywords, and company name matching. FMP provides metadata,
with yfinance and disk cache as fallbacks.
"""

import os
import json
import requests
import streamlit as st

# ── File paths ───────────────────────────────────────────────────────────────

_DIR = os.path.dirname(os.path.abspath(__file__))
THEMES_FILE = os.path.join(_DIR, "themes.json")
META_CACHE_FILE = os.path.join(_DIR, "meta_cache.json")

# ── Default themes (ETF benchmarks per sector) ───────────────────────────────

DEFAULT_THEMES = {
    "themes": {
        "Space & Satellite": {"etfs": ["UFO", "ARKX"], "tickers": ["RKLB", "ASTS"]},
        "Uranium / Nuclear": {"etfs": ["URA"], "tickers": []},
        "AI Semis": {"etfs": ["SMH", "SOXX"], "tickers": []},
        "Crypto / Digital Assets": {"etfs": ["BITQ", "BITO"], "tickers": []},
        "Biotech": {"etfs": ["XBI"], "tickers": []},
        "Critical Materials": {"etfs": ["REMX"], "tickers": ["UAMY"]},
        "Defense & Drones": {"etfs": ["ITA"], "tickers": []},
        "Renewables / Grid": {"etfs": ["QCLN"], "tickers": []},
        "Non-Profitable Tech": {"etfs": ["ARKK"], "tickers": []},
        "Quantum Computing": {"etfs": ["QTUM"], "tickers": []},
        "Copper & Metals": {"etfs": ["COPX", "XME"], "tickers": []},
        "Robotics / Broad AI": {"etfs": ["BOTZ", "ROBO"], "tickers": []},
        "Fintech": {"etfs": ["FINX"], "tickers": ["HOOD"]},
        "EV & Autonomy": {"etfs": ["DRIV"], "tickers": ["TSLA"]},
        "Infrastructure": {"etfs": ["PAVE"], "tickers": []},
        "Mag 7": {"etfs": ["MAGS"], "tickers": []},
        "Cybersecurity": {"etfs": ["HACK"], "tickers": []},
        "Edge AI": {"etfs": ["BOTZ"], "tickers": []},
        "AI & Data": {"etfs": ["AIQ"], "tickers": ["PLTR"]},
        "AI Software": {"etfs": ["IGV"], "tickers": []},
        "Clean Energy": {"etfs": ["ICLN", "TAN"], "tickers": []},
    }
}

# ── Auto-theme rules ─────────────────────────────────────────────────────────
# Maps tickers to themes using industry + business description keywords.

THEME_RULES = {
    "Space & Satellite": {
        "industries": [],
        "desc_keywords": ["satellite", "rocket lab", "launch vehicle", "orbital", "spaceflight",
                          "low earth orbit", "space station", "spaceport"],
        "name_keywords": ["space", "rocket"],
    },
    "Uranium / Nuclear": {
        "industries": [],
        "desc_keywords": ["uranium", "nuclear fuel", "isotope separation", "enrichment",
                          "small modular reactor", "nuclear reactor"],
    },
    "AI Semis": {
        "industries": ["semiconductors", "semiconductor equipment & materials"],
        "desc_keywords": ["semiconductor fabricat", "chip manufactur", "semiconductor company",
                          "designs and sells semiconductor"],
    },
    "Crypto / Digital Assets": {
        "industries": [],
        "desc_keywords": ["bitcoin", "cryptocurrency", "blockchain", "stablecoin",
                          "digital asset", "crypto mining", "bitcoin mining"],
    },
    "Biotech": {
        "industries": ["biotechnology"],
        "desc_keywords": ["biotechnology company", "immunotherapy", "oncology drug"],
    },
    "Critical Materials": {
        "industries": ["other industrial metals & mining", "coking coal", "coal", "industrial materials"],
        "desc_keywords": ["antimony", "lithium mining", "rare earth", "critical mineral",
                          "critical metal", "strategic mineral"],
    },
    "Defense & Drones": {
        "industries": [],
        "desc_keywords": ["defense contract", "defense system", "drone", "military",
                          "unmanned aerial", "defense department", "tactical"],
    },
    "Fintech": {
        "industries": [],
        "desc_keywords": ["fintech", "trading platform", "brokerage service",
                          "payment processing", "neobank", "commission-free trading"],
    },
    "EV & Autonomy": {
        "industries": ["auto manufacturers"],
        "desc_keywords": ["electric vehicle", "ev charging",
                          "battery electric", "manufactures electric"],
    },
    "Cybersecurity": {
        "industries": [],
        "desc_keywords": ["cybersecurity", "cyber security", "threat detection",
                          "endpoint security", "zero trust"],
    },
    "AI & Data": {
        "industries": [],
        "desc_keywords": ["artificial intelligence", "machine learning platform",
                          "ai infrastructure", "ai industry", "generative ai",
                          "data analytics platform", "full-stack infrastructure for ai"],
    },
    "Edge AI / IoT": {
        "industries": [],
        "desc_keywords": ["edge computing", "internet of things", "embedded system",
                          "industrial computing", "edge ai", "ruggedized computing"],
    },
    "Clean Energy": {
        "industries": ["solar"],
        "desc_keywords": ["solar panel", "wind farm", "clean energy company",
                          "renewable energy company"],
    },
    "Robotics & Automation": {
        "industries": [],
        "desc_keywords": ["robotics company", "industrial automation", "autonomous robot"],
    },
    "Quantum Computing": {
        "industries": [],
        "desc_keywords": ["quantum comput", "quantum processor", "qubit"],
    },
}

# ── Metadata cache (disk-backed) ─────────────────────────────────────────────

def _load_meta_cache():
    try:
        if os.path.exists(META_CACHE_FILE):
            with open(META_CACHE_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_meta_cache(cache):
    try:
        with open(META_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


_META_DISK_CACHE = _load_meta_cache()


# ── API key helpers ──────────────────────────────────────────────────────────

def _get_fmp_key():
    """Get FMP API key from .env file or Streamlit secrets."""
    env_path = os.path.join(_DIR, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path) as f:
                for line in f:
                    if line.startswith("FMP_API_KEY="):
                        return line.strip().split("=", 1)[1]
        except Exception:
            pass
    try:
        return st.secrets.get("FMP_API_KEY", "")
    except Exception:
        return ""


def _get_anthropic_key():
    """Get Anthropic API key from .env file or Streamlit secrets."""
    env_path = os.path.join(_DIR, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path) as f:
                for line in f:
                    if line.startswith("ANTHROPIC_API_KEY="):
                        return line.strip().split("=", 1)[1]
        except Exception:
            pass
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        return ""


# ── Metadata fetching ────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_ticker_metadata(ticker):
    """Fetch industry, description, and name for theme auto-assignment.
    Tries FMP first (reliable on Cloud), then yfinance, then disk cache."""
    import yfinance as yf

    fmp_key = _get_fmp_key()
    if fmp_key:
        try:
            r = requests.get(
                f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={fmp_key}",
                timeout=8)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    d = data[0]
                    result = {
                        "industry": (d.get("industry") or "").lower(),
                        "sector": (d.get("sector") or "").lower(),
                        "desc": (d.get("description") or "").lower(),
                        "name": (d.get("companyName") or "").lower(),
                    }
                    _META_DISK_CACHE[ticker] = result
                    _save_meta_cache(_META_DISK_CACHE)
                    return result
        except Exception:
            pass

    # Fallback: yfinance
    try:
        info = yf.Ticker(ticker).info
        if info and any(info.get(k) for k in ["industry", "sector", "longBusinessSummary"]):
            result = {
                "industry": (info.get("industry") or "").lower(),
                "sector": (info.get("sector") or "").lower(),
                "desc": (info.get("longBusinessSummary") or "").lower(),
                "name": (info.get("shortName") or info.get("longName") or "").lower(),
            }
            _META_DISK_CACHE[ticker] = result
            _save_meta_cache(_META_DISK_CACHE)
            return result
    except Exception:
        pass

    # Final fallback: disk cache
    if ticker in _META_DISK_CACHE:
        return _META_DISK_CACHE[ticker]
    return {"industry": "", "sector": "", "desc": "", "name": ""}


# ── Theme assignment ─────────────────────────────────────────────────────────

def auto_assign_themes(ticker, metadata=None):
    """Auto-assign a ticker to themes based on industry + description + company name.
    Falls back to yfinance sector if no specific theme matches."""
    if metadata is None:
        metadata = _fetch_ticker_metadata(ticker)
    industry = metadata.get("industry", "")
    desc = metadata.get("desc", "")
    name = metadata.get("name", "").lower()
    industry_norm = industry.replace(" - ", " ").replace("-", " ").strip()
    matches = []
    for theme_name, rules in THEME_RULES.items():
        rule_industries = [i.lower().replace(" - ", " ").replace("-", " ") for i in rules.get("industries", [])]
        if industry_norm and industry_norm in rule_industries:
            matches.append(theme_name)
            continue
        if desc and any(kw in desc for kw in rules.get("desc_keywords", [])):
            matches.append(theme_name)
            continue
        if name and any(kw in name for kw in rules.get("name_keywords", [])):
            matches.append(theme_name)
    if matches:
        return matches
    sector = metadata.get("sector", "").title()
    return [sector] if sector else ["Other"]


# ── Theme data management ────────────────────────────────────────────────────

def _migrate_themes(raw):
    """Migrate old flat format to new {themes: {name: {etfs:[], tickers:[]}}}."""
    if "themes" in raw and isinstance(raw["themes"], dict):
        for name, val in raw["themes"].items():
            if isinstance(val, list):
                raw["themes"][name] = {"etfs": [], "tickers": val}
            elif isinstance(val, dict):
                val.setdefault("etfs", [])
                val.setdefault("tickers", [])
        return raw
    new = {"themes": {}}
    etf_map = {
        "Space & Satellite": ["UFO", "ARKX"],
        "Fintech": ["FINX"],
        "Critical Materials": ["REMX"],
        "EV & Autonomy": ["DRIV"],
        "AI & Data": ["AIQ"],
        "AI/HPC": ["SMH"],
        "Nuclear": ["URA"],
        "Digital Assets": ["BITQ", "BITO"],
        "Edge AI": ["BOTZ"],
        "Drones": ["ITA"],
        "BioTech": ["XBI"],
        "BTC Mining": ["BITQ"],
        "Clean Energy": ["ICLN", "TAN"],
        "Photonics & Sensors": ["BOTZ"],
    }
    for name, tickers in raw.items():
        if isinstance(tickers, list):
            new["themes"][name] = {"etfs": etf_map.get(name, []), "tickers": tickers}
        elif isinstance(tickers, dict):
            tickers.setdefault("etfs", [])
            tickers.setdefault("tickers", [])
            new["themes"][name] = tickers
    return new


def load_themes():
    if os.path.exists(THEMES_FILE):
        try:
            with open(THEMES_FILE) as f:
                raw = json.load(f)
            migrated = _migrate_themes(raw)
            if raw != migrated:
                save_themes(migrated)
            return migrated
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_THEMES))


def save_themes(themes_data):
    try:
        with open(THEMES_FILE, "w") as f:
            json.dump(themes_data, f, indent=2)
    except Exception:
        pass


def get_theme_tickers(themes_data):
    """Extract {theme_name: [user_tickers]} from themes structure."""
    themes_dict = themes_data.get("themes", {})
    return {name: val.get("tickers", []) for name, val in themes_dict.items()}


def get_ticker_themes(themes_data, ticker):
    """Return themes for a ticker. Manual overrides take priority over auto-assignment."""
    themes_dict = themes_data.get("themes", {})
    manual = [name for name, val in themes_dict.items() if ticker in val.get("tickers", [])]
    if manual:
        return manual
    return auto_assign_themes(ticker)


def get_theme_etfs(themes_data):
    """Extract {theme_name: [etf_tickers]} from themes structure."""
    themes_dict = themes_data.get("themes", {})
    return {name: val.get("etfs", []) for name, val in themes_dict.items()}


def get_all_etf_tickers(themes_data):
    """Get unique set of all ETF benchmark tickers across all themes."""
    etfs = set()
    for val in themes_data.get("themes", {}).values():
        etfs.update(val.get("etfs", []))
    return sorted(etfs)
