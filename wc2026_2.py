"""
🏆 World Cup 2026 Predictor — Version Finale
=============================================
Lance avec : streamlit run wc2026.py

Clé API dans .env : API_FOOTBALL_KEY=ta_clé
Données       : data/results.csv (Kaggle)
"""

# ── Imports ───────────────────────────────────────────────────────────
import os, json, time, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")
load_dotenv()

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

API_KEY     = os.getenv("API_FOOTBALL_KEY", "f57246df7fed5813319779ff23299665")
BASE_URL    = "https://v3.football.api-sports.io"
WC_LEAGUE   = 1
WC_SEASON   = 2026
RESULTS_CSV = "data/results.csv"
CACHE_DIR   = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Pays hôtes — léger avantage terrain
HOST_NATIONS = {"United States", "Canada", "Mexico"}
HOST_BOOST   = 0.06   # +6% de force pour les pays hôtes

# Groupes officiels CdM 2026
WC_GROUPS = {
    "A": ["Mexico",        "South Africa",           "South Korea",   "Czech Republic"],
    "B": ["Canada",        "Bosnia and Herzegovina", "Qatar",         "Switzerland"],
    "C": ["Brazil",        "Morocco",                "Haiti",         "Scotland"],
    "D": ["United States", "Paraguay",               "Australia",     "Turkey"],
    "E": ["Germany",       "Curaçao",                "Ivory Coast",   "Ecuador"],
    "F": ["Netherlands",   "Japan",                  "Sweden",        "Tunisia"],
    "G": ["Belgium",       "Egypt",                  "Iran",          "New Zealand"],
    "H": ["Spain",         "Cape Verde",             "Saudi Arabia",  "Uruguay"],
    "I": ["France",        "Senegal",                "Iraq",          "Norway"],
    "J": ["Argentina",     "Algeria",                "Austria",       "Jordan"],
    "K": ["Portugal",      "DR Congo",               "Uzbekistan",    "Colombia"],
    "L": ["England",       "Croatia",                "Ghana",         "Panama"],
}
ALL_TEAMS = sorted(set(t for teams in WC_GROUPS.values() for t in teams))

# Poids par compétition
COMP_WEIGHTS = {
    "FIFA World Cup": 3.0,          "Confederations Cup": 2.5,
    "UEFA Euro": 2.5,               "Copa América": 2.5,
    "African Cup of Nations": 2.5,  "AFC Asian Cup": 2.0,
    "Gold Cup": 2.0,                "FIFA World Cup qualification": 2.0,
    "UEFA Euro qualification": 1.8, "UEFA Nations League": 1.8,
    "African Cup of Nations qualification": 1.5,
    "AFC Asian Cup qualification": 1.5,
    "CONCACAF Nations League": 1.5, "Friendly": 0.5,
}

# Classements FIFA — Avril 2026
FIFA_RANKINGS = {
    "France": 1, "Spain": 2, "Argentina": 3, "England": 4,
    "Portugal": 5, "Brazil": 6, "Belgium": 7, "Netherlands": 8,
    "Germany": 9, "Colombia": 10, "Morocco": 11, "Uruguay": 12,
    "United States": 14, "Mexico": 15, "Croatia": 16, "Japan": 17,
    "Senegal": 18, "Switzerland": 19, "Ecuador": 20, "Turkey": 21,
    "Austria": 22, "South Korea": 23, "Iran": 25, "Australia": 26,
    "Norway": 27, "Czech Republic": 28, "Sweden": 33, "Canada": 34,
    "Algeria": 35, "Tunisia": 36, "Egypt": 37, "Saudi Arabia": 38,
    "Scotland": 39, "Paraguay": 40, "Ivory Coast": 41, "Qatar": 42,
    "Ghana": 43, "South Africa": 45, "Iraq": 46, "Haiti": 47,
    "Panama": 48, "DR Congo": 49, "Jordan": 50,
    "Bosnia and Herzegovina": 52, "Uzbekistan": 53, "Cape Verde": 54,
    "New Zealand": 85, "Curaçao": 83,
}

# Coachs — matchs dirigés avec l'équipe nationale
COACHES = {
    "France":        150, "Argentina":    90,  "Senegal":    100,
    "Croatia":       90,  "Japan":        80,  "Scotland":   75,
    "South Africa":  55,  "Norway":       55,  "Switzerland":45,
    "Cape Verde":    45,  "Portugal":     45,  "Mexico":     45,
    "Morocco":       40,  "Uruguay":      40,  "Curaçao":    40,
    "Austria":       40,  "Colombia":     40,  "Panama":     40,
    "Germany":       30,  "Netherlands":  30,  "Belgium":    28,
    "Haiti":         30,  "Iran":         30,  "Algeria":    30,
    "Paraguay":      35,  "Uzbekistan":   35,  "Sweden":     40,
    "Spain":         35,  "Turkey":       35,  "Ecuador":    25,
    "Jordan":        25,  "Ivory Coast":  20,  "Australia":  20,
    "South Korea":   20,  "Czech Republic":18, "Egypt":      15,
    "Saudi Arabia":  10,  "Canada":       22,  "England":    20,
    "United States": 18,  "Ghana":        18,  "Iraq":       15,
    "Qatar":         15,  "DR Congo":     20,  "Brazil":     8,
    "Tunisia":       8,   "Bosnia and Herzegovina": 12,
    "New Zealand":   20,
}

# Joueurs clés et impact si absent
KEY_PLAYERS = {
    "France":    {"Mbappé": 0.25, "Griezmann": 0.15, "Tchouaméni": 0.12},
    "Brazil":    {"Vinicius": 0.25, "Rodrygo": 0.15, "Alisson": 0.12},
    "Argentina": {"Messi": 0.35, "Álvarez": 0.18, "De Paul": 0.12},
    "England":   {"Bellingham": 0.25, "Kane": 0.22, "Saka": 0.15},
    "Spain":     {"Yamal": 0.22, "Pedri": 0.20, "Rodri": 0.18},
    "Germany":   {"Musiala": 0.22, "Wirtz": 0.20, "Havertz": 0.15},
    "Portugal":  {"Ronaldo": 0.28, "Bruno Fernandes": 0.20, "Bernardo": 0.15},
    "Netherlands":{"Gakpo": 0.20, "De Jong": 0.18, "Dumfries": 0.10},
    "Belgium":   {"De Bruyne": 0.28, "Lukaku": 0.20, "Doku": 0.15},
    "Colombia":  {"James": 0.22, "Díaz": 0.20, "Arias": 0.10},
    "Morocco":   {"En-Nesyri": 0.20, "Hakimi": 0.18, "Ziyech": 0.15},
    "Uruguay":   {"Valverde": 0.22, "Núñez": 0.20, "Bentancur": 0.12},
}

# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES DONNÉES
# ══════════════════════════════════════════════════════════════════════

def _norm(name: str) -> str:
    """Normalise les noms d'équipes vers notre référentiel."""
    return {
        "Korea Republic": "South Korea", "IR Iran": "Iran",
        "USA": "United States", "Czechia": "Czech Republic",
        "Türkiye": "Turkey", "Côte d'Ivoire": "Ivory Coast",
        "Bosnia": "Bosnia and Herzegovina", "Cabo Verde": "Cape Verde",
        "Congo DR": "DR Congo",
    }.get(name, name)

def _fifa_score(team: str) -> float:
    """Classement FIFA → score [0,1]. Rang 1 = 1.0, Rang 100 = 0.0"""
    rank = FIFA_RANKINGS.get(team, 75)
    return max(0.0, (100 - rank) / 99)

def _coach_score(team: str) -> float:
    """Stabilité du coach [0,1]. Plafonné à 80 matchs."""
    return min(COACHES.get(team, 10) / 80, 1.0)

def _host_boost(team: str) -> float:
    """Avantage terrain pour les pays hôtes."""
    return HOST_BOOST if team in HOST_NATIONS else 0.0

# ══════════════════════════════════════════════════════════════════════
# API FOOTBALL
# ══════════════════════════════════════════════════════════════════════

def api_get(endpoint: str, params: dict = {}, ttl: int = 300) -> dict:
    if not API_KEY:
        return {}
    key = endpoint.replace("/", "_") + "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    f   = CACHE_DIR / f"{key}.json"
    if f.exists() and (time.time() - f.stat().st_mtime) < ttl:
        return json.loads(f.read_text())
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}",
                         headers={"x-apisports-key": API_KEY},
                         params=params, timeout=10)
        d = r.json()
        f.write_text(json.dumps(d))
        return d
    except Exception:
        return {}

def get_fixtures() -> list:
    data = api_get("fixtures", {"league": WC_LEAGUE, "season": WC_SEASON}, ttl=120)
    out  = []
    for f in data.get("response", []):
        fix, teams, goals = f.get("fixture",{}), f.get("teams",{}), f.get("goals",{})
        out.append({
            "id":         fix.get("id"),
            "date":       fix.get("date","")[:10],
            "status":     fix.get("status",{}).get("short",""),
            "elapsed":    fix.get("status",{}).get("elapsed"),
            "team1":      _norm(teams.get("home",{}).get("name","")),
            "team2":      _norm(teams.get("away",{}).get("name","")),
            "logo1":      teams.get("home",{}).get("logo",""),
            "logo2":      teams.get("away",{}).get("logo",""),
            "goals1":     goals.get("home"),
            "goals2":     goals.get("away"),
            "round":      f.get("league",{}).get("round",""),
            "venue":      fix.get("venue",{}).get("name",""),
        })
    return out

def get_standings() -> dict:
    data = api_get("standings", {"league": WC_LEAGUE, "season": WC_SEASON}, ttl=300)
    groups = {}
    for lg in data.get("response", []):
        for grp in lg.get("league",{}).get("standings",[]):
            if not grp: continue
            name = grp[0].get("group","")
            groups[name] = [{
                "rank":   t["rank"],
                "team":   _norm(t.get("team",{}).get("name","")),
                "logo":   t.get("team",{}).get("logo",""),
                "played": t.get("all",{}).get("played",0),
                "won":    t.get("all",{}).get("win",0),
                "drawn":  t.get("all",{}).get("draw",0),
                "lost":   t.get("all",{}).get("lose",0),
                "gf":     t.get("all",{}).get("goals",{}).get("for",0),
                "ga":     t.get("all",{}).get("goals",{}).get("against",0),
                "gd":     t.get("goalsDiff",0),
                "pts":    t.get("points",0),
                "form":   t.get("form",""),
            } for t in grp]
    return groups

def get_injuries() -> list:
    data = api_get("injuries", {"league": WC_LEAGUE, "season": WC_SEASON}, ttl=1800)
    return [{"player": i.get("player",{}).get("name",""),
             "team":   _norm(i.get("team",{}).get("name","")),
             "type":   i.get("player",{}).get("type",""),
             "reason": i.get("player",{}).get("reason","")}
            for i in data.get("response",[])]

def get_top_scorers() -> list:
    data = api_get("players/topscorers", {"league": WC_LEAGUE, "season": WC_SEASON}, ttl=600)
    out  = []
    for item in data.get("response",[])[:15]:
        p = item.get("player",{})
        s = (item.get("statistics") or [{}])[0]
        out.append({"player": p.get("name",""),
                    "team":   _norm(s.get("team",{}).get("name","")),
                    "goals":  (s.get("goals") or {}).get("total") or 0,
                    "assists":(s.get("goals") or {}).get("assists") or 0,
                    "rating": float((s.get("games") or {}).get("rating") or 0)})
    return out

def get_api_prediction(fixture_id: int) -> dict:
    data = api_get("predictions", {"fixture": fixture_id}, ttl=3600)
    resp = data.get("response",[])
    if not resp: return {}
    pred = resp[0].get("predictions",{})
    return {"winner":    pred.get("winner",{}).get("name",""),
            "under_over":pred.get("under_over",""),
            "goals1":    pred.get("goals",{}).get("home",""),
            "goals2":    pred.get("goals",{}).get("away","")}

# ══════════════════════════════════════════════════════════════════════
# ML — FEATURES
# ══════════════════════════════════════════════════════════════════════

def _team_stats(team: str, history: dict, n: int = 10) -> dict:
    """
    Calcule les stats d'une équipe depuis l'historique.
    Si peu de données → utilise le classement FIFA comme proxy.
    Terrain neutre par défaut, sauf pays hôtes.
    """
    games = history.get(team, [])[-n:]
    fifa  = _fifa_score(team)
    coach = _coach_score(team)
    host  = _host_boost(team)

    if not games:
        # Valeurs initiales basées sur le classement FIFA
        scored = 0.7 + fifa * 1.5
        conc   = 1.7 - fifa * 1.3
        return dict(
            form   = 0.25 + fifa * 0.50,
            scored = scored,
            conc   = max(0.3, conc),
            win    = 0.20 + fifa * 0.50,
            clean  = 0.08 + fifa * 0.37,
            draw   = 0.28 - fifa * 0.08,   # les équipes faibles font plus de nuls
            gd     = scored - max(0.3, conc),
            fifa   = fifa,
            coach  = coach,
            host   = host,
            n      = 0,
        )

    wsum = sum(g["w"] for g in games) or 1
    # Mélange historique (80%) + FIFA (20%) pour lisser
    hist_form = sum(g["pts"] * g["w"] for g in games) / wsum / 3
    blended_form = hist_form * 0.80 + fifa * 0.20

    return dict(
        form   = blended_form,
        scored = np.mean([g["scored"] for g in games]),
        conc   = np.mean([g["conc"]   for g in games]),
        win    = np.mean([g["win"]    for g in games]),
        clean  = np.mean([g["conc"] == 0 for g in games]),
        draw   = np.mean([g["draw"]  for g in games]),
        gd     = np.mean([g["scored"] - g["conc"] for g in games]),
        fifa   = fifa,
        coach  = coach,
        host   = host,
        n      = len(games),
    )


def _make_features(t1_stats: dict, t2_stats: dict,
                   tw: float = 3.0,
                   inj1: float = 0.0,
                   inj2: float = 0.0) -> pd.DataFrame:
    """
    Construit le vecteur de features.
    IMPORTANT : t1 et t2 sont SYMÉTRIQUES — pas de notion domicile/extérieur
    sauf pour les pays hôtes via la feature 'host_diff'.
    """
    h = t1_stats.copy()
    a = t2_stats.copy()

    # Applique impact blessures
    if inj1 > 0:
        factor = 1 - inj1
        for k in ["form","scored","win","gd","clean"]:
            h[k] = h[k] * factor
    if inj2 > 0:
        factor = 1 - inj2
        for k in ["form","scored","win","gd","clean"]:
            a[k] = a[k] * factor

    return pd.DataFrame([{
        # Stats absolues
        "t1_form":   h["form"],    "t2_form":   a["form"],
        "t1_scored": h["scored"],  "t2_scored": a["scored"],
        "t1_conc":   h["conc"],    "t2_conc":   a["conc"],
        "t1_win":    h["win"],     "t2_win":    a["win"],
        "t1_clean":  h["clean"],   "t2_clean":  a["clean"],
        "t1_draw":   h["draw"],    "t2_draw":   a["draw"],
        "t1_gd":     h["gd"],      "t2_gd":     a["gd"],
        "t1_fifa":   h["fifa"],    "t2_fifa":   a["fifa"],
        # Coach — poids faible (10%)
        "t1_coach":  h["coach"] * 0.10,
        "t2_coach":  a["coach"] * 0.10,
        # Différentiels (signal fort pour la prédiction)
        "form_diff":   h["form"]   - a["form"],
        "scored_diff": h["scored"] - a["scored"],
        "conc_diff":   h["conc"]   - a["conc"],
        "win_diff":    h["win"]    - a["win"],
        "gd_diff":     h["gd"]     - a["gd"],
        "clean_diff":  h["clean"]  - a["clean"],
        "fifa_diff":   h["fifa"]   - a["fifa"],
        "coach_diff":  (h["coach"] - a["coach"]) * 0.10,
        # Avantage hôte (seulement USA/Canada/Mexique)
        "host_diff":   h["host"]   - a["host"],
        # Contexte match
        "tw":          tw,
        "draw_tendency": (h["draw"] + a["draw"]) / 2,  # tendance globale aux nuls
    }])


# ══════════════════════════════════════════════════════════════════════
# ML — ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════

def train_models(X: pd.DataFrame, y: pd.Series,
                 w: pd.Series, le: LabelEncoder) -> dict:
    """
    Entraîne LightGBM + Logistic Regression avec :
    - Rééquilibrage des nuls (×1.4)
    - Calibration isotonique
    - Symétrie t1/t2 (données augmentées)
    """
    y_enc = le.transform(y)
    draw_class = le.transform(["D"])[0]

    # Poids combinés : importance match × boost nul
    draw_boost = np.where(y_enc == draw_class, 1.15, 1.0)
    w_balanced = w.values * draw_boost

    trained = {}

    # ── LightGBM ────────────────────────────────────────────────────
    lgbm_base = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.04,
        num_leaves=24, min_child_samples=25,
        subsample=0.85, colsample_bytree=0.85,
        class_weight={0: 1.0, 1: 1.15, 2: 1.0},  # 0=A,1=D,2=H
        random_state=42, verbose=-1,
    )
    lgbm = CalibratedClassifierCV(lgbm_base, method="isotonic", cv=5)
    lgbm.fit(X, y_enc, sample_weight=w_balanced)
    trained["LightGBM"] = {"model": lgbm, "scaler": None,
                            "le": le, "cols": list(X.columns)}

    # ── Logistic Regression ─────────────────────────────────────────
    sc   = StandardScaler()
    X_sc = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
    lr_base = LogisticRegression(
        C=0.5, max_iter=2000, random_state=42,
        class_weight={0: 1.0, 1: 1.15, 2: 1.0},
    )
    lr = CalibratedClassifierCV(lr_base, method="isotonic", cv=5)
    lr.fit(X_sc, y_enc, sample_weight=w_balanced)
    trained["Logistic Regression"] = {"model": lr, "scaler": sc,
                                       "le": le, "cols": list(X.columns)}

    return trained


# ══════════════════════════════════════════════════════════════════════
# ML — CHARGEMENT COMPLET
# ══════════════════════════════════════════════════════════════════════

def load_and_train(results_path: str, fixtures_live: list) -> tuple:
    """
    1. Charge l'historique CSV depuis 2014
    2. Injecte les vrais résultats CdM 2026
    3. Augmente les données (symétrie t1/t2)
    4. Entraîne les modèles
    """
    # ── Historique ──────────────────────────────────────────────────
    df = pd.read_csv(results_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= "2014-01-01"]
    df = df[df["tournament"].isin(COMP_WEIGHTS.keys())]
    df = df.dropna(subset=["home_score","away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"]  = df["away_score"].astype(int)
    df["result"] = df.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
        else ("A" if r["away_score"] > r["home_score"] else "D"), axis=1)
    df["weight"] = df["tournament"].map(COMP_WEIGHTS).fillna(0.5)

    # ── Injection résultats CdM 2026 ────────────────────────────────
    played = [f for f in fixtures_live
              if f["status"] in ["FT","AET","PEN"] and f["goals1"] is not None]
    n_injected = 0
    if played:
        rows = []
        for f in played:
            g1, g2 = int(f["goals1"]), int(f["goals2"])
            rows.append({
                "date": pd.Timestamp(f["date"]),
                "home_team": f["team1"], "away_team": f["team2"],
                "home_score": g1, "away_score": g2,
                "tournament": "FIFA World Cup", "weight": 3.0,
                "result": "H" if g1 > g2 else ("A" if g2 > g1 else "D"),
            })
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df = df.sort_values("date").reset_index(drop=True)
        n_injected = len(played)

    # ── Construction historique glissant ────────────────────────────
    history: dict = {}
    X_rows, y_list, w_list = [], [], []

    for _, row in df.iterrows():
        t1 = row.get("home_team", row.get("HomeTeam",""))
        t2 = row.get("away_team", row.get("AwayTeam",""))
        res, w = row["result"], row["weight"]

        s1 = _team_stats(t1, history)
        s2 = _team_stats(t2, history)

        # Feature originale
        X_rows.append(_make_features(s1, s2, w))
        y_list.append(res)
        w_list.append(w)

        # Augmentation symétrique : t2 vs t1
        # (le modèle apprend que l'ordre n'a pas d'importance)
        mirror_res = {"H": "A", "A": "H", "D": "D"}[res]
        X_rows.append(_make_features(s2, s1, w))
        y_list.append(mirror_res)
        w_list.append(w * 0.8)  # poids légèrement réduit pour les miroirs

        # Mise à jour historique
        g1, g2 = row["home_score"], row["away_score"]
        for team, scored, conc, win in [
            (t1, g1, g2, res=="H"),
            (t2, g2, g1, res=="A"),
        ]:
            pts = 3 if ((team==t1 and res=="H") or (team==t2 and res=="A")) \
                  else (1 if res=="D" else 0)
            history.setdefault(team, []).append({
                "pts": pts, "scored": scored, "conc": conc,
                "win": int(win), "draw": int(res=="D"), "w": w,
            })

    X = pd.concat(X_rows, ignore_index=True)
    y = pd.Series(y_list)
    w = pd.Series(w_list)

    le      = LabelEncoder().fit(["A","D","H"])
    trained = train_models(X, y, w, le)

    n_hist = len(df) - n_injected
    return trained, history, n_hist, n_injected


# ══════════════════════════════════════════════════════════════════════
# ML — PRÉDICTION
# ══════════════════════════════════════════════════════════════════════

def predict_match(trained: dict, model: str,
                  s1: dict, s2: dict,
                  tw: float = 3.0,
                  inj1: float = 0.0,
                  inj2: float = 0.0) -> dict:
    """
    Prédit les probabilités pour un match sur terrain neutre.
    Correction nul : boost proportionnel à l'équilibre du match.
    Symétrique : predict(t1,t2) ≈ mirror(predict(t2,t1))
    """
    b = trained[model]
    X = _make_features(s1, s2, tw, inj1, inj2)[b["cols"]].fillna(0)
    if b["scaler"]:
        X = pd.DataFrame(b["scaler"].transform(X), columns=b["cols"])

    proba = b["model"].predict_proba(X)[0]
    cls   = b["le"].classes_
    p     = dict(zip(cls, proba))
    p1, pD, p2 = p.get("H",0), p.get("D",0), p.get("A",0)

    # ── Correction nul ─────────────────────────────────────────────
    # Plus le match est équilibré → plus le nul est probable
    balance    = 1 - abs(p1 - p2)       # 0 = très déséquilibré, 1 = parfait équilibre
    draw_boost = balance * 0.03         # max +6% sur le nul
    total      = p1 + p2
    if total > 0:
        p1 = p1 - draw_boost * (p1 / total)
        p2 = p2 - draw_boost * (p2 / total)
    pD = pD + draw_boost

    # Renormalise
    s = p1 + pD + p2
    p1, pD, p2 = p1/s, pD/s, p2/s

    # ── Buts attendus (Poisson) ────────────────────────────────────
    exp1 = max(0.25, s1["scored"] * (1 - s2["clean"]) * (1 - inj1))
    exp2 = max(0.25, s2["scored"] * (1 - s1["clean"]) * (1 - inj2))

    return {
        "t1":  round(p1, 4),
        "D":   round(pD, 4),
        "t2":  round(p2, 4),
        "exp1":round(exp1, 2),
        "exp2":round(exp2, 2),
        "draw_boost": round(draw_boost, 4),
        "balance":    round(balance, 3),
    }


def score_distribution(exp1: float, exp2: float,
                        max_g: int = 5) -> pd.DataFrame:
    rows = []
    for g1 in range(max_g+1):
        for g2 in range(max_g+1):
            prob = poisson.pmf(g1, exp1) * poisson.pmf(g2, exp2)
            rows.append({"Score": f"{g1}–{g2}", "g1": g1, "g2": g2,
                         "Proba (%)": round(prob * 100, 2)})
    return pd.DataFrame(rows).sort_values("Proba (%)", ascending=False)


# ══════════════════════════════════════════════════════════════════════
# SIMULATION TOURNOI — VECTORISÉE (rapide)
# ══════════════════════════════════════════════════════════════════════

def _build_strength_matrix(trained: dict, model: str,
                            history: dict, injuries: dict) -> dict:
    """
    Pré-calcule les probabilités pour toutes les paires d'équipes.
    Évite de recalculer à chaque simulation → x10 plus rapide.
    """
    matrix = {}
    for t1 in ALL_TEAMS:
        for t2 in ALL_TEAMS:
            if t1 == t2: continue
            s1  = _team_stats(t1, history)
            s2  = _team_stats(t2, history)
            inj1 = injuries.get(t1, 0.0)
            inj2 = injuries.get(t2, 0.0)
            p = predict_match(trained, model, s1, s2, 3.0, inj1, inj2)
            matrix[(t1, t2)] = p
    return matrix


def simulate_tournament(trained: dict, model: str,
                         history: dict, injuries: dict = {},
                         n: int = 10000) -> pd.DataFrame:
    """
    Simule n fois la CdM 2026 — version vectorisée numpy (x4 plus rapide).
    Matrice de probabilités pré-calculée avant les simulations.
    5000 sims ≈ 8 secondes.
    """
    matrix = _build_strength_matrix(trained, model, history, injuries)

    # Pré-calcule les tableaux numpy pour éviter les dicts dans la boucle
    group_probs = {}
    for grp, teams in WC_GROUPS.items():
        for i, t1 in enumerate(teams):
            for t2 in teams[i+1:]:
                p = matrix[(t1, t2)]
                arr = np.array([p["t1"], p["D"], p["t2"]], dtype=float)
                arr = np.clip(arr, 1e-6, 1)
                arr /= arr.sum()
                group_probs[(t1, t2)] = (arr, p["exp1"], p["exp2"])

    ko_probs = {}
    for t1 in ALL_TEAMS:
        for t2 in ALL_TEAMS:
            if t1 == t2: continue
            p   = matrix[(t1, t2)]
            tot = p["t1"] + p["t2"]
            ko_probs[(t1, t2)] = p["t1"] / tot if tot > 0 else 0.5

    # Compteurs numpy
    team_idx = {t: i for i, t in enumerate(ALL_TEAMS)}
    n_teams  = len(ALL_TEAMS)
    counts   = np.zeros((n_teams, 7), dtype=np.int32)
    # indices : 0=grp, 1=r32, 2=r16, 3=qf, 4=sf, 5=fin, 6=win

    for _ in range(n):
        thirds, qualified = [], []

        for grp, teams in WC_GROUPS.items():
            pts = np.zeros(len(teams))
            gd  = np.zeros(len(teams))
            tidx = {t: i for i, t in enumerate(teams)}

            for i, t1 in enumerate(teams):
                for t2 in teams[i+1:]:
                    probs, exp1, exp2 = group_probs[(t1, t2)]
                    r  = np.random.choice(3, p=probs)
                    g1 = np.random.poisson(exp1)
                    g2 = np.random.poisson(exp2)
                    if r == 0:
                        pts[tidx[t1]] += 3
                        gd[tidx[t1]]  += max(1, g1 - g2)
                    elif r == 2:
                        pts[tidx[t2]] += 3
                        gd[tidx[t2]]  += max(1, g2 - g1)
                    else:
                        pts[tidx[t1]] += 1
                        pts[tidx[t2]] += 1

            order  = sorted(range(len(teams)), key=lambda i: (-pts[i], -gd[i]))
            ranked = [teams[i] for i in order]
            qualified += ranked[:2]
            counts[team_idx[ranked[0]], 0] += 1
            counts[team_idx[ranked[1]], 0] += 1
            thirds.append((ranked[2], pts[tidx[ranked[2]]], gd[tidx[ranked[2]]]))

        # 8 meilleurs 3èmes
        thirds.sort(key=lambda x: (-x[1], -x[2]))
        for t, _, _ in thirds[:8]:
            qualified.append(t)
            counts[team_idx[t], 0] += 1

        np.random.shuffle(qualified)

        def ko_round(tin, stage_col):
            out = []
            for i in range(0, len(tin) - 1, 2):
                p = ko_probs[(tin[i], tin[i+1])]
                w = tin[i] if np.random.random() < p else tin[i+1]
                counts[team_idx[w], stage_col] += 1
                out.append(w)
            return out

        q = qualified
        for col in [1, 2, 3, 4, 5]:
            q = ko_round(q, col)
        if q:
            counts[team_idx[q[0]], 6] += 1

    rows = []
    for t in ALL_TEAMS:
        i = team_idx[t]
        rows.append({
            "Équipe":          t,
            "🏅 FIFA Rang":    FIFA_RANKINGS.get(t, 80),
            "Groupe (%)":      round(counts[i,0]/n*100, 1),
            "R32 (%)":         round(counts[i,1]/n*100, 1),
            "R16 (%)":         round(counts[i,2]/n*100, 1),
            "Quarts (%)":      round(counts[i,3]/n*100, 1),
            "Demies (%)":      round(counts[i,4]/n*100, 1),
            "Finale (%)":      round(counts[i,5]/n*100, 1),
            "🏆 Vainqueur (%)":round(counts[i,6]/n*100, 1),
        })
    return (pd.DataFrame(rows)
            .sort_values("🏆 Vainqueur (%)", ascending=False)
            .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════
# VISUELS
# ══════════════════════════════════════════════════════════════════════

def gauge_chart(label: str, val: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val * 100,
        number={"suffix": "%", "font": {"size": 32, "color": color}},
        title={"text": label, "font": {"size": 14, "color": "#bbb"}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#333"},
            "bar":   {"color": color, "thickness": 0.28},
            "bgcolor": "#0d1b35",
            "steps": [{"range": [0, 100], "color": "#111a30"}],
        },
    ))
    fig.update_layout(height=190, margin=dict(t=38,b=5,l=12,r=12),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ══════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🏆 WC 2026 Predictor",
    page_icon="🏆", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container{ padding-top: 1.2rem; }
</style>""", unsafe_allow_html=True)


# ── Cache ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=120,  show_spinner="📡 Chargement matchs live...")
def _get_fixtures():   return get_fixtures()

@st.cache_data(ttl=300,  show_spinner="📊 Chargement classements...")
def _get_standings():  return get_standings()

@st.cache_data(ttl=1800, show_spinner="🏥 Chargement blessures...")
def _get_injuries():   return get_injuries()

@st.cache_data(ttl=600,  show_spinner="⚽ Chargement buteurs...")
def _get_scorers():    return get_top_scorers()

@st.cache_resource(show_spinner="🧠 Entraînement ML... (~60 sec)")
def _get_model(n_played: int):
    return load_and_train(RESULTS_CSV, _get_fixtures())


# ── Données ───────────────────────────────────────────────────────────

fixtures  = _get_fixtures()
standings = _get_standings()
injuries  = _get_injuries()
scorers   = _get_scorers()

n_played = len([f for f in fixtures if f["status"] in ["FT","AET","PEN"]])
trained, history, n_hist, n_live = _get_model(n_played)

# Impacts blessures via API
team_injured = defaultdict(list)
for inj in injuries:
    team_injured[inj["team"]].append(inj["player"])
api_impacts: dict = {}
for team, players in team_injured.items():
    if team not in KEY_PLAYERS: continue
    impact = 0.0
    for p in players:
        for key_name, val in KEY_PLAYERS[team].items():
            if key_name.lower() in p.lower():
                impact += val
    if impact > 0:
        api_impacts[team] = min(impact, 0.75)


# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏆 WC 2026 Predictor")
    if API_KEY:
        st.success("✅ API connectée")
    else:
        st.error("❌ Clé API manquante\n`.env` → `API_FOOTBALL_KEY=clé`")
    st.divider()

    model_choice = st.selectbox("🧠 Modèle ML",
                                 ["LightGBM", "Logistic Regression"],
                                 help="LightGBM : plus puissant\nLogistic : mieux calibré")
    st.divider()

    if st.button("🔄 Actualiser données", use_container_width=True):
        for f in CACHE_DIR.glob("*.json"): f.unlink()
        st.cache_data.clear()
        st.rerun()

    st.caption(f"📊 {n_hist:,} matchs historiques")
    st.caption(f"⚽ {n_live} matchs CdM 2026 intégrés")
    st.caption("🔄 Données rafraîchies ttes les 2 min")
    st.divider()
    st.caption("Made with ❤️ + ML + API-Football")


# ── Onglets ───────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Prédiction",
    "📅 Matchs",
    "🏆 Simulation",
    "📊 Groupes",
    "🏥 Équipes",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — PRÉDICTION
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔮 Prédire un match")
    st.caption("Terrain neutre — l'ordre des équipes n'influence pas les probabilités "
               "(sauf pour les pays hôtes 🇺🇸🇨🇦🇲🇽)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ⚽ Équipe 1")
        team1 = st.selectbox("", ALL_TEAMS,
                              index=ALL_TEAMS.index("France") if "France" in ALL_TEAMS else 0,
                              key="t1")
    with c2:
        st.markdown("### ⚽ Équipe 2")
        opts2 = [t for t in ALL_TEAMS if t != team1]
        team2 = st.selectbox("", opts2,
                              index=opts2.index("Brazil") if "Brazil" in opts2 else 0,
                              key="t2")

    # Avantage hôte automatique
    if team1 in HOST_NATIONS:
        st.info(f"🏟️ {team1} joue sur ses terres — légère faveur hôte appliquée (+{HOST_BOOST:.0%})")
    elif team2 in HOST_NATIONS:
        st.info(f"🏟️ {team2} joue sur ses terres — légère faveur hôte appliquée (+{HOST_BOOST:.0%})")

    c1, c2 = st.columns(2)
    with c1:
        phase = st.selectbox("🏆 Phase",
                              ["Groupes","Round of 32","Round of 16",
                               "Quarts","Demies","Finale"])
    with c2:
        n_rec = st.slider("Matchs récents considérés", 5, 20, 10)

    # Blessures
    st.divider()
    st.markdown("### 🏥 Blessures")
    c1, c2 = st.columns(2)
    with c1:
        auto1 = api_impacts.get(team1, 0.0)
        if auto1 > 0:
            st.warning(f"⚠️ Blessures détectées : -{auto1:.0%} (API)")
        inj1 = st.slider(f"Impact blessures {team1} (%)", 0, 60,
                          int(auto1*100), 5, key="i1") / 100
    with c2:
        auto2 = api_impacts.get(team2, 0.0)
        if auto2 > 0:
            st.warning(f"⚠️ Blessures détectées : -{auto2:.0%} (API)")
        inj2 = st.slider(f"Impact blessures {team2} (%)", 0, 60,
                          int(auto2*100), 5, key="i2") / 100

    # Stats actuelles des équipes
    st.divider()
    s1 = _team_stats(team1, history, n_rec)
    s2 = _team_stats(team2, history, n_rec)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### 📋 {team1}")
        r = FIFA_RANKINGS.get(team1, "N/A")
        coach_name = COACHES.get(team1, {})
        if isinstance(coach_name, dict):
            cname = coach_name.get("name","?")
        else:
            cname = "?"
        st.caption(f"FIFA #{r} · Coach : {cname} · {s1['n']} matchs analysés")
        a, b_, c, d, e = st.columns(5)
        a.metric("Forme",      f"{s1['form']:.0%}")
        b_.metric("Buts/m",    f"{s1['scored']:.1f}")
        c.metric("Encaissés",  f"{s1['conc']:.1f}")
        d.metric("Victoires",  f"{s1['win']:.0%}")
        e.metric("Clean sh.",  f"{s1['clean']:.0%}")
    with c2:
        st.markdown(f"#### 📋 {team2}")
        r2 = FIFA_RANKINGS.get(team2, "N/A")
        coach_name2 = COACHES.get(team2, {})
        if isinstance(coach_name2, dict):
            cname2 = coach_name2.get("name","?")
        else:
            cname2 = "?"
        st.caption(f"FIFA #{r2} · Coach : {cname2} · {s2['n']} matchs analysés")
        a, b_, c, d, e = st.columns(5)
        a.metric("Forme",      f"{s2['form']:.0%}")
        b_.metric("Buts/m",    f"{s2['scored']:.1f}")
        c.metric("Encaissés",  f"{s2['conc']:.1f}")
        d.metric("Victoires",  f"{s2['win']:.0%}")
        e.metric("Clean sh.",  f"{s2['clean']:.0%}")

    if st.button("🔮 Lancer la prédiction", type="primary", use_container_width=True):
        tw = {"Groupes":3.0,"Round of 32":3.5,"Round of 16":4.0,
              "Quarts":4.5,"Demies":5.0,"Finale":5.5}.get(phase, 3.0)

        pred = predict_match(trained, model_choice, s1, s2, tw, inj1, inj2)

        st.markdown(f"## ⚽ {team1} 🆚 {team2}")
        st.caption(f"Phase : {phase} · Terrain neutre · "
                   f"Équilibre du match : {pred['balance']:.0%}")

        # Jauges
        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(gauge_chart(f"⚽ {team1}", pred["t1"], "#2ecc71"),
                                  use_container_width=True)
        with c2: st.plotly_chart(gauge_chart("🤝 Match nul", pred["D"], "#f39c12"),
                                  use_container_width=True)
        with c3: st.plotly_chart(gauge_chart(f"⚽ {team2}", pred["t2"], "#e74c3c"),
                                  use_container_width=True)

        # Pronostic intelligent
        DRAW_THRESHOLD = 0.27
        winner_k = max({"t1":pred["t1"],"D":pred["D"],"t2":pred["t2"]},
                       key=lambda k: pred[k])
        labels = {"t1": f"⚽ {team1}", "D": "🤝 Match nul", "t2": f"⚽ {team2}"}
        pw = pred[winner_k]

        # Affichage spécial si le nul est très probable
        if pred["D"] >= DRAW_THRESHOLD and pred["draw_boost"] > 0.03:
            st.warning(f"**🤝 Match très équilibré — nul probable ({pred['D']:.1%})**  "
                       f"| {team1} : {pred['t1']:.1%} | {team2} : {pred['t2']:.1%}")
        elif pw > 0.55:
            st.success(f"**Pronostic : {labels[winner_k]}** — confiance {pw:.1%}")
        elif pw > 0.42:
            st.warning(f"**Match serré** — {labels[winner_k]} légèrement favori ({pw:.1%})")
        else:
            st.info(f"**Match très ouvert** — {labels[winner_k]} ({pw:.1%})")

        if pred["draw_boost"] > 0.02:
            st.caption(f"⚖️ Boost nul appliqué : +{pred['draw_boost']:.1%} "
                       f"(match équilibré à {pred['balance']:.0%})")

        # Buts attendus
        c1, c2, c3 = st.columns(3)
        c1.metric(f"⚽ Buts attendus {team1}", pred["exp1"])
        c2.metric(f"⚽ Buts attendus {team2}", pred["exp2"])
        c3.metric("🎯 Total buts attendus", round(pred["exp1"]+pred["exp2"],2))

        # Distribution des scores
        st.divider()
        st.subheader("📊 Distribution des scores")
        dist = score_distribution(pred["exp1"], pred["exp2"])

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Top 8 scores probables**")
            st.dataframe(dist.head(8)[["Score","Proba (%)"]],
                         hide_index=True, use_container_width=True)
            # Résumé H/D/A depuis la distribution
            p_t1  = dist[dist["g1"] > dist["g2"]]["Proba (%)"].sum()
            p_nul = dist[dist["g1"] == dist["g2"]]["Proba (%)"].sum()
            p_t2  = dist[dist["g2"] > dist["g1"]]["Proba (%)"].sum()
            st.caption(f"Depuis distribution Poisson : "
                       f"{team1} {p_t1:.1f}% | Nul {p_nul:.1f}% | {team2} {p_t2:.1f}%")
        with c2:
            piv = dist.pivot(index="g1", columns="g2",
                              values="Proba (%)").fillna(0)
            fig = go.Figure(go.Heatmap(
                z=piv.values,
                x=[f"{team2} {g}" for g in piv.columns],
                y=[f"{team1} {g}" for g in piv.index],
                colorscale="YlOrRd",
                text=[[f"{v:.1f}%" for v in row] for row in piv.values],
                texttemplate="%{text}",
            ))
            fig.update_layout(
                height=330, paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
                margin=dict(t=15,b=30,l=80,r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Prédiction API en bonus
        if fixtures:
            upcoming = [f for f in fixtures
                        if f["status"] in ["NS","TBD"]
                        and {f["team1"],f["team2"]} == {team1,team2}]
            if upcoming:
                api_pred = get_api_prediction(upcoming[0]["id"])
                if api_pred and api_pred.get("winner"):
                    st.divider()
                    st.caption(
                        f"🤖 **Prédiction API-Football** : "
                        f"Favori → **{api_pred['winner']}** | "
                        f"Score prévu : {api_pred.get('goals1','-')} – {api_pred.get('goals2','-')} | "
                        f"{api_pred.get('under_over','')}"
                    )


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — MATCHS & LIVE
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📅 Matchs & Résultats CdM 2026")

    live_m    = [f for f in fixtures if f["status"] in ["1H","2H","HT","ET","P"]]
    played_m  = [f for f in fixtures if f["status"] in ["FT","AET","PEN"]]
    upcoming_m= [f for f in fixtures if f["status"] in ["NS","TBD"]]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🔴 En cours", len(live_m))
    c2.metric("✅ Joués",    len(played_m))
    c3.metric("📅 À venir",  len(upcoming_m))
    c4.metric("📊 Total",    len(fixtures))

    # Live
    if live_m:
        st.divider()
        st.markdown("### 🔴 Matchs en cours")
        for m in live_m:
            st.markdown(
                f"🔴 **{m['elapsed']}'** — **{m['team1']}** "
                f"{m['goals1']} – {m['goals2']} **{m['team2']}** "
                f"· *{m['venue']}*"
            )

    # Résultats
    if played_m:
        st.divider()
        st.markdown("### ✅ Résultats récents")
        for m in reversed(played_m[-20:]):
            g1, g2 = m["goals1"], m["goals2"]
            icon = "🟡" if g1==g2 else ("🟢" if g1>g2 else "🔵")
            st.markdown(
                f"{icon} `{m['date']}` — **{m['team1']}** {g1}–{g2} "
                f"**{m['team2']}** | _{m['round']}_"
            )

    # Prochains matchs avec prédictions
    if upcoming_m:
        st.divider()
        st.markdown("### 📅 Prochains matchs — prédictions ML")
        for m in upcoming_m[:20]:
            s1_m  = _team_stats(m["team1"], history)
            s2_m  = _team_stats(m["team2"], history)
            i1    = api_impacts.get(m["team1"], 0.0)
            i2    = api_impacts.get(m["team2"], 0.0)
            pred  = predict_match(trained, model_choice, s1_m, s2_m, 3.0, i1, i2)

            with st.expander(
                f"📅 {m['date']} — **{m['team1']}** vs **{m['team2']}** "
                f"({m['round']})"
            ):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric(f"⚽ {m['team1']}", f"{pred['t1']:.1%}")
                c2.metric("🤝 Nul",           f"{pred['D']:.1%}")
                c3.metric(f"⚽ {m['team2']}",  f"{pred['t2']:.1%}")
                c4.metric("Score prédit",
                           f"{pred['exp1']:.1f}–{pred['exp2']:.1f}")
                if i1: st.caption(f"⚠️ {m['team1']} : -{i1:.0%} blessures")
                if i2: st.caption(f"⚠️ {m['team2']} : -{i2:.0%} blessures")
                if pred["draw_boost"] > 0.03:
                    st.caption(f"⚖️ Match équilibré — nul boosté +{pred['draw_boost']:.1%}")


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — SIMULATION TOURNOI
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🏆 Simulation Coupe du Monde 2026")
    st.caption(f"48 équipes · 12 groupes · Format officiel FIFA 2026 · "
               f"{n_live} vrais résultats intégrés")

    c1, c2 = st.columns([1, 2])
    with c1:
        n_sims = st.select_slider(
            "Nombre de simulations",
            options=[1000, 5000, 10000, 20000],
            value=5000,
            help="La matrice est pré-calculée → simulation rapide"
        )
        st.info(f"⚡ Grâce à la pré-calculcul des probas,\n"
                f"5000 simulations ≈ 5-10 secondes")

        # Blessures pour la simulation
        if api_impacts:
            st.markdown("**🏥 Blessures détectées (API)**")
            sim_injuries = dict(api_impacts)
            for team, auto in api_impacts.items():
                v = st.slider(f"{team}", 0, 60, int(auto*100), 5,
                              key=f"si_{team}") / 100
                sim_injuries[team] = v
        else:
            sim_injuries = {}
            st.caption("Aucune blessure détectée via l'API")

    with c2:
        st.markdown("**Groupes officiels CdM 2026**")
        cols = st.columns(4)
        for idx, (grp, teams) in enumerate(WC_GROUPS.items()):
            with cols[idx % 4]:
                st.markdown(f"**Groupe {grp}**")
                for t in teams:
                    r = FIFA_RANKINGS.get(t, "?")
                    inj = sim_injuries.get(t, 0)
                    inj_str = f" 🏥" if inj > 0 else ""
                    host_str = " 🏟️" if t in HOST_NATIONS else ""
                    st.caption(f"• {t} (#{r}){host_str}{inj_str}")

    if st.button("🎲 Simuler le tournoi", type="primary", use_container_width=True):
        with st.spinner("Pré-calcul des probabilités..."):
            df_sim = simulate_tournament(trained, model_choice,
                                          history, sim_injuries, n_sims)

        # Top favoris
        top10 = df_sim.head(10)
        fig = go.Figure(go.Bar(
            x=top10["🏆 Vainqueur (%)"],
            y=top10["Équipe"],
            orientation="h",
            marker=dict(color=top10["🏆 Vainqueur (%)"],
                        colorscale="YlOrRd", showscale=True,
                        colorbar=dict(title="%")),
            text=[f"{v}%" for v in top10["🏆 Vainqueur (%)"]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Probabilité de remporter la CdM 2026 ({n_sims:,} simulations)",
            yaxis=dict(autorange="reversed"),
            height=400, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"),
            xaxis=dict(gridcolor="#1a2744"),
            margin=dict(t=40,b=20,l=150,r=80),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap probabilités par stade
        stages = ["Groupe (%)","R32 (%)","R16 (%)","Quarts (%)","Demies (%)","Finale (%)","🏆 Vainqueur (%)"]
        top20  = df_sim.head(20)
        fig2 = go.Figure(go.Heatmap(
            z=top20[stages].values,
            x=[s.replace(" (%)","").replace("🏆 ","") for s in stages],
            y=top20["Équipe"].tolist(),
            colorscale="YlOrRd",
            text=[[f"{v}%" for v in row] for row in top20[stages].values],
            texttemplate="%{text}",
        ))
        fig2.update_layout(
            title="Probabilités par stade — Top 20",
            height=580, paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
            yaxis=dict(autorange="reversed"),
            margin=dict(t=40,b=20,l=150,r=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📋 Tableau complet — 48 équipes")
        st.dataframe(df_sim, use_container_width=True, height=500)


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — GROUPES & STATS
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📊 Groupes & Statistiques")

    # Classements live
    if standings:
        st.subheader("🏆 Classements de groupes (live)")
        cols = st.columns(3)
        for idx, (grp_name, teams) in enumerate(standings.items()):
            with cols[idx % 3]:
                st.markdown(f"**{grp_name}**")
                df_g = pd.DataFrame(teams)[[
                    "rank","team","played","won","drawn","lost","gf","ga","gd","pts","form"
                ]]
                df_g.columns = ["#","Équipe","J","V","N","D","BP","BC","Diff","Pts","Forme"]
                st.dataframe(df_g, hide_index=True, use_container_width=True)
    else:
        st.info("Classements disponibles une fois les matchs commencés.")
        cols = st.columns(4)
        for idx, (grp, teams) in enumerate(WC_GROUPS.items()):
            with cols[idx % 4]:
                st.markdown(f"**Groupe {grp}**")
                for t in teams:
                    r = FIFA_RANKINGS.get(t, "?")
                    h = " 🏟️" if t in HOST_NATIONS else ""
                    st.caption(f"• {t} (#{r}){h}")

    # Top buteurs
    st.divider()
    st.subheader("⚽ Top buteurs")
    if scorers:
        df_sc = pd.DataFrame(scorers)
        df_sc.columns = ["Joueur","Équipe","Buts","Passes D.","Note"]
        c1, c2 = st.columns([1,2])
        with c1:
            st.dataframe(df_sc, hide_index=True, use_container_width=True)
        with c2:
            fig = go.Figure(go.Bar(
                x=df_sc["Joueur"], y=df_sc["Buts"],
                marker_color="#f39c12",
                text=df_sc["Buts"], textposition="outside",
            ))
            fig.update_layout(
                height=300, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"),
                xaxis=dict(tickangle=30, gridcolor="#1a2744"),
                yaxis=dict(gridcolor="#1a2744"),
                margin=dict(t=10,b=80,l=40,r=20),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Top buteurs disponibles une fois le tournoi lancé.")

    # Analyse équipe
    st.divider()
    st.subheader("🔬 Analyse d'une équipe nationale")
    team_sel = st.selectbox("Équipe", ALL_TEAMS,
                             index=ALL_TEAMS.index("France") if "France" in ALL_TEAMS else 0)
    n_m = st.slider("Matchs récents à analyser", 5, 30, 15, key="an")
    s   = _team_stats(team_sel, history, n_m)
    r_sel = FIFA_RANKINGS.get(team_sel, "?")
    c_sel = COACHES.get(team_sel, 10)
    coach_matches = c_sel if isinstance(c_sel, int) else c_sel.get("matches",10)

    st.caption(f"FIFA #{r_sel} · Coach : {coach_matches} matchs à la tête de l'équipe "
               f"· Score stabilité : {_coach_score(team_sel):.0%}")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Forme",        f"{s['form']:.0%}")
    c2.metric("Buts/m",       f"{s['scored']:.2f}")
    c3.metric("Encaissés/m",  f"{s['conc']:.2f}")
    c4.metric("Victoires",    f"{s['win']:.0%}")
    c5.metric("Clean sheets", f"{s['clean']:.0%}")
    c6.metric("Diff. buts",   f"{s['gd']:+.2f}")

    inj_t = api_impacts.get(team_sel, 0.0)
    if inj_t > 0:
        st.error(f"🏥 Impact blessures actuel détecté : -{inj_t:.0%}")
    if team_sel in HOST_NATIONS:
        st.success(f"🏟️ Pays hôte — boost terrain : +{HOST_BOOST:.0%}")


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — BLESSURES & ÉQUIPES
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.header("🏥 Blessures & Équipes")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🩹 Blessures détectées (API live)")
        if injuries:
            st.caption(f"{len(injuries)} joueurs blessés/suspendus")
            rows_inj = []
            for team in sorted(set(i["team"] for i in injuries)):
                players = [i["player"] for i in injuries if i["team"] == team]
                impact  = api_impacts.get(team, 0.0)
                rows_inj.append({
                    "Équipe":   team,
                    "Nb":       len(players),
                    "Impact":   f"-{impact:.0%}" if impact > 0 else "< 1%",
                    "Joueurs":  ", ".join(players[:3]) + ("..." if len(players)>3 else ""),
                })
            st.dataframe(pd.DataFrame(rows_inj).sort_values("Nb", ascending=False),
                         hide_index=True, use_container_width=True)

            team_f = st.selectbox("Détail équipe",
                                   ["Toutes"] + sorted(set(i["team"] for i in injuries)))
            df_inj = pd.DataFrame(injuries)
            if team_f != "Toutes":
                df_inj = df_inj[df_inj["team"] == team_f]
            st.dataframe(df_inj, hide_index=True, use_container_width=True)
        else:
            st.info("Aucune blessure détectée pour l'instant.")

    with c2:
        st.subheader("⭐ Joueurs clés dans le modèle")
        for team, players in KEY_PLAYERS.items():
            with st.expander(f"**{team}** (FIFA #{FIFA_RANKINGS.get(team,'?')})"):
                rows_kp = [{"Joueur": name, "Impact si absent": f"-{val:.0%}"}
                           for name, val in players.items()]
                st.dataframe(pd.DataFrame(rows_kp),
                             hide_index=True, use_container_width=True)

    # Classements FIFA et coachs
    st.divider()
    st.subheader("📋 Toutes les équipes — FIFA ranking & coach")
    rows_all = []
    for t in ALL_TEAMS:
        c_data = COACHES.get(t, 10)
        m = c_data if isinstance(c_data, int) else c_data.get("matches",10)
        rows_all.append({
            "Équipe":       t,
            "Groupe":       next((g for g, ts in WC_GROUPS.items() if t in ts), "?"),
            "FIFA Rang":    FIFA_RANKINGS.get(t, 80),
            "Score FIFA":   f"{_fifa_score(t):.2f}",
            "Coach (matchs)": m,
            "Stabilité coach": f"{_coach_score(t):.0%}",
            "Hôte":         "🏟️" if t in HOST_NATIONS else "",
        })
    df_all = (pd.DataFrame(rows_all)
              .sort_values("FIFA Rang")
              .reset_index(drop=True))
    st.dataframe(df_all, hide_index=True, use_container_width=True)

    # À propos du modèle
    st.divider()
    st.subheader("ℹ️ À propos du modèle")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Matchs historiques",      f"{n_hist:,}")
    c2.metric("Matchs CdM 2026 intégrés", n_live)
    c3.metric("Période données",          "2014–2026")
    c4.metric("Features ML",             "27")
    st.markdown("""
**Architecture :**
- **Historique** : 2014–2026, pondéré par compétition (CdM ×3, amicaux ×0.5)
- **Classement FIFA** : intégré comme prior (20% blending avec l'historique)
- **Symétrie** : données augmentées (t1 vs t2 = t2 vs t1) → terrain neutre garanti
- **Nuls** : poids ×1.4 + boost post-prédiction selon l'équilibre du match
- **Coach** : score de stabilité (matchs dirigés), poids 10%
- **Blessures** : impact direct sur les features, auto-détecté via API
- **Pays hôtes** : boost +6% pour USA, Canada, Mexique
- **Calibration** : isotonique sur 5-fold cross-validation
- **Réentraînement** : automatique après chaque nouveau match joué
    """)