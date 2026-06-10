"""
🏆 World Cup 2026 Predictor — Version Complète
===============================================
Lance avec : streamlit run wc2026.py
Clé API    : fichier .env → API_FOOTBALL_KEY=ta_clé
Données    : data/results.csv (Kaggle)
"""

import os, json, time, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
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

API_KEY     = os.getenv("API_FOOTBALL_KEY", "")
BASE_URL    = "https://v3.football.api-sports.io"
WC_LEAGUE   = 1
WC_SEASON   = 2026
RESULTS_CSV = "data/results.csv"
CACHE_DIR   = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

HOST_NATIONS = {"United States", "Canada", "Mexico"}
HOST_BOOST   = 0.06

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

# Appariements officiels Round of 32 (source FIFA)
# Format : (slot1, slot2) où slot = "W_X", "R_X", "T3_ABCD"
R32_SLOTS = [
    ("W_A", "T3_CEFHI"),   ("W_C", "R_F"),
    ("W_F", "R_C"),        ("W_E", "T3_ABDFK"),
    ("W_G", "T3_AEHIJ"),   ("W_I", "T3_CDFGH"),
    ("R_E", "R_I"),        ("W_B", "T3_EFGIJ"),
    ("W_L", "T3_EHIJK"),   ("W_D", "T3_BEFIJ"),
    ("R_K", "R_L"),        ("W_K", "T3_ACFHK"),
    ("W_H", "R_J"),        ("W_J", "R_H"),
    ("R_G", "R_A"),        ("R_B", "R_D"),
]

COMP_WEIGHTS = {
    "FIFA World Cup": 3.0, "Confederations Cup": 2.5,
    "UEFA Euro": 2.5, "Copa América": 2.5,
    "African Cup of Nations": 2.5, "AFC Asian Cup": 2.0,
    "Gold Cup": 2.0, "FIFA World Cup qualification": 2.0,
    "UEFA Euro qualification": 1.8, "UEFA Nations League": 1.8,
    "African Cup of Nations qualification": 1.5,
    "AFC Asian Cup qualification": 1.5,
    "CONCACAF Nations League": 1.5, "Friendly": 0.5,
}

FIFA_RANKINGS = {
    "France":1,"Spain":2,"Argentina":3,"England":4,"Portugal":5,"Brazil":6,
    "Belgium":7,"Netherlands":8,"Germany":9,"Colombia":10,"Morocco":11,
    "Uruguay":12,"United States":14,"Mexico":15,"Croatia":16,"Japan":17,
    "Senegal":18,"Switzerland":19,"Ecuador":20,"Turkey":21,"Austria":22,
    "South Korea":23,"Iran":25,"Australia":26,"Norway":27,"Czech Republic":28,
    "Sweden":33,"Canada":34,"Algeria":35,"Tunisia":36,"Egypt":37,
    "Saudi Arabia":38,"Scotland":39,"Paraguay":40,"Ivory Coast":41,"Qatar":42,
    "Ghana":43,"South Africa":45,"Iraq":46,"Haiti":47,"Panama":48,
    "DR Congo":49,"Jordan":50,"Bosnia and Herzegovina":52,
    "Uzbekistan":53,"Cape Verde":54,"New Zealand":85,"Curaçao":83,
}

COACHES = {
    "France":150,"Argentina":90,"Senegal":100,"Croatia":90,"Japan":80,
    "Scotland":75,"South Africa":55,"Norway":55,"Switzerland":45,
    "Cape Verde":45,"Portugal":45,"Mexico":45,"Morocco":40,"Uruguay":40,
    "Curaçao":40,"Austria":40,"Colombia":40,"Panama":40,"Germany":30,
    "Netherlands":30,"Belgium":28,"Haiti":30,"Iran":30,"Algeria":30,
    "Paraguay":35,"Uzbekistan":35,"Sweden":40,"Spain":35,"Turkey":35,
    "Ecuador":25,"Jordan":25,"Ivory Coast":20,"Australia":20,
    "South Korea":20,"Czech Republic":18,"Egypt":15,"Saudi Arabia":10,
    "Canada":22,"England":20,"United States":18,"Ghana":18,"Iraq":15,
    "Qatar":15,"DR Congo":20,"Brazil":8,"Tunisia":8,
    "Bosnia and Herzegovina":12,"New Zealand":20,
}

KEY_PLAYERS = {
    "France":    {"Mbappé":0.25,"Griezmann":0.15,"Tchouaméni":0.12},
    "Brazil":    {"Vinicius":0.25,"Rodrygo":0.15,"Alisson":0.12},
    "Argentina": {"Messi":0.35,"Álvarez":0.18,"De Paul":0.12},
    "England":   {"Bellingham":0.25,"Kane":0.22,"Saka":0.15},
    "Spain":     {"Yamal":0.22,"Pedri":0.20,"Rodri":0.18},
    "Germany":   {"Musiala":0.22,"Wirtz":0.20,"Havertz":0.15},
    "Portugal":  {"Ronaldo":0.28,"Bruno Fernandes":0.20,"Bernardo":0.15},
    "Netherlands":{"Gakpo":0.20,"De Jong":0.18,"Dumfries":0.10},
    "Belgium":   {"De Bruyne":0.28,"Lukaku":0.20,"Doku":0.15},
    "Colombia":  {"James":0.22,"Díaz":0.20,"Arias":0.10},
    "Morocco":   {"En-Nesyri":0.20,"Hakimi":0.18,"Ziyech":0.15},
    "Uruguay":   {"Valverde":0.22,"Núñez":0.20,"Bentancur":0.12},
}

FLAGS = {
    "France":"🇫🇷","Spain":"🇪🇸","Argentina":"🇦🇷","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Portugal":"🇵🇹","Brazil":"🇧🇷","Belgium":"🇧🇪","Netherlands":"🇳🇱",
    "Germany":"🇩🇪","Colombia":"🇨🇴","Morocco":"🇲🇦","Uruguay":"🇺🇾",
    "United States":"🇺🇸","Mexico":"🇲🇽","Croatia":"🇭🇷","Japan":"🇯🇵",
    "Senegal":"🇸🇳","Switzerland":"🇨🇭","Ecuador":"🇪🇨","Turkey":"🇹🇷",
    "Austria":"🇦🇹","South Korea":"🇰🇷","Iran":"🇮🇷","Australia":"🇦🇺",
    "Norway":"🇳🇴","Czech Republic":"🇨🇿","Sweden":"🇸🇪","Canada":"🇨🇦",
    "Algeria":"🇩🇿","Tunisia":"🇹🇳","Egypt":"🇪🇬","Saudi Arabia":"🇸🇦",
    "Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Paraguay":"🇵🇾","Ivory Coast":"🇨🇮",
    "Qatar":"🇶🇦","Ghana":"🇬🇭","South Africa":"🇿🇦","Iraq":"🇮🇶",
    "Haiti":"🇭🇹","Panama":"🇵🇦","DR Congo":"🇨🇩","Jordan":"🇯🇴",
    "Bosnia and Herzegovina":"🇧🇦","Uzbekistan":"🇺🇿","Cape Verde":"🇨🇻",
    "New Zealand":"🇳🇿","Curaçao":"🇨🇼",
}
def F(t): return FLAGS.get(t,"🏳️")
def FT(t): return f"{F(t)} {t}"

# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════

def _norm(n):
    return {"Korea Republic":"South Korea","IR Iran":"Iran","USA":"United States",
            "Czechia":"Czech Republic","Türkiye":"Turkey","Côte d'Ivoire":"Ivory Coast",
            "Bosnia":"Bosnia and Herzegovina","Cabo Verde":"Cape Verde","Congo DR":"DR Congo",
            }.get(n,n)

def _fifa(t): return max(0.0,(100-FIFA_RANKINGS.get(t,75))/99)
def _coach(t): return min(COACHES.get(t,10)/80,1.0)
def _host(t): return HOST_BOOST if t in HOST_NATIONS else 0.0

# ══════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════

def api_get(ep, params={}, ttl=300):
    if not API_KEY: return {}
    k = ep.replace("/","_")+"_".join(f"{a}{b}" for a,b in sorted(params.items()))
    f = CACHE_DIR/f"{k}.json"
    if f.exists() and (time.time()-f.stat().st_mtime)<ttl:
        return json.loads(f.read_text())
    try:
        r = requests.get(f"{BASE_URL}/{ep}",headers={"x-apisports-key":API_KEY},
                         params=params,timeout=10)
        d = r.json(); f.write_text(json.dumps(d)); return d
    except: return {}

def get_fixtures():
    data=api_get("fixtures",{"league":WC_LEAGUE,"season":WC_SEASON},ttl=120)
    out=[]
    for f in data.get("response",[]):
        fix,teams,goals=f.get("fixture",{}),f.get("teams",{}),f.get("goals",{})
        out.append({"id":fix.get("id"),"date":fix.get("date","")[:10],
                    "status":fix.get("status",{}).get("short",""),
                    "elapsed":fix.get("status",{}).get("elapsed"),
                    "team1":_norm(teams.get("home",{}).get("name","")),
                    "team2":_norm(teams.get("away",{}).get("name","")),
                    "goals1":goals.get("home"),"goals2":goals.get("away"),
                    "round":f.get("league",{}).get("round",""),
                    "venue":fix.get("venue",{}).get("name","")})
    return out

def get_standings():
    data=api_get("standings",{"league":WC_LEAGUE,"season":WC_SEASON},ttl=300)
    groups={}
    for lg in data.get("response",[]):
        for grp in lg.get("league",{}).get("standings",[]):
            if not grp: continue
            name=grp[0].get("group","")
            groups[name]=[{"rank":t["rank"],"team":_norm(t.get("team",{}).get("name","")),
                           "played":t.get("all",{}).get("played",0),
                           "won":t.get("all",{}).get("win",0),
                           "drawn":t.get("all",{}).get("draw",0),
                           "lost":t.get("all",{}).get("lose",0),
                           "gf":t.get("all",{}).get("goals",{}).get("for",0),
                           "ga":t.get("all",{}).get("goals",{}).get("against",0),
                           "gd":t.get("goalsDiff",0),"pts":t.get("points",0),
                           "form":t.get("form","")} for t in grp]
    return groups

def get_injuries():
    data=api_get("injuries",{"league":WC_LEAGUE,"season":WC_SEASON},ttl=1800)
    return [{"player":i.get("player",{}).get("name",""),
             "team":_norm(i.get("team",{}).get("name","")),
             "type":i.get("player",{}).get("type",""),
             "reason":i.get("player",{}).get("reason","")}
            for i in data.get("response",[])]

def get_top_scorers():
    data=api_get("players/topscorers",{"league":WC_LEAGUE,"season":WC_SEASON},ttl=600)
    out=[]
    for item in data.get("response",[])[:15]:
        p=item.get("player",{}); s=(item.get("statistics") or [{}])[0]
        out.append({"player":p.get("name",""),"team":_norm(s.get("team",{}).get("name","")),
                    "goals":(s.get("goals") or {}).get("total") or 0,
                    "assists":(s.get("goals") or {}).get("assists") or 0,
                    "rating":float((s.get("games") or {}).get("rating") or 0)})
    return out

def get_api_pred(fid):
    data=api_get("predictions",{"fixture":fid},ttl=3600)
    resp=data.get("response",[])
    if not resp: return {}
    pred=resp[0].get("predictions",{})
    return {"winner":pred.get("winner",{}).get("name",""),
            "goals1":pred.get("goals",{}).get("home",""),
            "goals2":pred.get("goals",{}).get("away","")}

# ══════════════════════════════════════════════════════════════════════
# ML
# ══════════════════════════════════════════════════════════════════════

def _stats(team, history, n=10):
    g=history.get(team,[])[-n:]
    fi=_fifa(team); co=_coach(team); ho=_host(team)
    if not g:
        s=0.7+fi*1.5; c=max(0.3,1.7-fi*1.3)
        return dict(form=0.25+fi*0.50,scored=s,conc=c,win=0.20+fi*0.50,
                    clean=0.08+fi*0.37,draw=0.28-fi*0.08,gd=s-c,
                    fifa=fi,coach=co,host=ho,n=0)
    qws=[gg.get("qw",gg["w"]) for gg in g]; qwt=sum(qws) or 1
    wt=sum(gg["w"] for gg in g) or 1
    hist_form=sum(gg["pts"]*qw for gg,qw in zip(g,qws))/qwt/3
    return dict(
        form=hist_form*0.80+fi*0.20,
        scored=sum(gg["scored"]*qw for gg,qw in zip(g,qws))/qwt,
        conc=sum(gg["conc"]*qw for gg,qw in zip(g,qws))/qwt,
        win=sum(gg["win"]*qw for gg,qw in zip(g,qws))/qwt,
        clean=sum((gg["conc"]==0)*qw for gg,qw in zip(g,qws))/qwt,
        draw=sum(gg["draw"]*gg["w"] for gg in g)/wt,
        gd=sum((gg["scored"]-gg["conc"])*qw for gg,qw in zip(g,qws))/qwt,
        fifa=fi,coach=co,host=ho,n=len(g))

def _feat(s1,s2,tw=3.0,i1=0.0,i2=0.0):
    h,a=s1.copy(),s2.copy()
    if i1>0:
        f=1-i1
        for k in ["form","scored","win","gd","clean"]: h[k]*=f
    if i2>0:
        f=1-i2
        for k in ["form","scored","win","gd","clean"]: a[k]*=f
    return pd.DataFrame([{
        "t1_form":h["form"],"t2_form":a["form"],
        "t1_scored":h["scored"],"t2_scored":a["scored"],
        "t1_conc":h["conc"],"t2_conc":a["conc"],
        "t1_win":h["win"],"t2_win":a["win"],
        "t1_clean":h["clean"],"t2_clean":a["clean"],
        "t1_draw":h["draw"],"t2_draw":a["draw"],
        "t1_gd":h["gd"],"t2_gd":a["gd"],
        "t1_fifa":h["fifa"],"t2_fifa":a["fifa"],
        "t1_coach":h["coach"]*0.10,"t2_coach":a["coach"]*0.10,
        "form_diff":h["form"]-a["form"],"scored_diff":h["scored"]-a["scored"],
        "conc_diff":h["conc"]-a["conc"],"win_diff":h["win"]-a["win"],
        "gd_diff":h["gd"]-a["gd"],"clean_diff":h["clean"]-a["clean"],
        "fifa_diff":h["fifa"]-a["fifa"],"coach_diff":(h["coach"]-a["coach"])*0.10,
        "host_diff":h["host"]-a["host"],"tw":tw,
        "draw_tendency":(h["draw"]+a["draw"])/2,
    }])

def _train(X,y,w,le):
    ye=le.transform(y); dc=le.transform(["D"])[0]
    db=np.where(ye==dc,1.15,1.0); wb=w.values*db
    trained={}
    lgbm=CalibratedClassifierCV(lgb.LGBMClassifier(
        n_estimators=300,learning_rate=0.04,num_leaves=24,min_child_samples=25,
        subsample=0.85,colsample_bytree=0.85,class_weight={0:1.0,1:1.15,2:1.0},
        random_state=42,verbose=-1),method="isotonic",cv=5)
    lgbm.fit(X,ye,sample_weight=wb)
    trained["LightGBM"]={"model":lgbm,"scaler":None,"le":le,"cols":list(X.columns)}
    sc=StandardScaler(); Xs=pd.DataFrame(sc.fit_transform(X),columns=X.columns)
    lr=CalibratedClassifierCV(LogisticRegression(C=0.5,max_iter=2000,random_state=42,
        class_weight={0:1.0,1:1.15,2:1.0}),method="isotonic",cv=5)
    lr.fit(Xs,ye,sample_weight=wb)
    trained["Logistic Regression"]={"model":lr,"scaler":sc,"le":le,"cols":list(X.columns)}
    return trained

def load_and_train(path,fixtures_live):
    df=pd.read_csv(path)
    df["date"]=pd.to_datetime(df["date"])
    df=df[df["date"]>="2014-01-01"]
    df=df[df["tournament"].isin(COMP_WEIGHTS.keys())]
    df=df.dropna(subset=["home_score","away_score"])
    df["home_score"]=df["home_score"].astype(int)
    df["away_score"]=df["away_score"].astype(int)
    df["result"]=df.apply(lambda r:"H" if r["home_score"]>r["away_score"]
                          else("A" if r["away_score"]>r["home_score"] else "D"),axis=1)
    df["weight"]=df["tournament"].map(COMP_WEIGHTS).fillna(0.5)
    played=[f for f in fixtures_live if f["status"] in ["FT","AET","PEN"] and f["goals1"] is not None]
    n_inj=0
    if played:
        rows=[]
        for f in played:
            g1,g2=int(f["goals1"]),int(f["goals2"])
            rows.append({"date":pd.Timestamp(f["date"]),"home_team":f["team1"],
                         "away_team":f["team2"],"home_score":g1,"away_score":g2,
                         "tournament":"FIFA World Cup","weight":3.0,
                         "result":"H" if g1>g2 else("A" if g2>g1 else "D")})
        df=pd.concat([df,pd.DataFrame(rows)],ignore_index=True)
        df=df.sort_values("date").reset_index(drop=True); n_inj=len(played)
    history={}; Xr,yr,wr=[],[],[]
    for _,row in df.iterrows():
        t1=row.get("home_team",row.get("HomeTeam",""))
        t2=row.get("away_team",row.get("AwayTeam",""))
        res,w=row["result"],row["weight"]
        s1=_stats(t1,history); s2=_stats(t2,history)
        Xr.append(_feat(s1,s2,w)); yr.append(res); wr.append(w)
        Xr.append(_feat(s2,s1,w))
        yr.append({"H":"A","A":"H","D":"D"}[res]); wr.append(w*0.8)
        g1,g2=row["home_score"],row["away_score"]
        for team,scored,conc,win,opp in [(t1,g1,g2,res=="H",t2),(t2,g2,g1,res=="A",t1)]:
            pts=3 if((team==t1 and res=="H")or(team==t2 and res=="A"))else(1 if res=="D" else 0)
            qw=w*max(0.25,_fifa(opp))
            history.setdefault(team,[]).append({"pts":pts,"scored":scored,"conc":conc,
                                                "win":int(win),"draw":int(res=="D"),"w":w,"qw":qw})
    X=pd.concat(Xr,ignore_index=True); y=pd.Series(yr); w=pd.Series(wr)
    le=LabelEncoder().fit(["A","D","H"])
    trained=_train(X,y,w,le)
    return trained,history,len(df)-n_inj,n_inj

def predict_match(trained,model,s1,s2,tw=3.0,i1=0.0,i2=0.0):
    b=trained[model]
    X=_feat(s1,s2,tw,i1,i2)[b["cols"]].fillna(0)
    if b["scaler"]: X=pd.DataFrame(b["scaler"].transform(X),columns=b["cols"])
    pr=b["model"].predict_proba(X)[0]; cls=b["le"].classes_
    p=dict(zip(cls,pr)); p1,pD,p2=p.get("H",0),p.get("D",0),p.get("A",0)
    bal=1-abs(p1-p2); boost=bal*0.03; tot=p1+p2
    if tot>0: p1-=boost*(p1/tot); p2-=boost*(p2/tot)
    pD+=boost; s=p1+pD+p2
    e1=max(0.25,s1["scored"]*(1-s2["clean"])*(1-i1))
    e2=max(0.25,s2["scored"]*(1-s1["clean"])*(1-i2))
    return {"t1":p1/s,"D":pD/s,"t2":p2/s,
            "exp1":round(e1,2),"exp2":round(e2,2),
            "draw_boost":round(boost,4),"balance":round(bal,3)}

def score_dist(e1,e2,mg=5):
    rows=[]
    for g1 in range(mg+1):
        for g2 in range(mg+1):
            p=poisson.pmf(g1,e1)*poisson.pmf(g2,e2)
            rows.append({"Score":f"{g1}–{g2}","g1":g1,"g2":g2,"Proba (%)":round(p*100,2)})
    return pd.DataFrame(rows).sort_values("Proba (%)",ascending=False)

# ══════════════════════════════════════════════════════════════════════
# SIMULATION TOURNOI COMPLÈTE (avec vrai tableau)
# ══════════════════════════════════════════════════════════════════════

def build_matrix(trained,model,history,injuries={}):
    m={}
    for t1 in ALL_TEAMS:
        for t2 in ALL_TEAMS:
            if t1==t2: continue
            s1=_stats(t1,history); s2=_stats(t2,history)
            m[(t1,t2)]=predict_match(trained,model,s1,s2,3.0,
                                     injuries.get(t1,0),injuries.get(t2,0))
    return m

def simulate_one(matrix):
    """Simule UN tournoi complet et retourne le tableau détaillé."""
    group_res={}; all_thirds=[]

    for grp,teams in WC_GROUPS.items():
        pts={t:0 for t in teams}; gd={t:0 for t in teams}; gs={t:0 for t in teams}
        matches=[]
        for i,t1 in enumerate(teams):
            for t2 in teams[i+1:]:
                p=matrix[(t1,t2)]
                arr=np.array([p["t1"],p["D"],p["t2"]],dtype=float)
                arr=np.clip(arr,1e-6,1); arr/=arr.sum()
                r=np.random.choice(["W1","D","W2"],p=arr)
                g1=np.random.poisson(p["exp1"]); g2=np.random.poisson(p["exp2"])
                if r=="W1" and g1<=g2: g1=g2+1
                elif r=="W2" and g2<=g1: g2=g1+1
                elif r=="D": g2=g1
                if r=="W1":   pts[t1]+=3
                elif r=="W2": pts[t2]+=3
                else:         pts[t1]+=1; pts[t2]+=1
                gd[t1]+=g1-g2; gd[t2]+=g2-g1
                gs[t1]+=g1; gs[t2]+=g2
                matches.append((t1,t2,g1,g2,r))
        ranked=sorted(teams,key=lambda t:(pts[t],gd[t],gs[t]),reverse=True)
        group_res[grp]={"ranked":ranked,"pts":pts,"gd":gd,"gs":gs,
                        "matches":matches,
                        "1st":ranked[0],"2nd":ranked[1],
                        "3rd":ranked[2],"4th":ranked[3]}
        all_thirds.append({"team":ranked[2],"group":grp,
                           "pts":pts[ranked[2]],"gd":gd[ranked[2]],"gs":gs[ranked[2]]})

    # 8 meilleurs 3èmes
    thirds_s=sorted(all_thirds,key=lambda x:(-x["pts"],-x["gd"],-x["gs"]))
    q3=[x["team"] for x in thirds_s[:8]]
    q3g=[x["group"] for x in thirds_s[:8]]

    def resolve(slot):
        if slot.startswith("W_"):   return group_res[slot[2:]]["1st"]
        elif slot.startswith("R_"): return group_res[slot[2:]]["2nd"]
        elif slot.startswith("T3_"):
            eligible=list(slot[3:])
            for g in q3g:
                if g in eligible: return q3[q3g.index(g)]
            return q3[0] if q3 else "TBD"
        return "TBD"

    def ko_match(t1,t2):
        if t1=="TBD": return t2
        if t2=="TBD": return t1
        p=matrix[(t1,t2)]; tot=p["t1"]+p["t2"]
        return t1 if np.random.random()<(p["t1"]/tot if tot>0 else 0.5) else t2

    r32_pairs=[(resolve(s1),resolve(s2)) for s1,s2 in R32_SLOTS]
    r32_res=[(t1,t2,ko_match(t1,t2)) for t1,t2 in r32_pairs]
    r32w=[w for _,_,w in r32_res]

    r16_pairs=[(r32w[i],r32w[i+1]) for i in range(0,len(r32w)-1,2)]
    r16_res=[(t1,t2,ko_match(t1,t2)) for t1,t2 in r16_pairs]
    r16w=[w for _,_,w in r16_res]

    qf_pairs=[(r16w[i],r16w[i+1]) for i in range(0,len(r16w)-1,2)]
    qf_res=[(t1,t2,ko_match(t1,t2)) for t1,t2 in qf_pairs]
    qfw=[w for _,_,w in qf_res]

    sf_pairs=[(qfw[i],qfw[i+1]) for i in range(0,len(qfw)-1,2)]
    sf_res=[(t1,t2,ko_match(t1,t2)) for t1,t2 in sf_pairs]
    sfw=[w for _,_,w in sf_res]

    fin=(sfw[0],sfw[1]) if len(sfw)>=2 else ("TBD","TBD")
    champion=ko_match(fin[0],fin[1])
    finalist=fin[1] if champion==fin[0] else fin[0]

    return {"groups":group_res,"thirds":thirds_s,"q3":q3,
            "r32":r32_res,"r16":r16_res,"qf":qf_res,"sf":sf_res,
            "final":(fin[0],fin[1],champion),"champion":champion,"finalist":finalist}

def simulate_n(trained,model,history,injuries={},n=5000):
    """Simule n tournois et retourne les probabilités agrégées."""
    matrix=build_matrix(trained,model,history,injuries)
    gp={}
    for grp,teams in WC_GROUPS.items():
        for i,t1 in enumerate(teams):
            for t2 in teams[i+1:]:
                p=matrix[(t1,t2)]
                arr=np.array([p["t1"],p["D"],p["t2"]],dtype=float)
                arr=np.clip(arr,1e-6,1); arr/=arr.sum()
                gp[(t1,t2)]=(arr,p["exp1"],p["exp2"])
    kop={}
    for t1 in ALL_TEAMS:
        for t2 in ALL_TEAMS:
            if t1==t2: continue
            p=matrix[(t1,t2)]; tot=p["t1"]+p["t2"]
            kop[(t1,t2)]=p["t1"]/tot if tot>0 else 0.5
    ti={t:i for i,t in enumerate(ALL_TEAMS)}
    counts=np.zeros((len(ALL_TEAMS),7),dtype=np.int32)
    for _ in range(n):
        thirds=[]; qualified=[]
        for grp,teams in WC_GROUPS.items():
            pts=np.zeros(len(teams)); gd=np.zeros(len(teams))
            tidx={t:i for i,t in enumerate(teams)}
            for i,t1 in enumerate(teams):
                for t2 in teams[i+1:]:
                    probs,e1,e2=gp[(t1,t2)]; r=np.random.choice(3,p=probs)
                    g1=np.random.poisson(e1); g2=np.random.poisson(e2)
                    if r==0:   pts[tidx[t1]]+=3; gd[tidx[t1]]+=max(1,g1-g2)
                    elif r==2: pts[tidx[t2]]+=3; gd[tidx[t2]]+=max(1,g2-g1)
                    else:      pts[tidx[t1]]+=1; pts[tidx[t2]]+=1
            order=sorted(range(len(teams)),key=lambda i:(-pts[i],-gd[i]))
            ranked=[teams[i] for i in order]
            qualified+=ranked[:2]
            counts[ti[ranked[0]],0]+=1; counts[ti[ranked[1]],0]+=1
            thirds.append((ranked[2],pts[tidx[ranked[2]]],gd[tidx[ranked[2]]]))
        thirds.sort(key=lambda x:(-x[1],-x[2]))
        for t,_,_ in thirds[:8]: qualified.append(t); counts[ti[t],0]+=1
        np.random.shuffle(qualified)
        def ko_rnd(tin,col):
            out=[]
            for i in range(0,len(tin)-1,2):
                p=kop[(tin[i],tin[i+1])]
                w=tin[i] if np.random.random()<p else tin[i+1]
                counts[ti[w],col]+=1; out.append(w)
            return out
        q=qualified
        for col in [1,2,3,4,5]: q=ko_rnd(q,col)
        if q: counts[ti[q[0]],6]+=1
    rows=[]
    for t in ALL_TEAMS:
        i=ti[t]
        rows.append({"Équipe":t,"FIFA":FIFA_RANKINGS.get(t,80),
                     "Groupe (%)":round(counts[i,0]/n*100,1),
                     "R32 (%)":round(counts[i,1]/n*100,1),
                     "R16 (%)":round(counts[i,2]/n*100,1),
                     "Quarts (%)":round(counts[i,3]/n*100,1),
                     "Demies (%)":round(counts[i,4]/n*100,1),
                     "Finale (%)":round(counts[i,5]/n*100,1),
                     "🏆 Victoire (%)":round(counts[i,6]/n*100,1)})
    return pd.DataFrame(rows).sort_values("🏆 Victoire (%)",ascending=False).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════
# VISUALISATION BRACKET
# ══════════════════════════════════════════════════════════════════════

def render_groups(result):
    """Affiche les tableaux de tous les groupes."""
    cols=st.columns(4)
    for idx,(grp,data) in enumerate(result["groups"].items()):
        with cols[idx%4]:
            st.markdown(f"**Groupe {grp}**")
            rows=[]
            for pos,team in enumerate(data["ranked"],1):
                if pos<=2: q="✅"
                elif team in result["q3"]: q="🟡"
                else: q="❌"
                rows.append({"":q,"Équipe":f"{F(team)} {team}",
                             "Pts":data["pts"][team],"DB":data["gd"][team],
                             "BP":data["gs"][team]})
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)

def render_ko_round(matches, title, col_count=2):
    """Affiche une phase éliminatoire avec mise en forme."""
    st.markdown(f"### {title}")
    cols=st.columns(col_count)
    for i,(t1,t2,w) in enumerate(matches):
        with cols[i%col_count]:
            bg1="#1a3a1a" if w==t1 else "#1a1a1a"
            bg2="#1a3a1a" if w==t2 else "#1a1a1a"
            bord1="#2ecc71" if w==t1 else "#333"
            bord2="#2ecc71" if w==t2 else "#333"
            st.markdown(f"""
<div style="border-radius:8px;overflow:hidden;margin:4px 0;border:1px solid #444">
  <div style="background:{bg1};border-left:4px solid {bord1};
       padding:8px 12px;font-size:0.9rem;color:{'#2ecc71' if w==t1 else '#ccc'}">
    {'🏆 ' if w==t1 else ''}{F(t1)} <b>{t1}</b>
  </div>
  <div style="background:#111;padding:2px 12px;font-size:0.7rem;color:#555;text-align:center">VS</div>
  <div style="background:{bg2};border-left:4px solid {bord2};
       padding:8px 12px;font-size:0.9rem;color:{'#2ecc71' if w==t2 else '#ccc'}">
    {'🏆 ' if w==t2 else ''}{F(t2)} <b>{t2}</b>
  </div>
</div>""",unsafe_allow_html=True)

def render_bracket_visual(result):
    """Vue d'ensemble du bracket en Plotly."""
    stages_data=[
        ("R32",result["r32"],16),
        ("R16",result["r16"],8),
        ("QF", result["qf"],4),
        ("SF", result["sf"],2),
        ("F",  [result["final"]],1),
    ]
    stage_labels={"R32":"Round of 32","R16":"Round of 16",
                  "QF":"Quarts","SF":"Demies","F":"Finale 🏆"}
    stage_colors={"R32":"#3498db","R16":"#f39c12","QF":"#e67e22","SF":"#e74c3c","F":"gold"}

    fig=go.Figure()
    x_pos={"R32":0,"R16":2,"QF":4,"SF":6,"F":8}
    y_spacing={"R32":1.5,"R16":3,"QF":6,"SF":12,"F":24}

    for stage,(label,matches,_) in zip(["R32","R16","QF","SF","F"],
                                        [(sl,m,n) for sl,m,n in stages_data]):
        n=len(matches)
        if n==0: continue
        sp=y_spacing[stage]; x=x_pos[stage]; col=stage_colors[stage]

        # Titre de colonne
        fig.add_annotation(x=x,y=(n-1)*sp*2+2.5,text=f"<b>{stage_labels[stage]}</b>",
                           showarrow=False,font=dict(size=10,color=col),xanchor="center")

        for mi,(t1,t2,w) in enumerate(matches):
            yc=(mi-(n-1)/2)*sp*2
            for ti,(team,pos) in enumerate([(t1,0.5),(t2,-0.5)]):
                is_win=team==w
                c="#2ecc71" if is_win else "#888"
                bg="rgba(46,204,113,0.15)" if is_win else "rgba(40,40,40,0.8)"
                txt=f"{'🏆 ' if (stage=='F' and is_win) else ''}{F(team)} {team}"
                fig.add_annotation(
                    x=x,y=yc+pos*0.6,text=txt,showarrow=False,
                    font=dict(size=8,color=c,family="Arial"),
                    xanchor="center",bgcolor=bg,
                    bordercolor=c,borderwidth=1 if is_win else 0,borderpad=3)

            # Ligne de connexion verticale
            fig.add_shape(type="line",x0=x,y0=yc-0.3,x1=x,y1=yc+0.3,
                         line=dict(color="#333",width=1))

    # Champion
    champ=result["champion"]
    fig.add_annotation(x=9.5,y=0,
                       text=f"<b>🏆 {F(champ)} {champ}</b>",
                       showarrow=False,font=dict(size=14,color="gold"),
                       xanchor="center",bgcolor="rgba(20,15,0,0.95)",
                       bordercolor="gold",borderwidth=2,borderpad=10)

    max_y=(len(result["r32"])-1)*y_spacing["R32"]*2+4
    fig.update_layout(
        height=max(700,max_y*25),
        paper_bgcolor="#0a0f1e",plot_bgcolor="#0a0f1e",
        xaxis=dict(visible=False,range=[-1,11]),
        yaxis=dict(visible=False,range=[-(max_y/2+2),(max_y/2+4)]),
        margin=dict(t=30,b=20,l=10,r=10),showlegend=False)
    return fig

# ══════════════════════════════════════════════════════════════════════
# GAUGE CHART
# ══════════════════════════════════════════════════════════════════════

def gauge_chart(label,val,color):
    fig=go.Figure(go.Indicator(
        mode="gauge+number",value=val*100,
        number={"suffix":"%","font":{"size":32,"color":color}},
        title={"text":label,"font":{"size":14,"color":"#bbb"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#333"},
               "bar":{"color":color,"thickness":0.28},
               "bgcolor":"#0d1b35","steps":[{"range":[0,100],"color":"#111a30"}]}))
    fig.update_layout(height=190,margin=dict(t=38,b=5,l=12,r=12),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ══════════════════════════════════════════════════════════════════════
# STREAMLIT
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="🏆 WC 2026",page_icon="🏆",
                   layout="wide",initial_sidebar_state="expanded")
st.markdown("<style>.block-container{padding-top:1rem}</style>",unsafe_allow_html=True)

# Cache
@st.cache_data(ttl=120,show_spinner="📡 Matchs live...")
def _fixtures(): return get_fixtures()
@st.cache_data(ttl=300,show_spinner="📊 Classements...")
def _standings(): return get_standings()
@st.cache_data(ttl=1800,show_spinner="🏥 Blessures...")
def _injuries(): return get_injuries()
@st.cache_data(ttl=600,show_spinner="⚽ Buteurs...")
def _scorers(): return get_top_scorers()
@st.cache_resource(show_spinner="🧠 Entraînement ML... (~60 sec)")
def _model(n): return load_and_train(RESULTS_CSV,_fixtures())
@st.cache_data(show_spinner="⚡ Calcul matrice...")
def _matrix(mod,inj_key): return build_matrix(trained,mod,history,sim_injuries)

# Chargement
fixtures=_fixtures(); standings=_standings()
injuries=_injuries(); scorers=_scorers()
n_played=len([f for f in fixtures if f["status"] in ["FT","AET","PEN"]])
trained,history,n_hist,n_live=_model(n_played)

# Blessures API
team_inj=defaultdict(list)
for inj in injuries: team_inj[inj["team"]].append(inj["player"])
api_impacts={}
for team,players in team_inj.items():
    if team not in KEY_PLAYERS: continue
    impact=0.0
    for p in players:
        for kn,val in KEY_PLAYERS[team].items():
            if kn.lower() in p.lower(): impact+=val
    if impact>0: api_impacts[team]=min(impact,0.75)

# Sidebar
with st.sidebar:
    st.markdown("## 🏆 WC 2026")
    if API_KEY: st.success("✅ API connectée")
    else: st.error("❌ Clé API manquante")
    st.divider()
    model_choice=st.selectbox("🧠 Modèle",["LightGBM","Logistic Regression"])
    st.divider()
    if st.button("🔄 Actualiser",use_container_width=True):
        for f in CACHE_DIR.glob("*.json"): f.unlink()
        st.cache_data.clear(); st.rerun()
    st.caption(f"📊 {n_hist:,} matchs · {n_live} CdM 2026 intégrés")
    st.divider()
    st.caption("Made with ❤️ + ML + API-Football")

# Onglets
tab1,tab2,tab3,tab4,tab5=st.tabs([
    "🔮 Prédiction","📅 Matchs","🏆 Tableau CdM","📊 Groupes","🏥 Équipes"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — PRÉDICTION
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔮 Prédire un match")
    st.caption("Terrain neutre · Symétrique · Pays hôtes 🇺🇸🇨🇦🇲🇽 bénéficient d'un léger avantage")
    c1,c2=st.columns(2)
    with c1:
        st.markdown("### ⚽ Équipe 1")
        team1=st.selectbox("",ALL_TEAMS,index=ALL_TEAMS.index("France") if "France" in ALL_TEAMS else 0,key="t1")
    with c2:
        st.markdown("### ⚽ Équipe 2")
        opts2=[t for t in ALL_TEAMS if t!=team1]
        team2=st.selectbox("",opts2,index=opts2.index("Brazil") if "Brazil" in opts2 else 0,key="t2")
    if team1 in HOST_NATIONS: st.info(f"🏟️ {team1} joue à domicile (+{HOST_BOOST:.0%})")
    elif team2 in HOST_NATIONS: st.info(f"🏟️ {team2} joue à domicile (+{HOST_BOOST:.0%})")
    c1,c2=st.columns(2)
    with c1: phase=st.selectbox("🏆 Phase",["Groupes","Round of 32","Round of 16","Quarts","Demies","Finale"])
    with c2: n_rec=st.slider("Matchs récents",5,20,10)
    st.divider()
    st.markdown("### 🏥 Blessures")
    c1,c2=st.columns(2)
    with c1:
        a1=api_impacts.get(team1,0.0)
        if a1>0: st.warning(f"⚠️ -{a1:.0%} détecté (API)")
        inj1=st.slider(f"{team1} (%)",0,60,int(a1*100),5,key="i1")/100
    with c2:
        a2=api_impacts.get(team2,0.0)
        if a2>0: st.warning(f"⚠️ -{a2:.0%} détecté (API)")
        inj2=st.slider(f"{team2} (%)",0,60,int(a2*100),5,key="i2")/100
    st.divider()
    s1=_stats(team1,history,n_rec); s2=_stats(team2,history,n_rec)
    c1,c2=st.columns(2)
    with c1:
        st.markdown(f"#### {F(team1)} {team1}")
        st.caption(f"FIFA #{FIFA_RANKINGS.get(team1,'?')} · Coach {COACHES.get(team1,10)} matchs · {s1['n']} analysés")
        a,b_,c,d,e=st.columns(5)
        a.metric("Forme",f"{s1['form']:.0%}"); b_.metric("Buts/m",f"{s1['scored']:.1f}")
        c.metric("Encaissés",f"{s1['conc']:.1f}"); d.metric("Victoires",f"{s1['win']:.0%}")
        e.metric("Clean sh.",f"{s1['clean']:.0%}")
    with c2:
        st.markdown(f"#### {F(team2)} {team2}")
        st.caption(f"FIFA #{FIFA_RANKINGS.get(team2,'?')} · Coach {COACHES.get(team2,10)} matchs · {s2['n']} analysés")
        a,b_,c,d,e=st.columns(5)
        a.metric("Forme",f"{s2['form']:.0%}"); b_.metric("Buts/m",f"{s2['scored']:.1f}")
        c.metric("Encaissés",f"{s2['conc']:.1f}"); d.metric("Victoires",f"{s2['win']:.0%}")
        e.metric("Clean sh.",f"{s2['clean']:.0%}")

    if st.button("🔮 Prédire",type="primary",use_container_width=True):
        tw={"Groupes":3.0,"Round of 32":3.5,"Round of 16":4.0,"Quarts":4.5,"Demies":5.0,"Finale":5.5}.get(phase,3.0)
        pred=predict_match(trained,model_choice,s1,s2,tw,inj1,inj2)
        st.markdown(f"## {F(team1)} {team1} 🆚 {team2} {F(team2)}")
        st.caption(f"Phase : {phase} · Équilibre : {pred['balance']:.0%}")
        c1,c2,c3=st.columns(3)
        with c1: st.plotly_chart(gauge_chart(f"{F(team1)} {team1}",pred["t1"],"#2ecc71"),use_container_width=True)
        with c2: st.plotly_chart(gauge_chart("🤝 Match nul",pred["D"],"#f39c12"),use_container_width=True)
        with c3: st.plotly_chart(gauge_chart(f"{F(team2)} {team2}",pred["t2"],"#e74c3c"),use_container_width=True)
        wk=max({"t1":pred["t1"],"D":pred["D"],"t2":pred["t2"]},key=lambda k:pred[k])
        lb={"t1":f"{F(team1)} {team1}","D":"🤝 Nul","t2":f"{F(team2)} {team2}"}; pw=pred[wk]
        if pred["D"]>=0.27 and pred["draw_boost"]>0.03:
            st.warning(f"**🤝 Match équilibré — nul probable ({pred['D']:.1%})** | {team1} : {pred['t1']:.1%} | {team2} : {pred['t2']:.1%}")
        elif pw>0.55: st.success(f"**Pronostic : {lb[wk]}** — confiance {pw:.1%}")
        elif pw>0.42: st.warning(f"**Match serré** — {lb[wk]} légèrement favori ({pw:.1%})")
        else: st.info(f"**Très ouvert** — {lb[wk]} ({pw:.1%})")
        if pred["draw_boost"]>0.02: st.caption(f"⚖️ Boost nul +{pred['draw_boost']:.1%}")
        c1,c2,c3=st.columns(3)
        c1.metric(f"⚽ {team1}",pred["exp1"]); c2.metric(f"⚽ {team2}",pred["exp2"])
        c3.metric("Total",round(pred["exp1"]+pred["exp2"],2))
        st.divider(); st.subheader("📊 Distribution des scores")
        dist=score_dist(pred["exp1"],pred["exp2"])
        c1,c2=st.columns([1,2])
        with c1:
            st.dataframe(dist.head(8)[["Score","Proba (%)"]],hide_index=True,use_container_width=True)
            pt1=dist[dist["g1"]>dist["g2"]]["Proba (%)"].sum()
            pnu=dist[dist["g1"]==dist["g2"]]["Proba (%)"].sum()
            pt2=dist[dist["g2"]>dist["g1"]]["Proba (%)"].sum()
            st.caption(f"Poisson : {team1} {pt1:.1f}% | Nul {pnu:.1f}% | {team2} {pt2:.1f}%")
        with c2:
            piv=dist.pivot(index="g1",columns="g2",values="Proba (%)").fillna(0)
            fig=go.Figure(go.Heatmap(z=piv.values,
                x=[f"{team2} {g}" for g in piv.columns],
                y=[f"{team1} {g}" for g in piv.index],
                colorscale="YlOrRd",
                text=[[f"{v:.1f}%" for v in row] for row in piv.values],
                texttemplate="%{text}"))
            fig.update_layout(height=330,paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#ccc"),margin=dict(t=10,b=30,l=80,r=10))
            st.plotly_chart(fig,use_container_width=True)
        if fixtures:
            up=[f for f in fixtures if f["status"] in ["NS","TBD"] and {f["team1"],f["team2"]}=={team1,team2}]
            if up:
                ap=get_api_pred(up[0]["id"])
                if ap and ap.get("winner"):
                    st.divider()
                    st.caption(f"🤖 API-Football : Favori → **{ap['winner']}** | Score prévu : {ap.get('goals1','-')}–{ap.get('goals2','-')}")

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — MATCHS
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📅 Matchs & Résultats CdM 2026")
    live_m=[f for f in fixtures if f["status"] in ["1H","2H","HT","ET","P"]]
    played_m=[f for f in fixtures if f["status"] in ["FT","AET","PEN"]]
    upcoming_m=[f for f in fixtures if f["status"] in ["NS","TBD"]]
    c1,c2,c3,c4=st.columns(4)
    c1.metric("🔴 En cours",len(live_m)); c2.metric("✅ Joués",len(played_m))
    c3.metric("📅 À venir",len(upcoming_m)); c4.metric("📊 Total",len(fixtures))
    if live_m:
        st.divider(); st.markdown("### 🔴 En cours")
        for m in live_m:
            st.markdown(f"🔴 **{m['elapsed']}'** — **{FT(m['team1'])}** {m['goals1']}–{m['goals2']} **{FT(m['team2'])}** · _{m['venue']}_")
    if played_m:
        st.divider(); st.markdown("### ✅ Résultats")
        for m in reversed(played_m[-20:]):
            g1,g2=m["goals1"],m["goals2"]
            icon="🟡" if g1==g2 else("🟢" if g1>g2 else "🔵")
            st.markdown(f"{icon} `{m['date']}` — **{FT(m['team1'])}** {g1}–{g2} **{FT(m['team2'])}** | _{m['round']}_")
    if upcoming_m:
        st.divider(); st.markdown("### 📅 Prochains matchs + prédictions")
        for m in upcoming_m[:20]:
            s1m=_stats(m["team1"],history); s2m=_stats(m["team2"],history)
            i1=api_impacts.get(m["team1"],0.0); i2=api_impacts.get(m["team2"],0.0)
            pred=predict_match(trained,model_choice,s1m,s2m,3.0,i1,i2)
            with st.expander(f"📅 {m['date']} — **{FT(m['team1'])}** vs **{FT(m['team2'])}** ({m['round']})"):
                c1,c2,c3,c4=st.columns(4)
                c1.metric(FT(m["team1"]),f"{pred['t1']:.1%}"); c2.metric("🤝 Nul",f"{pred['D']:.1%}")
                c3.metric(FT(m["team2"]),f"{pred['t2']:.1%}"); c4.metric("Score",f"{pred['exp1']:.1f}–{pred['exp2']:.1f}")

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — TABLEAU CdM (nouveau !)
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🏆 Tableau Coupe du Monde 2026")

    c1,c2=st.columns([1,2])
    with c1:
        st.markdown("#### Options")
        sim_injuries=dict(api_impacts)
        if api_impacts:
            st.markdown("**🏥 Blessures actives**")
            for team,auto in api_impacts.items():
                v=st.slider(f"{FT(team)}",0,60,int(auto*100),5,key=f"bi_{team}")/100
                sim_injuries[team]=v

        n_sims=st.select_slider("Simulations probabilités",
                                 options=[1000,3000,5000,10000],value=3000)

        col_sim,col_one=st.columns(2)
        btn_one=col_one.button("🎲 Simuler UN tournoi",type="primary",use_container_width=True)
        btn_n=col_sim.button(f"📊 Probas ({n_sims:,} sims)",use_container_width=True)

    with c2:
        st.markdown("**Groupes officiels CdM 2026**")
        gc=st.columns(4)
        for idx,(grp,teams) in enumerate(WC_GROUPS.items()):
            with gc[idx%4]:
                st.markdown(f"**Groupe {grp}**")
                for t in teams:
                    inj=sim_injuries.get(t,0); h=" 🏟️" if t in HOST_NATIONS else ""
                    inj_s=f" 🏥" if inj>0 else ""
                    st.caption(f"{F(t)} {t} (#{FIFA_RANKINGS.get(t,'?')}){h}{inj_s}")

    # ── Simulation UN tournoi ──────────────────────────────────────
    if btn_one:
        with st.spinner("Simulation en cours..."):
            mat=build_matrix(trained,model_choice,history,sim_injuries)
            result=simulate_one(mat)
        st.session_state["bracket_result"]=result

    if "bracket_result" in st.session_state:
        result=st.session_state["bracket_result"]
        champ=result["champion"]; final=result["finalist"]

        # Bandeau champion
        st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a1200,#2a1f00);
     border:2px solid gold;border-radius:12px;padding:16px;
     text-align:center;margin:12px 0">
  <div style="color:gold;font-size:1.1rem;font-weight:600">🏆 CHAMPION CdM 2026</div>
  <div style="color:white;font-size:2rem;font-weight:700;margin:4px 0">{F(champ)} {champ}</div>
  <div style="color:#aaa;font-size:0.9rem">Finaliste : {F(final)} {final}</div>
</div>""",unsafe_allow_html=True)

        # Sélecteur de vue
        view=st.radio("",["🏟️ Groupes","📐 Bracket visuel","🔵 Round of 32",
                           "🟡 Round of 16","🟠 Quarts","🔴 Demies & Finale"],
                      horizontal=True)

        if view=="🏟️ Groupes":
            st.subheader("Phase de groupes")
            render_groups(result)
            st.divider()
            st.subheader("🟡 Meilleurs 3èmes qualifiés")
            rows3=[]
            for x in result["thirds"][:8]:
                rows3.append({"Équipe":FT(x["team"]),"Groupe":x["group"],
                              "Pts":x["pts"],"DB":x["gd"],"BP":x["gs"],"Statut":"✅ Qualifié"})
            for x in result["thirds"][8:]:
                rows3.append({"Équipe":FT(x["team"]),"Groupe":x["group"],
                              "Pts":x["pts"],"DB":x["gd"],"BP":x["gs"],"Statut":"❌ Éliminé"})
            st.dataframe(pd.DataFrame(rows3),hide_index=True,use_container_width=True)

        elif view=="📐 Bracket visuel":
            st.plotly_chart(render_bracket_visual(result),use_container_width=True)

        elif view=="🔵 Round of 32":
            render_ko_round(result["r32"],"🔵 Round of 32",2)

        elif view=="🟡 Round of 16":
            render_ko_round(result["r16"],"🟡 Round of 16",2)

        elif view=="🟠 Quarts":
            render_ko_round(result["qf"],"🟠 Quarts de finale",2)

        elif view=="🔴 Demies & Finale":
            render_ko_round(result["sf"],"🔴 Demi-finales",2)
            st.divider()
            t1,t2,w=result["final"]
            st.markdown(f"""
<div style="background:linear-gradient(135deg,#0a0a20,#15152e);
     border:2px solid gold;border-radius:16px;padding:24px;text-align:center;margin:16px 0">
  <div style="color:gold;font-size:1.3rem;font-weight:700">🏆 FINALE — 19 JUILLET 2026</div>
  <div style="color:#aaa;font-size:0.85rem;margin:4px 0">MetLife Stadium · East Rutherford, NJ</div>
  <div style="margin:16px 0;display:flex;justify-content:center;align-items:center;gap:24px">
    <div style="text-align:center">
      <div style="font-size:2.5rem">{F(t1)}</div>
      <div style="color:{'#2ecc71' if w==t1 else '#ccc'};font-size:1.2rem;font-weight:{'700' if w==t1 else '400'}">{t1}</div>
      {'<div style="color:gold">🏆 CHAMPION</div>' if w==t1 else ''}
    </div>
    <div style="color:#f39c12;font-size:1.5rem;font-weight:700">🆚</div>
    <div style="text-align:center">
      <div style="font-size:2.5rem">{F(t2)}</div>
      <div style="color:{'#2ecc71' if w==t2 else '#ccc'};font-size:1.2rem;font-weight:{'700' if w==t2 else '400'}">{t2}</div>
      {'<div style="color:gold">🏆 CHAMPION</div>' if w==t2 else ''}
    </div>
  </div>
</div>""",unsafe_allow_html=True)

    # ── Probabilités sur N simulations ────────────────────────────
    if btn_n:
        with st.spinner(f"Simulation {n_sims:,} tournois..."):
            df_sim=simulate_n(trained,model_choice,history,sim_injuries,n_sims)
        st.divider()
        st.subheader(f"📊 Probabilités sur {n_sims:,} simulations")
        top10=df_sim.head(10)
        fig=go.Figure(go.Bar(
            x=top10["🏆 Victoire (%)"],y=[FT(t) for t in top10["Équipe"]],
            orientation="h",
            marker=dict(color=top10["🏆 Victoire (%)"],colorscale="YlOrRd",showscale=False),
            text=[f"{v}%" for v in top10["🏆 Victoire (%)"]],textposition="outside"))
        fig.update_layout(title=f"Probabilité de remporter la CdM 2026",
                          yaxis=dict(autorange="reversed"),height=380,
                          paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#ccc"),xaxis=dict(gridcolor="#222"),
                          margin=dict(t=40,b=20,l=180,r=80))
        st.plotly_chart(fig,use_container_width=True)
        stages=["Groupe (%)","R32 (%)","R16 (%)","Quarts (%)","Demies (%)","Finale (%)","🏆 Victoire (%)"]
        top20=df_sim.head(20)
        fig2=go.Figure(go.Heatmap(
            z=top20[stages].values,
            x=[s.replace(" (%)","").replace("🏆 ","") for s in stages],
            y=[FT(t) for t in top20["Équipe"]],
            colorscale="YlOrRd",
            text=[[f"{v}%" for v in row] for row in top20[stages].values],
            texttemplate="%{text}"))
        fig2.update_layout(title="Probabilités par stade — Top 20",height=580,
                           paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#ccc"),
                           yaxis=dict(autorange="reversed"),
                           margin=dict(t=40,b=20,l=200,r=20))
        st.plotly_chart(fig2,use_container_width=True)
        st.dataframe(df_sim,use_container_width=True,height=500)

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — GROUPES & STATS
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📊 Groupes & Statistiques")
    if standings:
        st.subheader("🏆 Classements live")
        cols=st.columns(3)
        for idx,(grp_name,teams) in enumerate(standings.items()):
            with cols[idx%3]:
                st.markdown(f"**{grp_name}**")
                df_g=pd.DataFrame(teams)[["rank","team","played","won","drawn","lost","gf","ga","gd","pts","form"]]
                df_g.columns=["#","Équipe","J","V","N","D","BP","BC","Diff","Pts","Forme"]
                st.dataframe(df_g,hide_index=True,use_container_width=True)
    else:
        st.info("Classements disponibles une fois les matchs commencés.")
        cols=st.columns(4)
        for idx,(grp,teams) in enumerate(WC_GROUPS.items()):
            with cols[idx%4]:
                st.markdown(f"**Groupe {grp}**")
                for t in teams:
                    h=" 🏟️" if t in HOST_NATIONS else ""
                    st.caption(f"{F(t)} {t} (#{FIFA_RANKINGS.get(t,'?')}){h}")
    st.divider(); st.subheader("⚽ Top buteurs")
    if scorers:
        df_sc=pd.DataFrame(scorers); df_sc.columns=["Joueur","Équipe","Buts","Passes D.","Note"]
        c1,c2=st.columns([1,2])
        with c1: st.dataframe(df_sc,hide_index=True,use_container_width=True)
        with c2:
            fig=go.Figure(go.Bar(x=df_sc["Joueur"],y=df_sc["Buts"],marker_color="#f39c12",
                                  text=df_sc["Buts"],textposition="outside"))
            fig.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#ccc"),
                               xaxis=dict(tickangle=30,gridcolor="#222"),
                               yaxis=dict(gridcolor="#222"),margin=dict(t=10,b=80,l=40,r=10))
            st.plotly_chart(fig,use_container_width=True)
    else: st.info("Top buteurs disponibles une fois le tournoi lancé.")
    st.divider(); st.subheader("🔬 Analyse d'une équipe")
    team_sel=st.selectbox("Équipe",ALL_TEAMS,index=ALL_TEAMS.index("France") if "France" in ALL_TEAMS else 0)
    n_m=st.slider("Matchs récents",5,30,15,key="an")
    s=_stats(team_sel,history,n_m)
    st.caption(f"FIFA #{FIFA_RANKINGS.get(team_sel,'?')} · Coach {COACHES.get(team_sel,10)} matchs · Stabilité {_coach(team_sel):.0%}")
    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.metric("Forme",f"{s['form']:.0%}"); c2.metric("Buts/m",f"{s['scored']:.2f}")
    c3.metric("Encaissés/m",f"{s['conc']:.2f}"); c4.metric("Victoires",f"{s['win']:.0%}")
    c5.metric("Clean sheets",f"{s['clean']:.0%}"); c6.metric("Diff. buts",f"{s['gd']:+.2f}")
    if api_impacts.get(team_sel,0)>0: st.error(f"🏥 Impact blessures : -{api_impacts[team_sel]:.0%}")
    if team_sel in HOST_NATIONS: st.success(f"🏟️ Pays hôte — boost +{HOST_BOOST:.0%}")

# ══════════════════════════════════════════════════════════════════════
# TAB 5 — ÉQUIPES & BLESSURES
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.header("🏥 Équipes & Blessures")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("🩹 Blessures (API live)")
        if injuries:
            rows_inj=[]
            for team in sorted(set(i["team"] for i in injuries)):
                players=[i["player"] for i in injuries if i["team"]==team]
                impact=api_impacts.get(team,0.0)
                rows_inj.append({"Équipe":FT(team),"Nb":len(players),
                                  "Impact":f"-{impact:.0%}" if impact>0 else "<1%",
                                  "Joueurs":", ".join(players[:3])+("..." if len(players)>3 else "")})
            st.dataframe(pd.DataFrame(rows_inj).sort_values("Nb",ascending=False),
                         hide_index=True,use_container_width=True)
            tf=st.selectbox("Détail",["Toutes"]+sorted(set(i["team"] for i in injuries)))
            df_inj=pd.DataFrame(injuries)
            if tf!="Toutes": df_inj=df_inj[df_inj["team"]==tf]
            st.dataframe(df_inj,hide_index=True,use_container_width=True)
        else: st.info("Aucune blessure détectée.")
    with c2:
        st.subheader("⭐ Joueurs clés")
        for team,players in KEY_PLAYERS.items():
            with st.expander(f"{FT(team)} (FIFA #{FIFA_RANKINGS.get(team,'?')})"):
                st.dataframe(pd.DataFrame([{"Joueur":n,"Impact si absent":f"-{v:.0%}"}
                                            for n,v in players.items()]),
                             hide_index=True,use_container_width=True)
    st.divider(); st.subheader("📋 Toutes les équipes")
    rows_all=[]
    for t in ALL_TEAMS:
        rows_all.append({"Équipe":FT(t),
                         "Groupe":next((g for g,ts in WC_GROUPS.items() if t in ts),"?"),
                         "FIFA":FIFA_RANKINGS.get(t,80),
                         "Score FIFA":f"{_fifa(t):.2f}",
                         "Coach (m)":COACHES.get(t,10),
                         "Stabilité":f"{_coach(t):.0%}",
                         "Hôte":"🏟️" if t in HOST_NATIONS else ""})
    st.dataframe(pd.DataFrame(rows_all).sort_values("FIFA"),hide_index=True,use_container_width=True)
    st.divider(); st.subheader("ℹ️ Modèle")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Matchs historiques",f"{n_hist:,}"); c2.metric("CdM 2026 intégrés",n_live)
    c3.metric("Période","2014–2026"); c4.metric("Features ML","27")
    st.markdown("""
- **FIFA ranking** : prior fort (20% blending), pondère par niveau adversaire
- **Symétrie** : augmentation miroir → terrain neutre garanti  
- **Nuls** : ×1.15 + boost post-prédiction selon équilibre du match
- **Blessures** : détection automatique API + ajustement manuel
- **Coach** : score stabilité, poids 10% dans les features
- **Calibration** : isotonique 5-fold cross-validation
    """)
