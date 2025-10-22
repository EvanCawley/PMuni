# ==========================================================
# RELEX BS — Workflow Build (Portfolio + Sourcing Wizard + Awards)
# Clean, production-style; guided multi-screen sourcing flow
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import uuid, random, math, re
from datetime import datetime, timedelta

# ==========================================================
# PAGE CONFIG & THEME
# ==========================================================
st.set_page_config(page_title="RELEX BS", layout="wide")
TOP_PAD_PX = 40   # ≈1 cm white space
INDIGO="#4F46E5"; CYAN="#06B6D4"; GREEN="#10B981"; SLATE="#64748B"; GRAY="#6B7280"

st.markdown(f"""
<style>
.block-container {{ padding-top:{TOP_PAD_PX}px !important; }}
/* Top header row: title + bell */
.header-row {{ display:flex; align-items:center; justify-content:space-between; }}
.title-main {{
  font-size:40px; font-weight:800; text-align:left; margin: 0 0 6px 0;
  background: linear-gradient(90deg, {INDIGO}, {CYAN});
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.breadcrumbs {{ color:{SLATE}; font-size:14px; margin: 0 0 12px 0; }}
.bell-wrap {{ position: relative; font-size: 26px; line-height: 1; }}
.bell-badge {{
  position:absolute; top:-4px; right:-10px; background:#ef4444; color:#fff;
  border-radius:999px; padding:2px 6px; font-size:11px;
}}
/* Workflow tiles */
.workflow {{
  border-radius: 16px; padding: 20px; color: white; min-height: 220px;
  display: flex; flex-direction: column; justify-content: space-between;
  transition: 0.15s ease-in-out;
}}
.workflow:hover {{ filter: brightness(0.96); cursor: pointer; }}
.portfolio {{ background: {INDIGO}; }}
.sourcing  {{ background: {CYAN};   }}
.awards    {{ background: {GREEN};  }}
/* KPI + Cards */
.kpi {{ border:1px solid #e5e7eb; border-radius:16px; padding:16px; background:#fff; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid #e5e7eb; }}
.badge.good {{ background:#ECFDF5; color:#10B981; border-color:#D1FAE5; }}
.badge.warn {{ background:#FFFBEB; color:#F59E0B; border-color:#FEF3C7; }}
.badge.bad  {{ background:#FEF2F2; color:#EF4444; border-color:#FEE2E2; }}
.card {{ border:1px solid #e5e7eb; border-radius:16px; padding:16px; background:#fff; }}
/* Buttons row for wizard */
.btnrow {{ display:flex; gap:10px; justify-content:flex-end; }}
hr {{
  margin: 8px 0 18px;
}}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SESSION STATE INIT
# ==========================================================
defaults = {
    "view": "Landing",                    # Landing | Portfolio | Sourcing | Awards
    "notifications": [],                  # [{msg, time}]
    "price_alerts": {},                   # sku_code -> target float
    "portfolio": None,                    # DataFrame
    "portfolio_ai": {},                   # sku_code -> insight
    "sourcing": {                         # Wizard state
        "step": "Scope",                  # Scope | Bids | Review | Award
        "project": {},                    # scope dict
        "bids": None,                     # DataFrame
        "ranked": None,                   # DataFrame
        "weights": {"price":55,"volume":15,"service":20,"contract":10},
    },
    "awards": [],                         # live awards list
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================================
# HELPERS
# ==========================================================
def add_notification(msg: str):
    st.session_state.notifications.append({
        "msg": msg,
        "time": datetime.utcnow().strftime("%H:%M UTC %Y-%m-%d")
    })

def bell_html():
    count = len(st.session_state.notifications)
    badge = f'<span class="bell-badge">{count}</span>' if count>0 else ""
    return f'<div class="bell-wrap">🔔{badge}</div>'

def breadcrumbs(*parts):
    st.markdown(f"<div class='breadcrumbs'>{' › '.join(parts)}</div>", unsafe_allow_html=True)

def load_csv(path: str):
    try:
        df = pd.read_csv(path)
        if df.empty: return None
        return df
    except Exception:
        return None

def demo_portfolio():
    cats=["Haircare","Oral Care","Snacks","Dairy","Skincare"]
    sups=["Unilever","P&G","Own Label Co.","Regional Dairy","Niche Snacks","Acme Beauty","OptiCare"]
    rows=[]
    for c in cats:
        for i in range(1,8):
            rows.append([
                f"{c[:3].upper()}-{i:03d}",
                f"{c} Item {i}",
                c,
                random.choice(sups),
                round(random.uniform(0.8,3.5),2),
                random.randint(30000,90000),
                (datetime.now()+timedelta(days=random.randint(60,540))).date().isoformat(),
                round(random.uniform(18,30),1)
            ])
    df=pd.DataFrame(rows,columns=[
        "sku_code","product_name","category","current_supplier",
        "current_unit_cost","monthly_volume_units","contract_end","target_margin_pct"
    ])
    # AI insight: compare vs category median
    med = df.groupby("category")["current_unit_cost"].transform("median")
    cond_over = df["current_unit_cost"] > 1.12 * med
    insight = np.where(cond_over, "Overpaying", "Healthy")
    # add some “Opportunity”
    mask_op = (~cond_over) & (np.random.rand(len(df))<0.15)
    insight = np.where(mask_op, "Opportunity", insight)
    df["ai_insight"] = insight
    return df

def price_history_for_sku(sku_code: str, base_cost: float) -> pd.DataFrame:
    seed = sum(ord(c) for c in sku_code)
    np.random.seed(seed % (2**32))
    months = pd.date_range(end=pd.Timestamp.today().normalize(), periods=12, freq="MS")
    steps = np.random.normal(0, 0.02, size=len(months))
    values = [max(0.2, base_cost * (1 + steps[:i+1].sum())) for i in range(len(months))]
    return pd.DataFrame({"month": months, "unit_cost": np.round(values, 2)})

def normalized_weights(w_cost, w_vol, w_serv, w_con):
    arr = np.array([w_cost, w_vol, w_serv, w_con], dtype=float)
    s = arr.sum()
    if s <= 0: return [0.55,0.15,0.20,0.10]
    return (arr / s).tolist()

def score_bid(row, weights):
    # Effective cost adjusted by rebates/promo and simple penalties/bonuses
    price = float(row["offer_price"])
    rebate = float(row.get("rebate_pct", 0.0))/100.0
    promo  = float(row.get("promo_funding_pct", 0.0))/100.0
    eff = price * (1 - rebate - promo)

    # penalties/bonuses
    sl = float(row.get("service_level_target_pct", 98))
    lt = float(row.get("lead_time_days", 7))
    contract = float(row.get("contract_months", 12))
    freight_included = str(row.get("freight_included","yes")).lower() in ["yes","true","1","y"]

    eff += (0.03 if not freight_included else 0.0)  # freight penalty per unit
    eff += max(0, 98 - sl) * 0.001                  # service penalty per pp below 98
    eff += max(0, lt - 7) * 0.002                   # lead time penalty per extra day
    eff -= (contract / 24.0) * 0.01                 # contract length bonus

    w_cost, w_vol, w_serv, w_con = normalized_weights(
        weights["price"], weights["volume"], weights["service"], weights["contract"]
    )
    cost_component = 1.0 / max(eff, 1e-6)
    vol_component = np.log1p(float(row.get("min_volume_units_per_month", 0.0)))
    service_component = sl / 100.0
    score = (w_cost*cost_component) + (w_vol*vol_component) + (w_serv*service_component) + (w_con*(contract/24.0))
    return float(score), float(eff)

def ai_why(row, eff, w):
    parts = [f"Effective unit cost ≈ £{eff:.2f} (after rebates/promo)."]
    if w["price"] >= 0.5: parts.append("High emphasis on cost → lower effective price dominates.")
    if float(row.get("min_volume_units_per_month",0)) >= 50000 and w["volume"]>0: parts.append("Strong monthly capacity boosts ranking.")
    if float(row.get("service_level_target_pct",0)) >= 97 and w["service"]>0: parts.append("High OTIF reduces risk.")
    if float(row.get("contract_months",0)) >= 12 and w["contract"]>0: parts.append("Longer contract earns stability bonus.")
    if str(row.get("freight_included","yes")).lower() not in ["yes","true","1","y"]: parts.append("Penalty applied: freight not included.")
    return " ".join(parts)

# ==========================================================
# LOAD EXTERNAL DATA (OPTIONAL)
# ==========================================================
PORTF_HIST = load_csv("portfolio_history.csv")    # optional
AWARDS_HIST = load_csv("awards_history.csv")      # optional
BIDS_HIST   = load_csv("bids_history.csv")        # optional

# ==========================================================
# NOTIFICATIONS SIDEBAR
# ==========================================================
def notifications_panel():
    st.sidebar.markdown("### 🔔 Notifications")
    if len(st.session_state.notifications)==0:
        st.sidebar.caption("No notifications yet.")
    else:
        for n in reversed(st.session_state.notifications[-12:]):
            st.sidebar.success(f"{n['msg']}  \n*{n['time']}*")

# ==========================================================
# TOP BAR (TITLE + BELL) + NAV
# ==========================================================
def top_bar_and_nav():
    c1,c2 = st.columns([6,1])
    with c1:
        st.markdown("<div class='title-main'>RELEX BS</div>", unsafe_allow_html=True)
        # breadcrumbs per view
        if st.session_state.view == "Portfolio":
            breadcrumbs("Home","Portfolio")
        elif st.session_state.view == "Sourcing":
            step = st.session_state.sourcing["step"]
            breadcrumbs("Home","Sourcing", step)
        elif st.session_state.view == "Awards":
            breadcrumbs("Home","Awards")
        else:
            breadcrumbs("Home")
    with c2:
        st.markdown(bell_html(), unsafe_allow_html=True)

    # Nav buttons
    nav = st.container()
    n1,n2,n3 = st.columns(3)
    if n1.button("📊 Portfolio", use_container_width=True): st.session_state.view="Portfolio"
    if n2.button("🧩 Sourcing",   use_container_width=True): st.session_state.view="Sourcing"
    if n3.button("🏆 Awards",     use_container_width=True): st.session_state.view="Awards"
    st.markdown("<hr/>", unsafe_allow_html=True)

# ==========================================================
# LANDING PAGE (Tiles)
# ==========================================================
def landing():
    top_bar_and_nav()
    st.caption("Portfolio Intelligence • Sourcing Wizard • AI Reasoning • Awards Analytics")
    colA, colB, colC = st.columns(3)
    def tile(title, desc, cls, target):
        st.markdown(f"<div class='workflow {cls}'><h3>{title}</h3><p>{desc}</p></div>", unsafe_allow_html=True)
        st.button(f"Open {title}", key=f"open_{target}", use_container_width=True, on_click=lambda: set_view(target))
    with colA:
        tile("SKU Portfolio","See suppliers, costs, trends & AI opportunities.","portfolio","Portfolio")
    with colB:
        tile("SKU Sourcing","Create scope, collect bids, AI-rank, and award.","sourcing","Sourcing")
    with colC:
        tile("Awards","Savings, performance, expiries, and history.","awards","Awards")

def set_view(v): st.session_state.view = v

# ==========================================================
# PORTFOLIO (Workflow 1)
# ==========================================================
def ensure_portfolio_loaded():
    if st.session_state.portfolio is None:
        st.session_state.portfolio = demo_portfolio()

def portfolio_page():
    top_bar_and_nav()
    ensure_portfolio_loaded()
    df = st.session_state.portfolio.copy()

    # KPIs
    needs = (df["ai_insight"]!="Healthy").sum()
    k1,k2,k3 = st.columns(3)
    with k1: st.markdown(f"<div class='kpi'><div class='badge bad'>Potentially Overpaying</div><h2>{needs}</h2></div>", unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='kpi'><div class='badge good'>Healthy</div><h2>{(df['ai_insight']=='Healthy').sum()}</h2></div>", unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='kpi'><div class='badge warn'>Total SKUs</div><h2>{len(df)}</h2></div>", unsafe_allow_html=True)

    # Category visuals
    g1,g2 = st.columns(2)
    with g1:
        st.markdown("**Price Parity vs Market**")
        cat_cost = df.groupby("category")["current_unit_cost"].mean().sort_values()
        st.bar_chart(cat_cost)
    with g2:
        st.markdown("**Total monthly volume by category**")
        cat_vol = df.groupby("category")["monthly_volume_units"].sum().sort_values()
        st.bar_chart(cat_vol)

    st.markdown("### Active Product Portfolio")
    # Natural-language query
    q = st.text_input("Ask RELEX BS (e.g. 'Show me SKUs overpaying by 10% vs median')","")
    df_view = df
    if q.strip():
        m = re.search(r'(\d+)\s*%', q.lower())
        pct = int(m.group(1)) if m else 10
        if "overpay" in q.lower():
            med = df.groupby("category")["current_unit_cost"].transform("median")
            df_view = df[df["current_unit_cost"] > (1 + pct/100.0) * med]
            st.caption(f"Matched {len(df_view)} SKUs over {pct}% vs. category median.")
    st.dataframe(df_view, use_container_width=True)

    # Drilldown: select SKU
    st.markdown("#### Drilldown & Actions")
    cA,cB,cC = st.columns(3)
    with cA:
        sku_opts = sorted(df["sku_code"].unique())
        if not sku_opts:
            st.info("No SKUs found."); return
        sku = st.selectbox("Select SKU", sku_opts, key="portfolio_sku")
        if sku:
            row = df[df["sku_code"]==sku].iloc[0]
            hist = price_history_for_sku(sku, float(row["current_unit_cost"]))
            st.markdown(f"**Price history — {sku} ({row['product_name']})**")
            st.line_chart(hist.set_index("month"))

            # Contract timeline if CSV present
            if PORTF_HIST is not None:
                hh = PORTF_HIST[PORTF_HIST["sku_code"]==sku].copy()
                if not hh.empty:
                    st.markdown("**Contract timeline (unit cost midpoint)**")
                    hh["start_date"] = pd.to_datetime(hh["start_date"], errors="coerce")
                    hh["end_date"] = pd.to_datetime(hh["end_date"], errors="coerce")
                    hh["midpoint"] = hh["start_date"] + (hh["end_date"] - hh["start_date"])/2
                    st.line_chart(hh.set_index("midpoint")["unit_cost"])
    with cB:
        # Price alert trigger
        if sku:
            latest = hist["unit_cost"].iloc[-1]
            tgt_default = max(0.1, float(row["current_unit_cost"]) - 0.1)
            target = st.number_input("Set price alert target (£)", 0.0, 50.0, float(tgt_default), 0.05, key="price_alert_input")
            if st.button("Trigger Notification when price ≤ target", key="set_alert_btn"):
                st.session_state.price_alerts[sku] = float(target)
                add_notification(f"Alert set for {sku}: notify when ≤ £{target:.2f}")
                st.success("Notification added!")
            existing = st.session_state.price_alerts.get(sku)
            if existing is not None and latest <= existing:
                add_notification(f"PRICE HIT for {sku}: £{latest:.2f} ≤ £{existing:.2f}")
                st.info(f"Latest price (£{latest:.2f}) is at/under your alert (£{existing:.2f}). Notification queued.")
    with cC:
        # Re-source action
        if sku and st.button("Start Re-Sourcing ▶️", use_container_width=True):
            st.session_state.sourcing["project"] = dict(
                product_name=row["product_name"],
                target_price=float(row["current_unit_cost"]) * 0.95,
                contract_months=12,
                monthly_volume=int(row["monthly_volume_units"]),
                must_have="",
                nice_to_have="",
                must_not_have="",
                lead_time_days=10,
                service_level_target_pct=97,
                certifications="Vegan, Cruelty Free",
                origin_pref="EU or UK",
                sustainability="PCR packaging ≥30%",
                packaging="Recyclable bottle",
            )
            st.session_state.sourcing["step"] = "Scope"
            set_view("Sourcing")
        # Terminate (demo)
        if sku and st.button("Terminate Contract (demo)"):
            st.session_state.portfolio.loc[st.session_state.portfolio["sku_code"]==sku,"contract_end"] = datetime.utcnow().date().isoformat()
            add_notification(f"Contract terminated for {sku}")
            st.warning(f"Contract for {sku} set to end today (prototype).")

# ==========================================================
# SOURCING WIZARD (Workflow 2)
# Steps: Scope → Bids → Review → Award
# ==========================================================
def sourcing_page():
    top_bar_and_nav()
    step = st.session_state.sourcing["step"]

    if step == "Scope":
        scope_step()
    elif step == "Bids":
        bids_step()
    elif step == "Review":
        review_step()
    elif step == "Award":
        award_step()
    else:
        st.session_state.sourcing["step"] = "Scope"
        scope_step()

def scope_step():
    st.subheader("1️⃣ Scope Setup")
    proj = st.session_state.sourcing["project"] or {}
    c1,c2 = st.columns(2)
    with c1:
        product_name = st.text_input("Product Name", proj.get("product_name","Hydrating Shampoo 300ml"))
        target_price = st.number_input("Target Price (£/unit)", 0.0, 50.0, float(proj.get("target_price", 2.50)), 0.05)
        contract_months = st.number_input("Contract Length (months)", 1, 60, int(proj.get("contract_months", 12)))
        monthly_volume = st.number_input("Monthly Volume (units)", 0, 10_000_000, int(proj.get("monthly_volume", 50000)), 1000)
        lead_time_days = st.slider("Max Lead Time (days)", 1, 60, int(proj.get("lead_time_days", 10)))
        service_level_target_pct = st.slider("Service Level Target (OTIF %)", 90, 100, int(proj.get("service_level_target_pct", 97)))
    with c2:
        must_have = st.text_area("Must-Have Ingredients / Attributes", proj.get("must_have","aloe vera, pro-vitamin B5"))
        nice_to_have = st.text_area("Nice-to-Have Ingredients / Attributes", proj.get("nice_to_have","argan oil"))
        must_not = st.text_area("Must-Not-Have Ingredients / Attributes", proj.get("must_not_have","sulfates, parabens"))
        certifications = st.text_input("Required Certifications", proj.get("certifications","Vegan, Cruelty Free"))
        origin_pref = st.text_input("Origin Preference", proj.get("origin_pref","EU or UK"))
        sustainability = st.text_input("Sustainability Requirements", proj.get("sustainability","PCR packaging ≥30%"))
        packaging = st.text_input("Packaging Type", proj.get("packaging","Recyclable bottle with flip-cap"))

    # Save scope
    st.session_state.sourcing["project"] = dict(
        product_name=product_name, target_price=target_price, contract_months=int(contract_months),
        monthly_volume=int(monthly_volume), lead_time_days=int(lead_time_days),
        service_level_target_pct=int(service_level_target_pct), must_have=must_have,
        nice_to_have=nice_to_have, must_not_have=must_not, certifications=certifications,
        origin_pref=origin_pref, sustainability=sustainability, packaging=packaging
    )

    # Wizard navigation
    colL, colR = st.columns([1,1])
    with colL:
        st.button("← Back to Portfolio", use_container_width=True, on_click=lambda: set_view("Portfolio"))
    with colR:
        st.button("Next: Tender Launch →", type="primary", use_container_width=True,
                  on_click=lambda: set_sourcing_step("Bids"))

def set_sourcing_step(s): st.session_state.sourcing["step"] = s

def generate_sample_bids(tid: str, proj: dict) -> pd.DataFrame:
    suppliers = ["Own Label Co.","Unilever","P&G","Regional Dairy","Niche Snacks","Acme Beauty","OptiCare"]
    base = float(proj["target_price"])
    rows=[]
    for s in suppliers:
        offer = round(max(0.2, base * np.random.uniform(0.88, 1.10)), 2)
        rows.append([
            tid, s, f"SKU-{uuid.uuid4().hex[:6].upper()}", offer,
            random.choice([12,12,18,24]),
            int(np.random.normal(proj["monthly_volume"]*0.9, proj["monthly_volume"]*0.2)),
            random.choice([6,7,8,9,10,12,14]),
            random.choice([96,97,98,99]),
            random.choice(["yes","yes","no"]),
            round(max(0.0, np.random.normal(1.0,0.8)),2),
            round(max(0.0, np.random.normal(0.8,0.6)),2),
            "300","ml", random.choice(["UK","EU","PL","ES","DE","FR","IE"]),
            random.choice(["Vegan","Vegan|Cruelty Free","-","FSC packaging"]),
            random.choice(["none","nuts","milk","-"]),
            random.choice(["-","BUNDLE-HAIR","BUNDLE-ORAL","BUNDLE-SKIN"]),
            round(np.random.uniform(0.0, 4.0),1),
            random.choice(["Bundle available","Stable supply","Winter LT risk","Freight included","Rebate tied to volume"])
        ])
    bids = pd.DataFrame(rows, columns=[
        "tender_id","supplier","sku_code","offer_price","contract_months","min_volume_units_per_month",
        "lead_time_days","service_level_target_pct","freight_included","rebate_pct","promo_funding_pct",
        "pack_size","uom","country_of_origin","certifications","allergens_present",
        "bundle_offer_reference","bundle_discount_pct","notes"
    ])
    # Apply bundle discount to offer_price where applicable
    mask = bids["bundle_offer_reference"].fillna("-").astype(str).str.strip().ne("-")
    bids.loc[mask, "offer_price"] = bids.loc[mask, "offer_price"] * (1 - bids.loc[mask, "bundle_discount_pct"]/100.0)
    return bids

def bids_step():
    st.subheader("2️⃣ Tender Launch — Collect Supplier Bids")
    proj = st.session_state.sourcing["project"]
    if not proj:
        st.warning("Define the project scope first."); set_sourcing_step("Scope"); return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Upload supplier bids (CSV)**")
        up = st.file_uploader("Choose CSV file", type=["csv"], key="bids_csv")
    with c2:
        st.markdown("**Or generate sample bids**")
        gen = st.button("Generate realistic sample bids")

    if up:
        bids = pd.read_csv(up)
        st.success(f"Loaded {len(bids)} bids from file.")
    elif gen:
        tid = str(uuid.uuid4())[:8].upper()
        bids = generate_sample_bids(tid, proj)
        st.success(f"Generated {len(bids)} sample bids. Tender ID: {tid}")
    else:
        bids = st.session_state.sourcing.get("bids")

    if bids is not None:
        st.session_state.sourcing["bids"] = bids
        st.dataframe(bids, use_container_width=True)

    # Wizard navigation
    colL, colR = st.columns([1,1])
    with colL:
        st.button("← Back: Scope", use_container_width=True, on_click=lambda: set_sourcing_step("Scope"))
    with colR:
        st.button("Next: AI Review →", type="primary", use_container_width=True,
                  on_click=lambda: set_sourcing_step("Review"))

def review_step():
    st.subheader("3️⃣ Rebot — Ranking, Rationale & Scenarios")
    bids = st.session_state.sourcing.get("bids")
    proj = st.session_state.sourcing["project"]
    if bids is None or len(bids)==0:
        st.warning("No bids loaded yet."); set_sourcing_step("Bids"); return

    # Weighting sliders (auto-balance)
    w = st.session_state.sourcing["weights"]
    wc, wv, ws, wk = st.columns(4)
    w["price"]    = wc.slider("Cost %",      0, 100, w["price"])
    w["volume"]   = wv.slider("Volume %",    0, 100, w["volume"])
    w["service"]  = ws.slider("Service %",   0, 100, w["service"])
    w["contract"] = wk.slider("Contract %",  0, 100, w["contract"])
    # auto-balance to sum 100 by nudging "contract"
    total = sum([w["price"], w["volume"], w["service"], w["contract"]])
    if total != 100:
        diff = total - 100
        w["contract"] = max(0, min(100, w["contract"] - diff))
    nw = dict(zip(["price","volume","service","contract"], normalized_weights(w["price"],w["volume"],w["service"],w["contract"])))

    # Score
    offers = bids.copy()
    scores, effs, whys = [], [], []
    for _, r in offers.iterrows():
        s, eff = score_bid(r, w)  # use raw weights for penalties/bonus; normalized used implicitly via components
        scores.append(s); effs.append(eff); whys.append(ai_why(r, eff, nw))
    offers["effective_cost"] = effs
    offers["score"] = scores
    offers["why"] = whys
    ranked = offers.sort_values("score", ascending=False).reset_index(drop=True)
    st.dataframe(ranked, use_container_width=True)
    st.session_state.sourcing["ranked"] = ranked

    # AI reasoning panel for the top few
    st.markdown("#### 🔎 Rebot Review (top offers)")
    topN = min(3, len(ranked))
    for i in range(topN):
        r = ranked.iloc[i]
        with st.expander(f"{i+1}. {r['supplier']} — {r['sku_code']} (score {r['score']:.3f})", expanded=(i==0)):
            st.write(r["why"])
            st.json({
                "effective_cost": r["effective_cost"],
                "offer_price": float(r["offer_price"]),
                "rebate_pct": float(r.get("rebate_pct",0.0)),
                "promo_funding_pct": float(r.get("promo_funding_pct",0.0)),
                "volume_capacity": int(r.get("min_volume_units_per_month",0)),
                "service_level_target_pct": int(r.get("service_level_target_pct",0)),
                "contract_months": int(r.get("contract_months",0)),
                "freight_included": str(r.get("freight_included",""))
            })

    # Alternative scenario (quick nudge of cost +10%)
    st.markdown("#### 🔁 Alternative scenario")
    if st.button("Increase Cost weight by +10% (auto-normalize)"):
        wc2 = w["price"] + 10
        alt_w = dict(zip(["price","volume","service","contract"], normalized_weights(wc2, w["volume"], w["service"], w["contract"])))
        alt_scores=[]
        for _, r in offers.iterrows():
            sw = {"price": alt_w["price"]*100, "volume": alt_w["volume"]*100, "service": alt_w["service"]*100, "contract": alt_w["contract"]*100}
            s2, _ = score_bid(r, sw)
            alt_scores.append(s2)
        comp = offers[["supplier","sku_code","score"]].copy()
        comp["alt_score"] = alt_scores
        comp["Δ"] = comp["alt_score"] - comp["score"]
        st.dataframe(comp.sort_values("alt_score", ascending=False).reset_index(drop=True), use_container_width=True)

    # Wizard nav
    colL, colR = st.columns([1,1])
    with colL:
        st.button("← Back: Bids", use_container_width=True, on_click=lambda: set_sourcing_step("Bids"))
    with colR:
        st.button("Next: Award →", type="primary", use_container_width=True,
                  on_click=lambda: set_sourcing_step("Award"))

def award_step():
    st.subheader("4️⃣ Award Contract")
    ranked = st.session_state.sourcing.get("ranked")
    proj = st.session_state.sourcing["project"]
    if ranked is None or ranked.empty:
        st.warning("Run AI review first."); set_sourcing_step("Review"); return

    options = [f"{i+1}. {row['supplier']} — {row['sku_code']} @ £{row['offer_price']:.2f}" for i, row in ranked.iterrows()]
    idx = st.selectbox("Choose an offer to award", list(range(len(options))), format_func=lambda i: options[i])
    choice = ranked.iloc[int(idx)]

    c1,c2 = st.columns(2)
    with c1:
        st.write("**Award summary**")
        st.json({
            "product_name": proj.get("product_name"),
            "winner": choice["supplier"],
            "sku_code": str(choice["sku_code"]),
            "offer_price": float(choice["offer_price"]),
            "effective_cost": float(choice["effective_cost"]),
            "score": float(choice["score"]),
            "contract_months": int(choice.get("contract_months",12))
        })
    with c2:
        st.write("**Why this choice**")
        st.write(choice["why"])

    if st.button("🏆 Confirm & Award"):
        prev_cost = proj.get("target_price", None)
        award = dict(
            tender_id=str(uuid.uuid4())[:8].upper(),
            awarded_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            supplier=choice["supplier"],
            sku_code=str(choice["sku_code"]),
            offer_price=float(choice["offer_price"]),
            contract_months=int(choice.get("contract_months",12)),
            score=float(choice["score"]),
            previous_cost=float(prev_cost) if prev_cost is not None else None
        )
        st.session_state.awards.append(award)
        add_notification(f"Awarded {award['sku_code']} to {award['supplier']} @ £{award['offer_price']:.2f}")
        st.success("Award completed! Added to Awards dashboard.")
        # Reset wizard to Scope for next project
        st.session_state.sourcing["step"] = "Scope"

    colL, colR = st.columns([1,1])
    with colL:
        st.button("← Back: Review", use_container_width=True, on_click=lambda: set_sourcing_step("Review"))
    with colR:
        st.button("Go to Awards Dashboard →", type="primary", use_container_width=True,
                  on_click=lambda: set_view("Awards"))

# ==========================================================
# AWARDS DASHBOARD (Workflow 3)
# ==========================================================
def awards_page():
    top_bar_and_nav()
    # Preload historical awards if no live awards
    if len(st.session_state.awards)==0 and AWARDS_HIST is not None:
        st.session_state.awards = AWARDS_HIST.to_dict("records")

    if len(st.session_state.awards)==0:
        st.info("No awards yet. Complete a sourcing project to see analytics.")
        return

    df = pd.DataFrame(st.session_state.awards).copy()
    st.markdown("### Awarded Contracts")
    st.dataframe(df.sort_values("awarded_at", ascending=False), use_container_width=True)

    # KPIs
    if "previous_cost" in df.columns:
        df["savings"] = df.apply(lambda r: (r["previous_cost"] - r["offer_price"]) if pd.notna(r["previous_cost"]) else np.nan, axis=1)
    else:
        df["savings"] = np.nan
    df["savings_pct"] = df["savings"] / df["previous_cost"] * 100.0 if "previous_cost" in df.columns else np.nan

    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(f"<div class='kpi'><div class='badge good'>Total Awards</div><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='kpi'><div class='badge good'>Avg £ Saved</div><h2>£{pd.to_numeric(df['savings']).dropna().mean():.2f}</h2></div>", unsafe_allow_html=True)
    with c3: 
        mean_pct = pd.to_numeric(df["savings_pct"]).dropna().mean()
        st.markdown(f"<div class='kpi'><div class='badge warn'>Avg % Saved</div><h2>{(mean_pct if not np.isnan(mean_pct) else 0):.1f}%</h2></div>", unsafe_allow_html=True)

    # Charts
    st.markdown("### Analytics")
    # Savings trend
    tmp = df.copy()
    if "awarded_at" in tmp.columns:
        tmp["awarded_at"] = pd.to_datetime(tmp["awarded_at"], errors="coerce")
    else:
        tmp["awarded_at"] = pd.Timestamp.utcnow()
    tmp["month"] = tmp["awarded_at"].dt.to_period("M").astype(str)
    st.markdown("**Savings over time**")
    st.line_chart(tmp.groupby("month")["savings"].sum().fillna(0))

    # Supplier heatmap (bar of mean savings)
    st.markdown("**Supplier performance (avg savings)**")
    key_col = "supplier" if "supplier" in df.columns else ("winner" if "winner" in df.columns else None)
    if key_col:
        st.bar_chart(df.groupby(key_col)["savings"].mean().fillna(0))

    # Contract expiry projection (synthetic)
    st.markdown("**Contract expiry projection (12 months)**")
    expiry = pd.DataFrame({
        "month": pd.date_range(datetime.utcnow(), periods=12, freq="MS"),
        "expiring_contracts": np.random.randint(1, 12, 12)
    })
    st.area_chart(expiry.set_index("month"))

# ==========================================================
# ROUTER
# ==========================================================
def router():
    v = st.session_state.view
    if v == "Landing":   landing()
    elif v == "Portfolio": portfolio_page()
    elif v == "Sourcing":  sourcing_page()
    elif v == "Awards":    awards_page()
    else: landing()

# ==========================================================
# ENTRYPOINT
# ==========================================================
notifications_panel()
router()

# ==========================================================
# MOCK ANALYTICS BACKUP (if no awards CSV and no live awards)
# ==========================================================
if (AWARDS_HIST is None) and (len(st.session_state.awards)==0):
    st.markdown("---")
    st.caption("ℹ️ Demo mode — showing mock analytics.")
    cats = ["Haircare","Oral Care","Snacks","Dairy","Skincare"]
    st.bar_chart(pd.DataFrame({"avg_cost":np.random.uniform(0.8,3.0,5)}, index=cats))
    months = pd.date_range(datetime.now()-timedelta(days=360), periods=12, freq="MS")
    st.line_chart(pd.Series(np.random.uniform(2000,8000,12), index=months))
