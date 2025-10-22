import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid

st.set_page_config(page_title="RELEX BS", layout="wide")

# Utilities
def clean_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def redflag_badges(row):
    flags = []
    if str(row.get("freight_included","yes")).lower() not in ["yes","true","1","y"]:
        flags.append("No Freight")
    if row.get("service_level_target_pct", 100) < 96:
        flags.append("SL<96%")
    if row.get("lead_time_days", 7) > 12:
        flags.append("Lead>12d")
    if row.get("contract_months", 12) < 12:
        flags.append("Short Contract")
    return ", ".join(flags) if flags else "OK"

def score_bid(row, weights):
    eff = float(row["offer_price"])
    eff -= (float(row.get("rebate_pct", 0.0))/100.0) * float(row["offer_price"])
    eff -= (float(row.get("promo_funding_pct", 0.0))/100.0) * float(row["offer_price"])
    if str(row.get("freight_included","yes")).lower() not in ["yes","true","1","y"]:
        eff += float(weights.get("freight_penalty_per_unit", 0.03))
    sl = float(row.get("service_level_target_pct", 98))
    lt = float(row.get("lead_time_days", 7))
    eff += max(0.0, 98.0 - sl) * float(weights.get("service_level_penalty_per_pp", 0.001))
    eff += max(0.0, lt - 7.0) * float(weights.get("lead_time_penalty_per_day", 0.002))
    contract = float(row.get("contract_months", 12))
    eff -= (contract/24.0) * float(weights.get("contract_length_bonus_per_unit", 0.01))
    components = {
        "base_price": float(row["offer_price"]),
        "rebate_value": (float(row.get("rebate_pct",0.0))/100.0)*float(row["offer_price"]),
        "promo_value": (float(row.get("promo_funding_pct",0.0))/100.0)*float(row["offer_price"]),
        "freight_penalty": float(weights.get("freight_penalty_per_unit",0.03)) if str(row.get("freight_included","yes")).lower() not in ["yes","true","1","y"] else 0.0,
        "service_penalty": max(0.0, 98.0 - sl) * float(weights.get("service_level_penalty_per_pp", 0.001)),
        "lead_penalty": max(0.0, lt - 7.0) * float(weights.get("lead_time_penalty_per_day", 0.002)),
        "contract_bonus": (contract/24.0) * float(weights.get("contract_length_bonus_per_unit", 0.01)),
    }
    cost_component = 1.0 / max(eff, 1e-6)
    vol_component = np.log1p(float(row.get("min_volume_units_per_month", 0.0)))
    service_component = sl / 100.0
    score = (
        float(weights.get("w_cost", 0.55)) * cost_component +
        float(weights.get("w_volume", 0.15)) * vol_component +
        float(weights.get("w_service", 0.20)) * service_component +
        float(weights.get("w_contract", 0.10)) * (contract / 24.0)
    )
    return float(score), float(eff), components

def why_chosen(row, components, weights):
    reasons = []
    effective = components['base_price'] - components['rebate_value'] - components['promo_value'] + components['freight_penalty'] + components['service_penalty'] + components['lead_penalty'] - components['contract_bonus']
    reasons.append(f"Effective unit cost estimated at **{effective:.2f}**.")
    if weights.get("w_cost",0.55) >= 0.5:
        reasons.append("Cost weight is high, so offers with **lower effective cost** rank higher.")
    if float(row.get("min_volume_units_per_month",0)) >= 50000:
        reasons.append("Supplier pledged **strong monthly volume**, improving ranking.")
    if float(row.get("service_level_target_pct",0)) >= 97:
        reasons.append("High **OTIF target** reduces supply risk.")
    if float(row.get("contract_months",0)) >= 12:
        reasons.append("**Longer contract** duration improves score via stability bonus.")
    if components['freight_penalty']>0:
        reasons.append("Penalty applied because **freight not included**.")
    if components['contract_bonus']>0.0:
        reasons.append("Bonus applied for **longer contract commitment**.")
    return " ".join(reasons)

# Session init
if "view" not in st.session_state:
    st.session_state.view = "Landing"
if "portfolio" not in st.session_state:
    try:
        st.session_state.portfolio = pd.read_csv("sample_portfolio.csv")
    except Exception:
        st.session_state.portfolio = pd.DataFrame(columns=[
            "sku_code","product_name","category","current_supplier","current_unit_cost",
            "monthly_volume_units","contract_end","last_tender_date","pack_size","uom","target_margin_pct","ai_insight"
        ])
if "projects" not in st.session_state:
    st.session_state.projects = {}
if "awards" not in st.session_state:
    st.session_state.awards = []

# Landing
def landing():
    st.title("Retail Tender Portal — Plus")
    st.caption("Portfolio Intelligence + SKU Sourcing with AI rationale and awarding.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("## SKU Portfolio")
        st.caption("See all SKUs, suppliers, costs, contract ends, AI insights. Kick off re-sourcing or terminate contracts.")
        if st.button("Open SKU Portfolio ▶️", use_container_width=True):
            st.session_state.view = "Portfolio"
    with c2:
        st.markdown("## SKU Sourcing")
        st.caption("Set scope & scoring in Project Setup, publish RFPs, ingest bids, AI explain picks, award contracts.")
        if st.button("Open SKU Sourcing ▶️", use_container_width=True):
            st.session_state.view = "Sourcing"

# Portfolio
def portfolio_page():
    st.title("SKU Portfolio")
    df = st.session_state.portfolio.copy()
    st.markdown("#### Current SKUs")
    st.dataframe(df, use_container_width=True)

    st.markdown("#### Actions")
    colA, colB, colC = st.columns(3)
    with colA:
        if not df.empty:
            sku = st.selectbox("Select SKU to re-source", sorted(df["sku_code"].unique()))
            if st.button("Start Re-Sourcing from SKU", use_container_width=True):
                sku_row = df[df["sku_code"]==sku].iloc[0]
                st.session_state.prefill_scope = dict(
                    product_name=sku_row["product_name"],
                    target_price=float(sku_row["current_unit_cost"]) * 0.95,
                    contract_months=12,
                    monthly_volume_units=int(sku_row["monthly_volume_units"]),
                    pack_size=str(sku_row["pack_size"]), uom=str(sku_row["uom"]),
                    must_have_ingredients="", nice_to_have_ingredients="", must_not_have_ingredients="",
                    allergen_constraints="", certifications_required="",
                    origin_preference="", sustainability_requirements="",
                    packaging_type="", brand_type="Either", exclusivity_required="No",
                    max_lead_time_days=10, service_level_target_pct=97,
                    promo_support_expected_pct=1.0, rebate_target_pct=1.0, bundling_interest="Yes"
                )
                st.success(f"Scope prefilled from {sku}.")
                st.session_state.view = "Sourcing"
    with colB:
        if not df.empty:
            sku_t = st.selectbox("Select SKU to terminate", sorted(df["sku_code"].unique()))
            if st.button("Terminate Contract", use_container_width=True):
                st.session_state.portfolio.loc[st.session_state.portfolio["sku_code"]==sku_t,"contract_end"] = datetime.utcnow().date().isoformat()
                st.warning(f"Contract for {sku_t} set to end today (prototype).")
    with colC:
        if st.button("Back to Landing", use_container_width=True):
            st.session_state.view = "Landing"

# Sourcing
def sourcing_page():
    st.title("SKU Sourcing")
    tabs = st.tabs(["📋 Project Setup", "📥 Bids", "🤖 Ranking & Why", "🏆 Award"])

    with tabs[0]:
        st.markdown("#### Define Scope and Scoring")
        pf = st.session_state.get("prefill_scope", {})
        colA, colB = st.columns(2)
        with colA:
            product_name = st.text_input("Product Name", pf.get("product_name","Hydrating Shampoo 300ml"))
            target_price = st.number_input("Target Price (per unit)", 0.0, 100.0, float(pf.get("target_price", 2.50)), 0.05)
            contract_months = st.number_input("Contract Length (months)", 1, 60, int(pf.get("contract_months", 12)))
            monthly_volume_units = st.number_input("Monthly Volume (units)", 0, 10_000_000, int(pf.get("monthly_volume_units", 50000)), 1000)
            pack_size = st.text_input("Pack Size", pf.get("pack_size","300"))
            uom = st.selectbox("Unit of Measure", ["ml","g","kg","L","units"], index=0)
            max_lead_time_days = st.number_input("Max Lead Time (days)", 1, 120, int(pf.get("max_lead_time_days", 10)))
            service_level_target_pct = st.slider("Service Level Target (OTIF %)", 90, 100, int(pf.get("service_level_target_pct", 97)))
        with colB:
            must_have = st.text_area("Must-Have Ingredients (comma-separated)", pf.get("must_have_ingredients","aloe vera, pro-vitamin B5"))
            nice_to_have = st.text_area("Nice-to-Have Ingredients", pf.get("nice_to_have_ingredients","argan oil"))
            must_not_have = st.text_area("Must-Not-Have Ingredients", pf.get("must_not_have_ingredients","sulfates, parabens"))
            allergens = st.text_input("Allergen Constraints", pf.get("allergen_constraints","none"))
            certifications = st.text_input("Certifications Required", pf.get("certifications_required","Vegan, Cruelty Free"))
            origin_pref = st.text_input("Origin Preference", pf.get("origin_preference","EU or UK"))
            sustainability = st.text_input("Sustainability Requirements", pf.get("sustainability_requirements","PCR packaging ≥30%"))
            packaging = st.text_input("Packaging Type", pf.get("packaging_type","Recyclable bottle with flip-cap"))
            brand_type = st.selectbox("Brand Type", ["Private Label","Branded","Either"], index=2)
            exclusivity = st.selectbox("Exclusivity Required", ["No","Yes"], index=0)

        st.markdown("**Scoring Weights**")
        wc1, wc2, wc3, wc4 = st.columns(4)
        w_cost = wc1.slider("Weight: Cost", 0.0, 1.0, 0.55, 0.05)
        w_vol = wc2.slider("Weight: Volume", 0.0, 1.0, 0.15, 0.05)
        w_serv = wc3.slider("Weight: Service", 0.0, 1.0, 0.20, 0.05)
        w_con = wc4.slider("Weight: Contract", 0.0, 1.0, 0.10, 0.05)
        weights = dict(
            w_cost=w_cost, w_volume=w_vol, w_service=w_serv, w_contract=w_con,
            freight_penalty_per_unit=0.03, service_level_penalty_per_pp=0.001,
            lead_time_penalty_per_day=0.002, contract_length_bonus_per_unit=0.01
        )

        if st.button("Publish RFP ➕"):
            tender_id = str(uuid.uuid4())[:8].upper()
            st.session_state.projects[tender_id] = dict(
                tender_id=tender_id,
                scope=dict(
                    product_name=product_name, target_price=target_price, contract_months=contract_months,
                    monthly_volume_units=monthly_volume_units, pack_size=pack_size, uom=uom,
                    must_have_ingredients=must_have, nice_to_have_ingredients=nice_to_have, must_not_have_ingredients=must_not_have,
                    allergen_constraints=allergens, certifications_required=certifications, origin_preference=origin_pref,
                    sustainability_requirements=sustainability, packaging_type=packaging, brand_type=brand_type,
                    exclusivity_required=exclusivity, max_lead_time_days=max_lead_time_days, service_level_target_pct=service_level_target_pct
                ),
                weights=weights,
                status="Open"
            )
            st.session_state.active_tender = tender_id
            st.success(f"Published RFP — Tender ID: {tender_id}")

        if "active_tender" in st.session_state:
            tid = st.session_state.active_tender
            proj = st.session_state.projects[tid]
            st.markdown("##### Active Project")
            st.json(proj)

    with tabs[1]:
        if "active_tender" not in st.session_state:
            st.info("Create and publish an RFP in Project Setup first.")
        else:
            tid = st.session_state.active_tender
            st.markdown(f"#### Bids for Tender {tid}")
            up = st.file_uploader("Upload SKU Bids CSV", type=["csv"], key="sku_bids")
            if up:
                bids = pd.read_csv(up)
            else:
                bids = pd.DataFrame([
                    [tid,"Own Label Co.","PL-SHAM-300",2.10,12,52000,8,98,"yes",2.0,0.0,"300","ml","UK","Vegan|Cruelty Free","none","BUNDLE-HAIR",3.0,"Bundle with conditioner"],
                    [tid,"Unilever","UNI-SHAM-300",2.65,12,50000,7,98,"yes",1.0,1.5,"300","ml","EU","Vegan","none","BUNDLE-HAIR",2.0,"Bundle with conditioner for extra discount"],
                    [tid,"P&G","PNG-SHAM-300",2.80,18,45000,9,97,"no",0.5,1.0,"300","ml","EU","Vegan","none","—",0.0,"Lead time spike in winter"],
                ], columns=["tender_id","supplier","sku_code","offer_price","contract_months","min_volume_units_per_month",
                            "lead_time_days","service_level_target_pct","freight_included","rebate_pct","promo_funding_pct",
                            "pack_size","uom","country_of_origin","certifications","allergens_present",
                            "bundle_offer_reference","bundle_discount_pct","notes"])
            bids = clean_numeric(bids, ["offer_price","contract_months","min_volume_units_per_month","lead_time_days","service_level_target_pct","rebate_pct","promo_funding_pct","bundle_discount_pct"])
            st.session_state.latest_bids = bids
            st.dataframe(bids, use_container_width=True)

    with tabs[2]:
        if "active_tender" not in st.session_state or "latest_bids" not in st.session_state:
            st.info("Publish a project and upload bids first.")
        else:
            tid = st.session_state.active_tender
            proj = st.session_state.projects[tid]
            weights = proj["weights"]
            bids = st.session_state.latest_bids.copy()
            if "bundle_offer_reference" in bids.columns and "bundle_discount_pct" in bids.columns:
                bids["bundle_offer_reference"] = bids["bundle_offer_reference"].fillna("").astype(str).str.strip()
                mask = bids["bundle_offer_reference"].str.lower().isin(["—","-","none","nan",""])
                bids.loc[~mask, "offer_price"] = bids.loc[~mask, "offer_price"] * (1.0 - (bids.loc[~mask, "bundle_discount_pct"] / 100.0))
            scores, effs, expls, flags = [], [], [], []
            for _, r in bids.iterrows():
                s, e, comps = score_bid(r, weights)
                scores.append(s); effs.append(e); flags.append(redflag_badges(r))
                expls.append(why_chosen(r, comps, weights))
            bids["effective_cost"] = effs
            bids["flags"] = flags
            bids["score"] = scores
            bids["why_this"] = expls
            ranked = bids.sort_values("score", ascending=False).reset_index(drop=True)
            st.markdown("#### Ranked Offers")
            st.dataframe(ranked, use_container_width=True)
            st.markdown("##### Why did we choose this? (Top Offer)")
            if not ranked.empty:
                top = ranked.iloc[0]
                st.info(f"**{top['supplier']} — {top['sku_code']}**: {top['why_this']}")
            st.session_state.ranked_bids = ranked

    with tabs[3]:
        if "ranked_bids" not in st.session_state:
            st.info("Generate rankings first in the previous tab.")
        else:
            ranked = st.session_state.ranked_bids
            st.markdown("#### Select Offer to Award")
            options = [f"{i+1}. {row['supplier']} — {row['sku_code']} @ {row['offer_price']:.2f}\" for i, row in ranked.iterrows()]
            idx = st.selectbox("Choose an offer", list(range(len(options))), format_func=lambda i: options[i])
            if st.button("Award Contract 🏆"):
                win = ranked.iloc[int(idx)]
                st.success(f"Awarded to **{win['supplier']}** for **{win['sku_code']}**.")
                tid = st.session_state.active_tender
                st.session_state.projects[tid]["status"] = "Awarded"
                st.session_state.projects[tid]["award"] = dict(supplier=win["supplier"], sku_code=win["sku_code"], offer_price=float(win["offer_price"]), contract_months=int(win["contract_months"]))
                st.session_state.awards.append(dict(tender_id=tid, **st.session_state.projects[tid]["award"]))
                sku_code = str(win["sku_code"])
                end_date = (datetime.utcnow() + timedelta(days=30*int(win["contract_months"]))).date().isoformat()
                if "portfolio" in st.session_state and not st.session_state.portfolio.empty and sku_code in set(st.session_state.portfolio["sku_code"]):
                    st.session_state.portfolio.loc[st.session_state.portfolio["sku_code"]==sku_code,["current_supplier","current_unit_cost","contract_end","last_tender_date"]] = [win["supplier"], float(win["offer_price"]), end_date, datetime.utcnow().date().isoformat()]
                else:
                    scope = st.session_state.projects[tid]["scope"]
                    new_row = dict(
                        sku_code=sku_code, product_name=scope["product_name"], category="Sourced", current_supplier=win["supplier"],
                        current_unit_cost=float(win["offer_price"]), monthly_volume_units=int(scope["monthly_volume_units"]),
                        contract_end=end_date, last_tender_date=datetime.utcnow().date().isoformat(),
                        pack_size=scope["pack_size"], uom=scope["uom"], target_margin_pct=22.0, ai_insight="Awarded recently"
                    )
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, pd.DataFrame([new_row])], ignore_index=True)
                st.balloons()

            st.markdown("#### Project Status")
            if "active_tender" in st.session_state:
                tid = st.session_state.active_tender
                st.json(st.session_state.projects[tid])

# Router
if st.session_state.view == "Landing":
    landing()
elif st.session_state.view == "Portfolio":
    portfolio_page()
elif st.session_state.view == "Sourcing":
    sourcing_page()

