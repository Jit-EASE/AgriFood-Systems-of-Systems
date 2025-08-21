# ==========================================================
# Agri-Food System-of-Systems: Offline AI Prototypes
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# -------- Optional deps (graceful fallbacks) ----------------
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

try:
    from skimage import color, feature, morphology, measure, exposure
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_infl_data import variance_inflation_factor  # older alias
except Exception:
    # if alias not found, try modern imports
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except Exception:
        sm = None
        variance_inflation_factor = None
STATSMODELS_OK = sm is not None

try:
    from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad
    DIAG_OK = True
except Exception:
    DIAG_OK = False

try:
    from graphviz import Digraph
    GRAPHVIZ_OK = True
except Exception:
    GRAPHVIZ_OK = False

try:
    # Interactive network (preferred)
    from pyvis.network import Network
    import tempfile, os
    PYVIS_OK = True
except Exception:
    PYVIS_OK = False

# ==========================================================
# Reference text used by contextual agent
# ==========================================================
IRELAND_REFS = [
    "DAFM AIM (livestock traceability) - national data spine.",
    "Teagasc PastureBase - grassland DSS; ideal integration for pasture metrics.",
    "Bord Bia Origin Green - sustainability & audit; supports welfare/traceability claims.",
    "SFI VistaMilk - research-to-industry pathway for dairy digitisation.",
    "GS1 standards - IDs/QRs as glue for supply-chain interoperability."
]

PRINCIPLES = [
    "Offline-by-design reduces privacy risk and adoption friction in low-connectivity areas.",
    "Prototype -> validate on research farms -> scale via advisory network (Teagasc model).",
    "Always quantify ROI and payback time for SMEs; keep UX minimal and farmer-friendly.",
    "Prefer open outputs that can be consumed by public spines and private platforms."
]

def contextual_agent(section, state):
    bullets = []
    if section == "adoption":
        bullets = [
            "Address SME friction with a one-screen workflow and clear ROI.",
            "Avoid vendor lock-in: export results in standard CSV/JSON.",
            "Leverage Teagasc advisors for training and trust-building."
        ] + PRINCIPLES
    elif section == "crack":
        csi = state.get("csi")
        bullets = [
            "Use clear thresholds to triage inspections (low / moderate / high).",
            "Capture before/after photos to build on-farm trust without cloud storage.",
            "Offer a one-page PDF result for auditors if needed (local only)."
        ]
        if csi is not None:
            if csi < 20:
                bullets.append("CSI is low -> routine monitoring; log timestamp and location.")
            elif csi < 50:
                bullets.append("CSI moderate -> schedule manual inspection; consider preventive maintenance.")
            else:
                bullets.append("CSI high -> prioritise immediate inspection; mitigate to avoid loss.")
        bullets += PRINCIPLES
    elif section == "roi":
        nb  = state.get("net_benefit")
        pay = state.get("payback_months")
        npv = state.get("npv")
        sims = state.get("sims")
        bullets = [
            "Target payback under 12–18 months for SME adoption.",
            "Bundle with existing inspections to avoid extra labour.",
            "Report avoided-losses in terms co-ops already track (waste, downtime)."
        ]
        if nb is not None:
            if nb <= 0:
                bullets.append("Current assumptions show ≤ 0 net benefit -> raise detection effectiveness or lower licence/maintenance costs.")
            else:
                bullets.append(f"Annual net benefit ≈ EUR {nb:,.0f}.")
        if pay is not None:
            bullets.append(f"Estimated payback ~ {pay} months under current inputs.")
        if npv is not None:
            bullets.append(f"Horizon NPV ≈ EUR {npv:,.0f} (discounted).")
        if isinstance(sims, np.ndarray) and sims.size > 0:
            pos = (sims > 0).mean() * 100
            med = float(np.median(sims))
            bullets.append(f"Monte Carlo: {pos:.1f}% probability of positive NPV; median NPV ≈ EUR {med:,.0f}.")
        bullets += PRINCIPLES
    elif section == "ols":
        r2 = state.get("r2")
        bullets = [
            "Use HC3 robust SE when heteroskedasticity is suspected.",
            "Show residual diagnostics and Cook's distance before inference.",
            "Clearly state: prototype results merit validation on Irish datasets."
        ]
        if r2 is not None:
            bullets.append(f"Model R² ≈ {r2:.2f}; interpret within behavioural/social science norms if applicable.")
        bullets += PRINCIPLES
    elif section == "system":
        bullets = [
            "Position the AI as a reliability layer, not a new silo.",
            "Expose open outputs (CSV/JSON) to plug into AIM, PastureBase, Origin Green.",
            "Design phased rollout: single farm -> research farm network -> co-op scale."
        ] + PRINCIPLES
    elif section == "compliance":
        bullets = [
            "Keep human-in-the-loop; outputs advisory, not punitive.",
            "Document model scope/limits; avoid biometric storage.",
            "Provide opt-in/opt-out and local data ownership."
        ] + PRINCIPLES
    else:
        bullets = PRINCIPLES
    return {"bullets": bullets, "references": IRELAND_REFS}

# ==========================================================
# Page
# ==========================================================
st.set_page_config(page_title="Agri-Food System-of-Systems: Offline AI Prototypes", page_icon="🌾", layout="wide")
st.title("Agri-Food System-of-Systems: Offline AI Prototypes")
st.caption("Ireland | Teagasc-aligned | Prototypes — merits further validation")
st.write("Purpose. This app presents offline, contextual AI prototypes for Irish agri-food supply chains. All local, no external APIs.")

# Keep original tabs
tabs = st.tabs(["Adoption", "Crack/Stress AI", "ROI", "OLS", "System Map", "Compliance"])

# ==========================================================
# TAB 1 — Adoption (unchanged)
# ==========================================================
with tabs[0]:
    st.subheader("Adoption & Best Practice (Ireland)")
    st.write("- Data silos and poor interoperability")
    st.write("- SME adoption friction and unclear ROI")
    st.write("- Connectivity constraints in rural areas")
    st.write("- Trust & governance (GDPR / EU AI Act)")
    st.write("Best practices: offline edge AI, open outputs, clear ROI, Teagasc validation pipeline.")
    with st.expander("Contextual Agent (offline)"):
        ag = contextual_agent("adoption", {})
        st.markdown("- " + "\n- ".join(ag["bullets"]))
        st.caption("References: " + " | ".join(ag["references"]))

# ==========================================================
# TAB 2 — Crack/Stress AI (original logic retained)
# ==========================================================
with tabs[1]:
    st.subheader("Crack/Stress AI (Offline Prototype)")

    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    def simple_csi(image_array: np.ndarray):
        if SKIMAGE_OK:
            gray = color.rgb2gray(image_array)
            eq = exposure.equalize_adapthist(gray, clip_limit=0.03)
            edges = feature.canny(eq, sigma=1.6)
            thin = morphology.thin(edges)
            labeled = measure.label(thin, connectivity=2)
            props = measure.regionprops(labeled)
            total_len = 0.0
            elong_bonus = 0.0
            for p in props:
                y0, x0, y1, x1 = p.bbox
                h = max(1, y1 - y0); w = max(1, x1 - x0)
                elong = max(h, w) / max(1.0, min(h, w))
                total_len += p.perimeter
                if elong > 1.0:
                    elong_bonus += min(elong - 1.0, 20.0)
            edge_density = edges.mean()
            raw = 0.6*(edge_density*100.0) + 0.4*(np.log1p(total_len + elong_bonus))
            csi = float(np.clip(raw, 0, 100))
            return {"csi": csi, "edge_density": float(edge_density), "segments": len(props)}
        else:
            arr = image_array.astype(np.float32)/255.0
            gx = np.abs(np.gradient(arr, axis=0)).mean()
            gy = np.abs(np.gradient(arr, axis=1)).mean()
            csi = float(np.clip((gx + gy) * 100, 0, 100))
            return {"csi": csi, "edge_density": float((gx+gy)/2.0), "segments": 0}

    results = {}
    if uploaded is not None and PIL_OK:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input", use_container_width=True)
        np_img = np.array(img)
        results = simple_csi(np_img)
        csi = results["csi"]
        st.metric("Crack Severity Index (CSI)", f"{csi:.1f} / 100")
        st.progress(min(1.0, csi/100.0))
        if csi < 20:
            st.success("Low crack likelihood. Continue routine monitoring.")
        elif csi < 50:
            st.warning("Moderate crack-like features detected. Inspect manually and schedule maintenance.")
        else:
            st.error("High crack severity signal. Prioritise detailed inspection to avoid losses.")
        with st.expander("Diagnostics"):
            st.json(results)
    elif uploaded is not None and not PIL_OK:
        st.warning("PIL is unavailable; cannot render the image.")
    else:
        st.info("Upload a photo to run the demo.")

    with st.expander("Contextual Agent (offline)"):
        ag = contextual_agent("crack", {"csi": results.get("csi") if results else None})
        st.markdown("- " + "\n- ".join(ag["bullets"]))
        st.caption("References: " + " | ".join(ag["references"]))
    st.caption("Status: Prototype — merits further validation on Irish farm imagery with domain labels.")

# ==========================================================
# TAB 3 — ROI (enhanced)
# ==========================================================
with tabs[2]:
    st.subheader("ROI & Payback Simulator")

    col1, col2 = st.columns(2)
    with col1:
        asset_type = st.selectbox("Asset type", ["Silo", "Sprayer", "Barn/Parlour", "Milk Tank", "Other"])
        asset_value = st.number_input("Asset value (EUR)", 10000, 1_000_000, 60_000, step=5_000)
        annual_failure_prob = st.slider("Annual failure probability (baseline, %)", 0.0, 50.0, 8.0, 0.5)
        failure_loss = st.number_input("Loss per failure (EUR)", 1000, 200_000, 25_000, step=1000)
        years = st.slider("Time horizon (years)", 1, 15, 5)
        discount_rate = st.slider("Discount rate (%)", 0.0, 15.0, 5.0, 0.5) / 100
    with col2:
        detection_effectiveness = st.slider("AI detection effectiveness (%)", 0.0, 100.0, 55.0, 1.0)
        annual_maintenance = st.number_input("Annual maintenance with AI (EUR)", 0, 50_000, 3_000, step=500)
        ai_annual_cost = st.number_input("AI annual cost (EUR)", 0, 50_000, 1_800, step=100)
        training_cost = st.number_input("One-time onboarding (EUR)", 0, 50_000, 1_000, step=100)
        # New: indirect benefits
        insurance_saving = st.number_input("Insurance premium reduction (EUR/yr)", 0, 50_000, 2_000, step=500)
        waste_saving     = st.number_input("Food waste reduction (EUR/yr)", 0, 50_000, 1_500, step=500)
        recall_saving    = st.number_input("Fewer recalls & compliance (EUR/yr)", 0, 50_000, 2_500, step=500)

    # Original linear annual calc
    baseline_expected_loss = (annual_failure_prob/100.0) * failure_loss
    post_ai_loss = baseline_expected_loss * (1 - detection_effectiveness/100.0)
    annual_savings_direct = baseline_expected_loss - post_ai_loss
    annual_savings_total  = annual_savings_direct + insurance_saving + waste_saving + recall_saving
    annual_net_benefit    = annual_savings_total - ai_annual_cost - annual_maintenance
    payback_months = float("inf") if annual_net_benefit <= 0 else max(1, int(round(12 * (training_cost / max(annual_net_benefit, 1e-9)))))

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline expected loss / year", f"EUR {baseline_expected_loss:,.0f}")
    m2.metric("Expected savings / year", f"EUR {annual_savings_total:,.0f}")
    m3.metric("Net benefit / year", f"EUR {annual_net_benefit:,.0f}")
    st.progress(min(1.0, max(0.0, annual_savings_total / max(1.0, baseline_expected_loss))))
    st.metric("Estimated payback", "N/A" if np.isinf(payback_months) else f"{payback_months} months")

    # NPV over horizon
    npv = -training_cost
    for t in range(1, years+1):
        npv += annual_net_benefit / ((1 + discount_rate)**t)
    st.metric("NPV over horizon", f"EUR {npv:,.0f}")

    # Monte Carlo (uncertainty in detection effectiveness & failure prob)
    run_mc = st.checkbox("Run Monte Carlo sensitivity (1000 runs)", value=True)
    sims = None
    if run_mc:
        N = 1000
        eff = np.clip(np.random.normal(detection_effectiveness, 5, N), 0, 100) / 100.0
        failp = np.clip(np.random.normal(annual_failure_prob, 2, N), 0, 100) / 100.0
        sims = []
        for e, fp in zip(eff, failp):
            base_loss  = fp * failure_loss
            post_loss  = base_loss * (1 - e)
            save_dir   = base_loss - post_loss
            save_total = save_dir + insurance_saving + waste_saving + recall_saving
            net_ben    = save_total - ai_annual_cost - annual_maintenance
            # NPV for this path
            npv_i = -training_cost
            for t in range(1, years+1):
                npv_i += net_ben / ((1 + discount_rate)**t)
            sims.append(npv_i)
        sims = np.array(sims)

        # Histogram
        fig, ax = plt.subplots()
        ax.hist(sims, bins=30, color="skyblue", edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", label="NPV = 0")
        ax.set_title("Monte Carlo NPV Distribution")
        ax.set_xlabel("NPV (€)"); ax.set_ylabel("Frequency"); ax.legend()
        st.pyplot(fig)

        # CDF
        srt = np.sort(sims)
        cdf = np.arange(1, len(srt)+1) / len(srt)
        fig2, ax2 = plt.subplots()
        ax2.plot(srt, cdf)
        ax2.axvline(0, color="red", linestyle="--")
        ax2.set_title("Cumulative Probability (CDF) of NPV"); ax2.set_xlabel("NPV (€)"); ax2.set_ylabel("Probability")
        st.pyplot(fig2)

        # CSV download
        df_export = pd.DataFrame({"NPV": sims})
        buf = io.StringIO(); df_export.to_csv(buf, index=False)
        st.download_button("📥 Download Monte Carlo Results (CSV)", data=buf.getvalue(),
                           file_name="roi_montecarlo_results.csv", mime="text/csv")

        # Tornado chart (simple partial sensitivities)
        st.subheader("Tornado Sensitivity (what moves NPV most?)")
        # +/- 20% shocks on key drivers
        def shock_npv(d_eff=None, a_failp=None, maint=None, ai_cost=None, ind=None):
            _eff  = (d_eff   if d_eff  is not None else detection_effectiveness)/100.0
            _fp   = (a_failp if a_failp is not None else annual_failure_prob)/100.0
            _mnt  = (maint   if maint  is not None else annual_maintenance)
            _aic  = (ai_cost if ai_cost is not None else ai_annual_cost)
            _ind  = (ind     if ind    is not None else (insurance_saving + waste_saving + recall_saving))
            base_loss  = _fp * failure_loss
            post_loss  = base_loss * (1 - _eff)
            save_dir   = base_loss - post_loss
            save_total = save_dir + _ind
            net_ben    = save_total - _aic - _mnt
            _npv = -training_cost
            for t in range(1, years+1):
                _npv += net_ben / ((1 + discount_rate)**t)
            return _npv

        drivers = {
            "Detection effectiveness": ("d_eff", detection_effectiveness),
            "Failure probability": ("a_failp", annual_failure_prob),
            "Maintenance cost": ("maint", annual_maintenance),
            "AI licence cost": ("ai_cost", ai_annual_cost),
            "Indirect benefits": ("ind", insurance_saving + waste_saving + recall_saving),
        }
        lows, highs, labels = [], [], []
        for lbl, (key, base) in drivers.items():
            # +/- 20% (bounded where appropriate)
            if key in ("d_eff", "a_failp"):
                low_param  = max(0.0, base * 0.8)
                high_param = min(100.0, base * 1.2)
            else:
                low_param  = max(0.0, base * 0.8)
                high_param = base * 1.2
            if key == "d_eff":
                npv_low  = shock_npv(d_eff=low_param)
                npv_high = shock_npv(d_eff=high_param)
            elif key == "a_failp":
                npv_low  = shock_npv(a_failp=low_param)
                npv_high = shock_npv(a_failp=high_param)
            elif key == "maint":
                npv_low  = shock_npv(maint=low_param)
                npv_high = shock_npv(maint=high_param)
            elif key == "ai_cost":
                npv_low  = shock_npv(ai_cost=low_param)
                npv_high = shock_npv(ai_cost=high_param)
            elif key == "ind":
                npv_low  = shock_npv(ind=low_param)
                npv_high = shock_npv(ind=high_param)
            lows.append(npv_low); highs.append(npv_high); labels.append(lbl)

        # Plot tornado
        order = np.argsort(np.array(highs) - np.array(lows))
        labels_ord = [labels[i] for i in order]
        lows_ord   = [lows[i]   for i in order]
        highs_ord  = [highs[i]  for i in order]

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        y = np.arange(len(labels_ord))
        for i, (lo, hi) in enumerate(zip(lows_ord, highs_ord)):
            ax3.plot([lo, hi], [y[i], y[i]], lw=8)
        ax3.axvline(npv, color="black", linestyle="--", label="Base NPV")
        ax3.set_yticks(y); ax3.set_yticklabels(labels_ord)
        ax3.set_title("Tornado Chart — NPV sensitivity"); ax3.set_xlabel("NPV (€)"); ax3.legend()
        st.pyplot(fig3)

    # Contextual advisor on ROI tab
    with st.expander("Contextual Agent (offline)"):
        ag = contextual_agent("roi", {
            "net_benefit": annual_net_benefit,
            "payback_months": None if np.isinf(payback_months) else payback_months,
            "npv": npv,
            "sims": sims
        })
        st.markdown("- " + "\n- ".join(ag["bullets"]))
        st.caption("References: " + " | ".join(ag["references"]))
    st.caption("Prototype calculator; actual ROI depends on validated rates and loss models.")

# ==========================================================
# TAB 4 — OLS (original, intact)
# ==========================================================
with tabs[3]:
    st.subheader("OLS Econometrics (Prototype)")
    if not STATSMODELS_OK:
        st.warning("statsmodels is not available in this environment.")
    uploaded_csv = st.file_uploader("Upload dataset (CSV)", type=["csv"])
    if uploaded_csv is not None and STATSMODELS_OK:
        df = pd.read_csv(uploaded_csv)
        st.write("Preview:", df.head())
        cols = list(df.columns)
        y_col = st.selectbox("Dependent variable (Y)", cols)
        X_cols = st.multiselect("Independent variables (X)", [c for c in cols if c != y_col])
        use_const = st.checkbox("Add intercept", value=True)
        robust = st.selectbox("Robust SE", ["None", "HC3"], index=1)
        encode_cat = st.checkbox("One-hot encode categorical X", value=True)
        if st.button("Run OLS"):
            X = df[X_cols].copy()
            if encode_cat:
                X = pd.get_dummies(X, drop_first=True)
            data = pd.concat([df[y_col], X], axis=1).dropna()
            y_vec = data[y_col].astype(float)
            X_mat = data.drop(columns=[y_col]).astype(float)
            if use_const:
                X_mat = sm.add_constant(X_mat)
            model = sm.OLS(y_vec, X_mat)
            results = model.fit(cov_type="HC3") if (robust == "HC3") else model.fit()
            st.subheader("Summary"); st.text(results.summary())
            r2_val = results.rsquared

            st.subheader("VIF (multicollinearity)")
            try:
                vif_X = X_mat.drop(columns=["const"]) if "const" in X_mat.columns else X_mat.copy()
                vif_vals = [variance_inflation_factor(vif_X.values, i) for i in range(vif_X.shape[1])]
                vif_df = pd.DataFrame({"variable": vif_X.columns, "VIF": vif_vals})
                st.dataframe(vif_df)
            except Exception as e:
                st.warning(f"VIF calc issue: {e}")

            resid = results.resid; fitted = results.fittedvalues
            st.subheader("Diagnostics")
            fig1, ax1 = plt.subplots(); ax1.scatter(fitted, resid); ax1.axhline(0, linestyle="--")
            ax1.set_xlabel("Fitted"); ax1.set_ylabel("Residuals"); ax1.set_title("Residuals vs Fitted"); st.pyplot(fig1)
            fig2 = sm.qqplot(resid, line="45", fit=True); plt.title("QQ Plot of Residuals"); st.pyplot(fig2)
            fig3, ax3 = plt.subplots(); ax3.scatter(fitted, np.sqrt(np.abs(resid)))
            ax3.set_xlabel("Fitted"); ax3.set_ylabel("sqrt(|Residuals|)"); ax3.set_title("Scale-Location"); st.pyplot(fig3)
            infl = results.get_influence(); cooks_d = infl.cooks_distance[0]
            fig4, ax4 = plt.subplots(); ax4.stem(np.arange(len(cooks_d)), cooks_d, use_line_collection=True)
            ax4.set_xlabel("Observation"); ax4.set_ylabel("Cook's D"); ax4.set_title("Influential Observations"); st.pyplot(fig4)

            if DIAG_OK:
                try:
                    bp_test = het_breuschpagan(resid, results.model.exog); st.write("Breusch–Pagan:", bp_test)
                except Exception as e:
                    st.warning(f"BP test issue: {e}")
                try:
                    jb_stat, jb_p = normal_ad(resid); st.write(f"Normality (Anderson–Darling): stat={jb_stat:.3f}, p={jb_p:.3f}")
                except Exception as e:
                    st.warning(f"Normality test issue: {e}")

            with st.expander("Contextual Agent (offline)"):
                ag = contextual_agent("ols", {"r2": r2_val})
                st.markdown("- " + "\n- ".join(ag["bullets"]))
                st.caption("References: " + " | ".join(ag["references"]))
    elif uploaded_csv is None:
        st.info("Upload a CSV to begin.")

# ==========================================================
# TAB 5 — System Map (now interactive with PyVis; Graphviz fallback)
# ==========================================================
with tabs[4]:
    st.subheader("System-of-Systems Map (Ireland) — Interactive")
    if PYVIS_OK:
        # Build a small SoS consistent with your original boxes
        def pyvis_html():
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
            # Public spines
            net.add_node("AIM", label="DAFM AIM\n(Livestock traceability)", shape="box", color="#e3f2fd")
            net.add_node("PBI", label="Teagasc PastureBase", shape="box", color="#e3f2fd")
            net.add_node("OG",  label="Bord Bia Origin Green", shape="box", color="#e3f2fd")
            net.add_node("VM",  label="SFI VistaMilk", shape="box", color="#e3f2fd")
            # Private
            net.add_node("KEEL", label="Keelvar (Sourcing)", shape="box", color="#ede7f6")
            net.add_node("SCUR", label="Scurri (Delivery orchestration)", shape="box", color="#ede7f6")
            net.add_node("TAO",  label="Taoglas (IoT/Edge)", shape="box", color="#ede7f6")
            # AI modules
            net.add_node("CRACK", label="Offline Crack/Stress AI", shape="ellipse", color="#e8f5e9")
            net.add_node("OLS",   label="OLS Module (analytics)", shape="ellipse", color="#e8f5e9")
            net.add_node("BEHAV", label="Behavioural AI (future)", shape="ellipse", color="#fff3e0")
            # Standards
            net.add_node("GS1", label="GS1 / Standards", shape="diamond", color="#fffde7")

            pubs = ["AIM","PBI","OG","VM"]
            for p in pubs:
                net.add_edge(p, "CRACK"); net.add_edge(p, "OLS")
            for q in ["KEEL","SCUR","TAO"]:
                net.add_edge("CRACK", q); net.add_edge("OLS", q)
            for p in pubs:
                net.add_edge(p, "BEHAV", color="#ffb74d", title="future link")
            net.add_edge("BEHAV", "CRACK", color="#ffb74d", title="future fusion")
            net.add_edge("GS1", "AIM"); net.add_edge("GS1", "OG"); net.add_edge("GS1", "KEEL")

            # Save to temp + return HTML string
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            net.write_html(tmp.name, notebook=False)
            html = open(tmp.name, "r", encoding="utf-8").read()
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            return html

        html_map = pyvis_html()
        st.components.v1.html(html_map, height=650, scrolling=True)

    elif GRAPHVIZ_OK:
        dot = Digraph(comment="Ireland Agri System-of-Systems", format="svg")
        dot.attr(rankdir="LR", fontsize="10")
        dot.node("AIM", "DAFM AIM\n(Livestock traceability)", shape="box", style="rounded,filled", fillcolor="#e3f2fd")
        dot.node("PBI", "Teagasc PastureBase", shape="box", style="rounded,filled", fillcolor="#e3f2fd")
        dot.node("OG", "Bord Bia Origin Green", shape="box", style="rounded,filled", fillcolor="#e3f2fd")
        dot.node("VM", "SFI VistaMilk", shape="box", style="rounded,filled", fillcolor="#e3f2fd")
        dot.node("KEEL", "Keelvar\n(Sourcing)", shape="box", style="rounded,filled", fillcolor="#ede7f6")
        dot.node("SCUR", "Scurri\n(Delivery orchestration)", shape="box", style="rounded,filled", fillcolor="#ede7f6")
        dot.node("TAO", "Taoglas\n(IoT/Edge)", shape="box", style="rounded,filled", fillcolor="#ede7f6")
        dot.node("CRACK", "Offline Crack/Stress AI", shape="component", style="filled", fillcolor="#e8f5e9")
        dot.node("OLS", "OLS Module\n(analytics)", shape="component", style="filled", fillcolor="#e8f5e9")
        dot.node("BEHAV", "Behavioural AI (future)", shape="component", style="dashed")
        dot.node("GS1", "GS1/Standards", shape="ellipse", style="filled", fillcolor="#fffde7")
        for pub in ["AIM","PBI","OG","VM"]:
            dot.edge(pub, "CRACK"); dot.edge(pub, "OLS")
        for prv in ["KEEL","SCUR","TAO"]:
            dot.edge("CRACK", prv); dot.edge("OLS", prv)
        dot.edge("BEHAV", "CRACK", label="future fusion", style="dashed")
        for pub in ["AIM","PBI","OG","VM"]:
            dot.edge(pub, "BEHAV", style="dashed")
        dot.edge("GS1", "AIM"); dot.edge("GS1", "OG"); dot.edge("GS1", "KEEL")
        st.graphviz_chart(dot)
    else:
        st.warning("Neither PyVis nor Graphviz are available. Install with: pip install pyvis graphviz")

    with st.expander("Contextual Agent (offline)"):
        ag = contextual_agent("system", {})
        st.markdown("- " + "\n- ".join(ag["bullets"]))
        st.caption("References: " + " | ".join(ag["references"]))

# ==========================================================
# TAB 6 — Compliance (original)
# ==========================================================
with tabs[5]:
    st.subheader("Compliance & Governance Checklist")
    gdpr   = st.checkbox("No personal data leaves the device/server (offline processing).", value=True)
    storage= st.checkbox("No biometric storage; transient processing only.", value=True)
    transp = st.checkbox("Clear model purpose & limitations disclosed to users.", value=True)
    risk   = st.checkbox("Risk-based alerts (advisory), not automated enforcement.", value=True)
    audit  = st.checkbox("Logging for model versioning and audit (without PII).", value=True)
    optout = st.checkbox("Farmer opt-in/opt-out with data ownership retained.", value=True)
    st.write("---")
    if st.button("Generate Compliance Statement"):
        bullets = []
        if gdpr:  bullets.append("• Processing is offline-by-design; no personal data is transmitted externally.")
        if storage:bullets.append("• No biometric data is stored; image inputs are processed transiently for crack/stress estimation.")
        if transp: bullets.append("• Model purpose, scope, and limitations are disclosed to end users.")
        if risk:   bullets.append("• Outputs are advisory for maintenance; no automated punitive decisions.")
        if audit:  bullets.append("• Model versions and inference events are logged without PII for auditability.")
        if optout: bullets.append("• Farmers retain ownership and can opt in/out at any time.")
        from datetime import date
        st.markdown(f"""
**Compliance Statement — {date.today().isoformat()}**

This prototype Crack/Stress AI module is designed for offline deployment in Irish agricultural contexts.
It adheres to the following safeguards:

{chr(10).join(bullets)}

**Status:** Prototype — merits further validation on Teagasc research farms prior to operational adoption.
""")
    with st.expander("Contextual Agent (offline)"):
        ag = contextual_agent("compliance", {})
        st.markdown("- " + "\n- ".join(ag["bullets"]))
        st.caption("References: " + " | ".join(ag["references"]))
