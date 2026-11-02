"""
PAGE 6: Interactive Stroke Simulators, Vascular Territory Game Viewer,
Comprehensive Stroke & Symptom Encyclopedia, 10-Year Prevention Calculator,
and UAE / Thumbay University Hospital (Ajman) EMS Telemetry Bridge.
"""
import math
import streamlit as st

from ai_engine.neuro_visuals import (
    render_time_is_brain_visual,
    render_vascular_territory_atlas,
)
from app.components.ui_cards import render_metric_card, render_top_banner
from database.db_manager import DatabaseManager


def render_neuro_simulators_page(db: DatabaseManager) -> None:
    render_top_banner(
        page_title="Interactive Neuro-Simulators, Stroke Encyclopedia & UAE EMS Bridge",
        page_subtitle="Game-Style 'Time Is Brain' & Vascular Territory Viewers + Text Mode + Thumbay Hospital EMS",
    )

    # Global View Mode Toggle: Interactive Game-Style Visuals vs. Clean Text-Only Reference
    mode_col1, mode_col2 = st.columns([2, 1])
    with mode_col1:
        display_mode = st.radio(
            "Select Viewing Mode",
            options=[
                "🎮 Interactive Game-Style Visual + Clinical Text Mode",
                "📄 Clean Text-Only Clinical Reference Mode",
            ],
            horizontal=True,
        )
    with mode_col2:
        st.caption(
            "Switch seamlessly between live interactive visual simulations and high-contrast text-only study notes."
        )

    show_visuals = "Interactive" in display_mode

    tab_tib, tab_atlas, tab_info, tab_prev, tab_ems = st.tabs(
        [
            "⏳ 1. 'Time Is Brain' Interactive Simulator",
            "🧠 2. Vascular Territory Game Explorer",
            "📖 3. Stroke & Symptoms Encyclopedia (BE-FAST)",
            "🛡️ 4. 10-Year Stroke Risk Forecaster",
            "🚑 5. UAE 998 & Thumbay Hospital EMS Bridge",
        ]
    )

    # =========================================================================
    # TAB 1: TIME IS BRAIN INTERACTIVE SIMULATOR
    # =========================================================================
    with tab_tib:
        st.markdown("### ⏳ Interactive 'Time Is Brain' Ischemic Core vs. Penumbra Simulator")
        st.caption(
            "Drag the slider or select a time preset to observe how untreated large-vessel ischemic stroke "
            "expands the irreversible infarct core into the salvageable penumbra (Saver, Stroke 2006)."
        )

        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        if p_col1.button("⚡ 15 Mins (Hyperacute)", use_container_width=True):
            st.session_state["tib_minutes"] = 15
        if p_col2.button("🏅 60 Mins (Golden Hour)", use_container_width=True):
            st.session_state["tib_minutes"] = 60
        if p_col3.button("⚠️ 180 Mins (3 Hours)", use_container_width=True):
            st.session_state["tib_minutes"] = 180
        if p_col4.button("🚨 270 Mins (4.5 Hr Limit)", use_container_width=True):
            st.session_state["tib_minutes"] = 270

        delay_mins = st.slider(
            "Untreated Ischemic Delay Since Symptom Onset (Minutes)",
            min_value=5,
            max_value=360,
            step=5,
            value=st.session_state.get("tib_minutes", 45),
        )
        st.session_state["tib_minutes"] = delay_mins

        tib_img, stats = render_time_is_brain_visual(delay_mins)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_metric_card(
                "Neurons Destroyed",
                f"{stats['neurons_lost_millions']:,.1f} M",
                "1.9 Million Neurons / Minute",
                is_urgent=delay_mins >= 60,
            )
        with k2:
            render_metric_card(
                "Synapses Lost",
                f"{stats['synapses_lost_billions']:,.0f} B",
                "14 Billion Synapses / Minute",
                is_urgent=delay_mins >= 60,
            )
        with k3:
            render_metric_card(
                "Myelinated Axons Lost",
                f"{stats['axons_lost_km']:,.0f} km",
                "12 km Axonal Fibers / Minute",
            )
        with k4:
            render_metric_card(
                "Accelerated Brain Aging",
                f"+{stats['brain_aging_years']:.1f} Yrs",
                f"Penumbra Reserve: {stats['salvageable_pct']}%",
                is_urgent=stats["salvageable_pct"] < 45,
            )

        if show_visuals:
            st.image(
                tib_img,
                caption=(
                    f"Interactive Simulation at T+{delay_mins} Minutes | Left: Expanding Infarct Core (Red) vs. "
                    f"Salvageable Penumbra (Amber) | Right: 100-Node Neural Network Viability Matrix "
                    f"({stats['salvageable_pct']}% Viable Reserve)"
                ),
                use_container_width=True,
            )

        st.markdown(
            f"""
            <div class="info-panel">
                <h4>Clinical Time-Window Status: <code>{stats['golden_hour_status']}</code></h4>
                <ul>
                    <li><strong>Core vs. Penumbra Physiology:</strong> During an acute ischemic stroke, the central <strong>infarct core</strong> suffers severe blood flow reduction (&lt;10–12 mL/100g/min) and dies rapidly. Surrounding it is the <strong>ischemic penumbra</strong>—stunned, electrically silent brain tissue kept alive temporarily by collateral blood vessels.</li>
                    <li><strong>The Golden Hour (0–60 Minutes):</strong> Patients evaluated and treated within the first 60 minutes have the highest likelihood of complete neurological recovery.</li>
                    <li><strong>IV Thrombolysis Window (Up to 4.5 Hours / 270 Mins):</strong> Intravenous tissue plasminogen activator (Alteplase / Tenecteplase) can dissolve clots if intracranial haemorrhage is first excluded on Non-Contrast Brain CT.</li>
                    <li><strong>Mechanical Thrombectomy Window (Up to 6–24 Hours):</strong> In selected patients with Large Vessel Occlusion (LVO) and salvageable penumbra on CT Perfusion / MRI, endovascular clot retrieval remains effective up to 24 hours.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 2: VASCULAR TERRITORY GAME EXPLORER
    # =========================================================================
    with tab_atlas:
        st.markdown("### 🧠 Interactive Cerebral Vascular Territory & Circle of Willis Explorer")
        ctl_col1, ctl_col2 = st.columns([2, 1])
        with ctl_col1:
            territory_choice = st.selectbox(
                "Select Cerebral Artery Territory to Simulate Occlusion",
                options=[
                    "Middle Cerebral Artery (MCA) — Lateral Hemisphere (Most Common)",
                    "Anterior Cerebral Artery (ACA) — Medial Frontal & Leg Motor Cortex",
                    "Posterior Cerebral Artery (PCA) — Occipital Visual Cortex & Thalamus",
                    "Vertebrobasilar / Brainstem — Posterior Fossa & Cranial Nerves",
                ],
            )
        with ctl_col2:
            occ_severity = st.slider(
                "Simulated Perfusion Deficit Severity (%)",
                min_value=20,
                max_value=100,
                step=5,
                value=85,
            )

        atlas_img, v_info = render_vascular_territory_atlas(territory_choice, occ_severity)

        if show_visuals:
            st.image(
                atlas_img,
                caption=(
                    f"Interactive Vascular Perfusion Map: {v_info['zone_title']} | "
                    f"Target Vessel: {v_info['vessel_focus']} ({occ_severity}% Perfusion Deficit)"
                ),
                use_container_width=True,
            )

        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.markdown(
                f"""
                <div class="info-panel">
                    <h4>📍 Anatomical & Vascular Localization</h4>
                    <p><strong>Arterial Territory:</strong> {v_info['zone_title']}</p>
                    <p><strong>Primary Occlusion Site:</strong> <code>{v_info['vessel_focus']}</code></p>
                    <p><strong>Brain Structures Supplied:</strong> {v_info['affected_lobes']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with vcol2:
            st.markdown(
                f"""
                <div class="info-panel">
                    <h4>🩺 Expected Clinical Presentation & Deficits</h4>
                    <p><strong>Motor & Sensory Pattern:</strong> {v_info['motor_sensory']}</p>
                    <p><strong>Speech, Vision & Higher Cortical Signs:</strong> {v_info['speech_vision']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 3: STROKE & SYMPTOMS ENCYCLOPEDIA (NORMAL CLINICAL INFORMATION)
    # =========================================================================
    with tab_info:
        st.markdown("### 📖 Comprehensive Guide to Stroke Types, Symptoms (BE-FAST) & First Aid")
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            st.markdown(
                """
                <div class="info-panel">
                    <h4>1. What Is a Stroke? (The 3 Major Types)</h4>
                    <ul>
                        <li><strong>Ischemic Stroke (~85–87% of cases):</strong> Caused by a blood clot (thrombus or embolus) blocking an artery supplying the brain. Common sources include carotid atherosclerosis, atrial fibrillation (cardioembolic), or small-vessel lacunar disease.</li>
                        <li><strong>Haemorrhagic Stroke (~13–15% of cases):</strong> Caused by rupture of a cerebral blood vessel leading to bleeding into the brain parenchyma (<strong>Intraparenchymal Haemorrhage</strong>) or subarachnoid space (<strong>Subarachnoid Haemorrhage</strong>). Often linked to uncontrolled hypertension, aneurysms, or arteriovenous malformations (AVMs).</li>
                        <li><strong>Transient Ischemic Attack (TIA — "Mini-Stroke"):</strong> A temporary blockage causing acute stroke symptoms that resolve completely without permanent infarction. <strong>Medical Emergency:</strong> Carries a high short-term risk of a full stroke within 48 hours.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="info-panel">
                    <h4>2. Why Brain CT Cannot Rule Out Early Ischemic Stroke</h4>
                    <p>Non-Contrast Brain CT (NCCT) is performed immediately to detect <strong>acute haemorrhage</strong> (which appears bright/hyperdense around +60 to +80 HU). However, in the first 3–6 hours of an <strong>ischemic stroke</strong>, the brain CT often looks completely normal. Therefore, <strong>a negative haemorrhage CT with positive FAST symptoms still requires immediate emergency stroke escalation</strong>.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with e_col2:
            st.markdown(
                """
                <div class="info-panel">
                    <h4>3. Recognizing Stroke Symptoms: The BE-FAST Protocol</h4>
                    <ul>
                        <li><strong>B — Balance:</strong> Sudden loss of balance, unsteadiness, dizziness, or coordination difficulty (ataxia).</li>
                        <li><strong>E — Eyes:</strong> Sudden blurred vision, double vision (diplopia), or loss of vision in one or both eyes.</li>
                        <li><strong>F — Face Drooping:</strong> One side of the face droops or feels numb. Ask the person to smile—is the smile uneven?</li>
                        <li><strong>A — Arm Weakness:</strong> One arm is weak or numb. Ask the person to raise both arms for 10 seconds—does one arm drift downward?</li>
                        <li><strong>S — Speech Difficulty:</strong> Slurred speech (dysarthria), difficulty finding words, or inability to repeat a simple sentence (aphasia).</li>
                        <li><strong>T — Time to Call Emergency Services:</strong> Call <strong>998 (UAE)</strong> / <strong>911</strong> immediately and note the exact <strong>Last Known Well (LKW)</strong> time.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="info-panel">
                    <h4>4. Critical Bystander First-Aid Rules (Do's & Don'ts)</h4>
                    <ul>
                        <li>✅ <strong>DO Lock the Exact Symptom Onset / LKW Time:</strong> Doctors need this timestamp to determine eligibility for clot-busting therapy.</li>
                        <li>✅ <strong>DO Position Safely:</strong> If unconscious or vomiting, place the patient in the lateral <strong>Recovery Position</strong> to protect the airway.</li>
                        <li>⛔ <strong>NEVER Give Aspirin Before a Brain CT:</strong> If the stroke is haemorrhagic, aspirin worsens intracranial bleeding.</li>
                        <li>⛔ <strong>NEVER Give Food, Water, or Oral Pills:</strong> Stroke frequently impairs swallowing (dysphagia), causing life-threatening aspiration pneumonia or choking.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 4: 10-YEAR STROKE RISK FORECASTER & BIOMARKER SIMULATOR
    # =========================================================================
    with tab_prev:
        st.markdown("### 🛡️ Interactive 10-Year Stroke Risk Forecaster & Modifiable Biomarker Simulator")
        p_left, p_right = st.columns([1, 1])
        with p_left:
            age_in = st.slider("Age (Years)", 20, 90, 58, 1)
            sbp_in = st.slider("Systolic Blood Pressure (mmHg)", 95, 210, 152, 1)
            bp_med_in = st.checkbox("Currently Taking Antihypertensive Medication", value=True)
            afib_in = st.checkbox("History of Atrial Fibrillation (AFib)", value=False)
            diab_in = st.checkbox("Type 2 Diabetes Mellitus", value=True)
            smoke_in = st.checkbox("Active Tobacco Smoker", value=True)
            bmi_in = st.slider("Body Mass Index (BMI kg/m²)", 16.0, 45.0, 29.0, 0.5)

        with p_right:
            base_points = max(0.0, (age_in - 40) * 0.35)
            if sbp_in >= 160:
                base_points += 6.5 if bp_med_in else 5.5
            elif sbp_in >= 140:
                base_points += 4.0 if bp_med_in else 3.2
            if afib_in:
                base_points += 6.0
            if diab_in:
                base_points += 3.2
            if smoke_in:
                base_points += 3.0
            if bmi_in >= 30.0:
                base_points += 1.5

            risk_10yr = min(68.0, round(1.2 * math.exp(0.11 * base_points), 1))
            opt_risk = min(68.0, round(1.2 * math.exp(0.11 * max(0.0, (age_in - 40) * 0.35)), 1))

            r_col1, r_col2 = st.columns(2)
            with r_col1:
                render_metric_card(
                    "Estimated 10-Yr Risk",
                    f"{risk_10yr}%",
                    "Current Risk Factor Profile",
                    is_urgent=risk_10yr >= 15.0,
                )
            with r_col2:
                render_metric_card(
                    "Optimized Target Risk",
                    f"{opt_risk}%",
                    "After Controlling Modifiable Factors",
                )

            actions = []
            if sbp_in >= 130:
                actions.append(
                    "<li><strong>Blood Pressure Optimization:</strong> Every 10 mmHg reduction in systolic BP lowers stroke risk by ~25–30%. Target &lt; 120/80 mmHg under physician guidance.</li>"
                )
            if afib_in:
                actions.append(
                    "<li><strong>Atrial Fibrillation Anticoagulation:</strong> AFib increases cardioembolic stroke risk 5-fold; discuss CHA₂DS₂-VASc scoring and oral anticoagulation with a cardiologist.</li>"
                )
            if smoke_in:
                actions.append(
                    "<li><strong>Smoking Cessation:</strong> Stopping tobacco use rapidly reduces endothelial inflammation and halves excess stroke risk within 2–4 years.</li>"
                )
            if diab_in:
                actions.append(
                    "<li><strong>Glycaemic & Lipid Control:</strong> Target HbA1c &lt; 7.0% and high-intensity statin therapy where clinically indicated.</li>"
                )
            if bmi_in >= 25.0:
                actions.append(
                    "<li><strong>Mediterranean Diet & Aerobic Activity:</strong> 150 minutes/week of moderate aerobic exercise improves vascular compliance.</li>"
                )
            if not actions:
                actions.append(
                    "<li><strong>Optimal Baseline:</strong> Continue routine annual blood pressure, lipid, and glucose screening.</li>"
                )

            st.markdown(
                f"""
                <div class="info-panel">
                    <h4>Personalized Primary Prevention Action Plan</h4>
                    <ul>{''.join(actions)}</ul>
                    <p style="font-size:0.8rem;color:#94A3B8;margin-top:8px;">Note: Educational Framingham-style risk illustration — not a substitute for formal clinical risk scoring.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 5: UAE 998 & THUMBAY UNIVERSITY HOSPITAL (AJMAN) EMS TELEMETRY BRIDGE
    # =========================================================================
    with tab_ems:
        st.markdown("### 🚑 Live UAE National Ambulance (998) & Thumbay University Hospital EMS Telemetry")
        st.caption(
            "Simulates pre-hospital Golden-Hour ambulance routing from Dubai to Thumbay University Hospital (Ajman) "
            "or Rashid Hospital Stroke Center, pre-notifying the CT suite before arrival."
        )
        dest_hospital = st.selectbox(
            "Select Receiving Stroke-Ready Center (UAE Network)",
            options=[
                "Thumbay University Hospital — GMU Academic Health System (Ajman)",
                "Rashid Hospital Comprehensive Stroke Center (Dubai Health)",
                "Cleveland Clinic Abu Dhabi Neurovascular Center",
            ],
        )
        transit_step = st.slider("Simulate Ambulance Route Progress (0 = Dispatch, 10 = Hospital Arrival)", 0, 10, 7)

        progress = min(1.0, transit_step / 10.0)
        # Coordinates interpolating between Al Qusais (Dubai) and Thumbay University Hospital (Al Jurf, Ajman)
        curr_lat = 25.2850 + (25.4111 - 25.2850) * progress
        curr_lon = 55.3950 + (55.5052 - 55.3950) * progress
        eta_mins = max(1, int(round(14 * (1.0 - progress))))

        geofence_status = (
            "🔔 GEOFENCE BREACH (<5 MINS OUT): CT SCANNER PRE-CLEARED & STROKE TEAM AT BAY"
            if eta_mins <= 5
            else f"🚑 AMBULANCE EN ROUTE — ESTIMATED ARRIVAL IN {eta_mins} MINUTES"
        )

        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:1.25fr 1fr;gap:18px;background:rgba(16,25,53,0.88);border:1px solid rgba(56,189,248,0.35);padding:20px;border-radius:14px;">
                <div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                        <strong style="color:#38BDF8;">📡 Live Pre-Hospital GPS Telemetry ({dest_hospital.split('—')[0].strip()})</strong>
                        <span style="background:rgba(239,68,68,0.22);border:1px solid #EF4444;color:#FCA5A5;padding:3px 10px;border-radius:99px;font-size:0.75rem;font-weight:700;">{geofence_status}</span>
                    </div>
                    <iframe width="100%" height="270" frameborder="0" scrolling="no" style="border-radius:10px;border:1px solid #334155;" src="https://www.openstreetmap.org/export/embed.html?bbox={curr_lon-0.03}%2C{curr_lat-0.02}%2C{curr_lon+0.03}%2C{curr_lat+0.02}&amp;layer=mapnik&amp;marker={curr_lat}%2C{curr_lon}"></iframe>
                </div>
                <div style="background:rgba(15,23,42,0.85);padding:16px;border-radius:12px;border:1px solid rgba(71,85,105,0.6);">
                    <h4 style="margin:0 0 8px 0;color:#F8FAFC;">Pre-Hospital Stroke Alert Packet</h4>
                    <p style="font-size:0.86rem;color:#CBD5E1;margin:4px 0;"><strong>Active Patient Ref:</strong> <code>{st.session_state.get('active_patient_ref_id', 'DEMO-SCENARIO-A')}</code></p>
                    <p style="font-size:0.86rem;color:#CBD5E1;margin:4px 0;"><strong>Active Assessment ID:</strong> <code>{st.session_state.get('active_assessment_id', 'ASM-DEMOA001')}</code></p>
                    <p style="font-size:0.86rem;color:#CBD5E1;margin:4px 0;"><strong>Destination:</strong> {dest_hospital}</p>
                    <p style="font-size:0.86rem;color:#CBD5E1;margin:4px 0;"><strong>UAE Emergency Dispatch:</strong> Dial <strong>998</strong> (National Ambulance)</p>
                    <hr style="border-color:#1E293B;margin:10px 0;">
                    <p style="font-size:0.82rem;color:#90E0EF;margin:0;"><strong>Hospital Readiness Checklist:</strong><br>
                    ✓ Non-Contrast Brain CT Table Reserved<br>
                    ✓ On-Call Neurologist & Radiologist Paged<br>
                    ✓ Point-of-Care Coagulation & Glucose Ready</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from app.styles import apply_medical_theme
    st.set_page_config(page_title="NeuroScan AI | Simulators & Atlas", page_icon="🧠", layout="wide")
    apply_medical_theme()
    for k, v in {"active_assessment_id": "ASM-DEMOA001", "active_patient_ref_id": "DEMO-SCENARIO-A", "current_page": "6. Interactive Stroke Simulators & UAE EMS"}.items():
        if k not in st.session_state:
            st.session_state[k] = v
    _db = DatabaseManager()
    render_neuro_simulators_page(_db)
