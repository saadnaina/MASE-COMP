"""
Interactive Visual & Game-Style Rendering Engine for NeuroScan AI (GMU MASE 2026).
Generates procedural high-resolution medical illustrations, neural network decay matrices,
and colour-coded cerebral vascular territory maps (MCA, ACA, PCA, Vertebrobasilar) in real time.
"""
from typing import Tuple, Dict, Any
import cv2
import numpy as np


def render_time_is_brain_visual(delay_minutes: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Renders a dual-panel interactive visual simulation of ischemic core vs. penumbra growth
    and a live neural-synaptic survival matrix based on elapsed untreated minutes.
    """
    h, w = 440, 860
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # Deep sleek clinical background
    canvas[:] = (14, 21, 42)

    # Quantitative neuro-loss math (Saver, Stroke 2006: 1.9M neurons/min, 14B synapses/min)
    neurons_lost_m = round(delay_minutes * 1.9, 1)
    synapses_lost_b = round(delay_minutes * 14.0, 1)
    axons_lost_km = round(delay_minutes * 12.0, 1)
    brain_aging_yrs = round((delay_minutes / 60.0) * 3.6, 1)
    progression = min(1.0, delay_minutes / 270.0)  # 4.5 hour thrombolytic window benchmark
    salvageable_pct = max(5, int(round((1.0 - progression * 0.88) * 100)))

    # --- LEFT PANEL: AXIAL BRAIN ISCHEMIC CORE VS PENUMBRA SIMULATOR ---
    cx, cy = 215, 225
    cv2.putText(
        canvas,
        "1. LIVE ISCHEMIC CORE vs. PENUMBRA EXPANSION",
        (25, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (144, 224, 239),
        2,
        cv2.LINE_AA,
    )

    # Outer skull halo & cranial vault
    cv2.ellipse(canvas, (cx, cy), (150, 172), 0, 0, 360, (65, 85, 125), 6, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), (144, 166), 0, 0, 360, (215, 225, 240), 3, cv2.LINE_AA)
    # Healthy brain parenchyma
    cv2.ellipse(canvas, (cx, cy), (138, 160), 0, 0, 360, (48, 62, 88), -1, cv2.LINE_AA)

    # Sulcal & gyral anatomical lines
    rng = np.random.default_rng(42)
    for _ in range(28):
        angle = rng.uniform(0, 2 * np.pi)
        r_in = rng.uniform(95, 120)
        x1 = int(cx + r_in * np.cos(angle))
        y1 = int(cy + r_in * 1.12 * np.sin(angle))
        x2 = int(cx + (r_in + 18) * np.cos(angle + 0.12))
        y2 = int(cy + (r_in + 18) * 1.12 * np.sin(angle + 0.12))
        cv2.line(canvas, (x1, y1), (x2, y2), (34, 45, 66), 2, cv2.LINE_AA)

    # Midline falx cerebri
    cv2.line(canvas, (cx, cy - 158), (cx, cy + 158), (80, 105, 140), 2, cv2.LINE_AA)

    # Ventricles
    cv2.ellipse(canvas, (cx - 20, cy - 5), (12, 34), 8, 0, 360, (20, 30, 50), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx + 20, cy - 5), (12, 34), -8, 0, 360, (20, 30, 50), -1, cv2.LINE_AA)

    # Penumbra (At-risk salvageable tissue - Amber/Orange glow in Right MCA territory)
    penumbra_rx = int(58 + 20 * np.sin(progression * np.pi))
    penumbra_ry = int(82 + 22 * np.sin(progression * np.pi))
    overlay = canvas.copy()
    cv2.ellipse(
        overlay,
        (cx - 64, cy + 4),
        (penumbra_rx, penumbra_ry),
        -10,
        0,
        360,
        (245, 158, 11),
        -1,
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0, canvas)

    # Irreversible Infarct Core (Expanding crimson core as delay_minutes increases)
    core_rx = int(12 + 52 * progression)
    core_ry = int(16 + 74 * progression)
    core_overlay = canvas.copy()
    cv2.ellipse(
        core_overlay,
        (cx - 66, cy + 4),
        (core_rx, core_ry),
        -10,
        0,
        360,
        (239, 68, 68),
        -1,
        cv2.LINE_AA,
    )
    cv2.addWeighted(core_overlay, 0.78, canvas, 0.22, 0, canvas)
    cv2.ellipse(
        canvas,
        (cx - 66, cy + 4),
        (core_rx, core_ry),
        -10,
        0,
        360,
        (255, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # Legend below brain
    cv2.rectangle(canvas, (35, 405), (52, 422), (239, 68, 68), -1)
    cv2.putText(canvas, "Infarct Core (Dead)", (60, 418), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (248, 250, 252), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (220, 405), (237, 422), (245, 158, 11), -1)
    cv2.putText(canvas, "Penumbra (Salvageable)", (245, 418), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (248, 250, 252), 1, cv2.LINE_AA)

    # --- RIGHT PANEL: INTERACTIVE NEURAL SYNAPTIC MATRIX ---
    cv2.putText(
        canvas,
        "2. NEURAL NETWORK SURVIVAL MATRIX (100 NODES)",
        (445, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (144, 224, 239),
        2,
        cv2.LINE_AA,
    )

    grid_x0, grid_y0 = 465, 72
    spacing_x, spacing_y = 36, 29
    dead_nodes = int(round((100 - salvageable_pct)))
    at_risk_nodes = min(100 - dead_nodes, 22)

    node_coords = []
    for r in range(10):
        for c in range(10):
            idx = r * 10 + c
            nx = grid_x0 + c * spacing_x + (r % 2) * 6
            ny = grid_y0 + r * spacing_y
            node_coords.append((nx, ny, idx))

    # Draw synaptic axon connections between neighbouring nodes
    for nx, ny, idx in node_coords:
        for nx2, ny2, idx2 in node_coords:
            if idx2 > idx and ((nx - nx2) ** 2 + (ny - ny2) ** 2) <= 1950:
                if idx < dead_nodes or idx2 < dead_nodes:
                    line_col = (40, 25, 35)
                elif idx < dead_nodes + at_risk_nodes:
                    line_col = (120, 90, 20)
                else:
                    line_col = (0, 110, 145)
                cv2.line(canvas, (nx, ny), (nx2, ny2), line_col, 1, cv2.LINE_AA)

    # Draw neuron soma nodes
    for nx, ny, idx in node_coords:
        if idx < dead_nodes:
            col = (220, 38, 38)  # Destroyed neuron
            radius = 5
        elif idx < dead_nodes + at_risk_nodes:
            col = (245, 158, 11)  # Ischemic penumbra neuron
            radius = 6
        else:
            col = (0, 212, 255)  # Viable firing neuron
            radius = 6
        cv2.circle(canvas, (nx, ny), radius, col, -1, cv2.LINE_AA)
        if idx >= dead_nodes:
            cv2.circle(canvas, (nx, ny), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

    # Progress / Golden-Hour Status Bar
    bar_x1, bar_y1, bar_x2, bar_y2 = 460, 368, 825, 392
    cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 41, 59), -1)
    fill_w = int((bar_x2 - bar_x1) * (salvageable_pct / 100.0))
    bar_col = (16, 185, 129) if salvageable_pct > 65 else ((245, 158, 11) if salvageable_pct > 35 else (239, 68, 68))
    cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), bar_col, -1)
    cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (148, 163, 184), 1)
    cv2.putText(
        canvas,
        f"Estimated Salvageable Penumbra Reserve: {salvageable_pct}%",
        (460, 418),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (248, 250, 252),
        1,
        cv2.LINE_AA,
    )

    stats = {
        "delay_minutes": delay_minutes,
        "neurons_lost_millions": neurons_lost_m,
        "synapses_lost_billions": synapses_lost_b,
        "axons_lost_km": axons_lost_km,
        "brain_aging_years": brain_aging_yrs,
        "salvageable_pct": salvageable_pct,
        "golden_hour_status": (
            "WITHIN 60-MIN GOLDEN HOUR (Optimal Reperfusion Window)"
            if delay_minutes <= 60
            else (
                "WITHIN 4.5-HR IV THROMBOLYSIS WINDOW (Urgent Action Required)"
                if delay_minutes <= 270
                else "BEYOND 4.5-HR STANDARD IV tPA WINDOW (Evaluate for Mechanical Thrombectomy)"
            )
        ),
    }
    return canvas, stats


def render_vascular_territory_atlas(
    territory_key: str, occlusion_severity_pct: int = 80
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Renders a dual-view (Axial + Coronal/Homunculus) interactive cerebral arterial territory map
    and functional body deficit diagram for MCA, ACA, PCA, and Vertebrobasilar syndromes.
    """
    h, w = 440, 860
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (14, 21, 42)

    cv2.putText(
        canvas,
        "AXIAL VASCULAR TERRITORYPERFUSION MAP",
        (28, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (144, 224, 239),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "CIRCLE OF WILLIS & FUNCTIONAL DEFICIT ZONES",
        (450, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (144, 224, 239),
        2,
        cv2.LINE_AA,
    )

    # --- LEFT: AXIAL BRAIN SLICE WITH VASCULAR TERRITORY POLYGONS ---
    cx, cy = 215, 225
    cv2.ellipse(canvas, (cx, cy), (150, 172), 0, 0, 360, (65, 85, 125), 5, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), (140, 162), 0, 0, 360, (45, 58, 82), -1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - 160), (cx, cy + 160), (90, 110, 145), 2, cv2.LINE_AA)

    overlay = canvas.copy()
    alpha = 0.35 + 0.45 * (occlusion_severity_pct / 100.0)

    if "MCA" in territory_key:
        # Lateral Wedge (Middle Cerebral Artery)
        cv2.ellipse(
            overlay,
            (cx, cy),
            (138, 160),
            0,
            125,
            235,
            (239, 68, 68),
            -1,
            cv2.LINE_AA,
        )
        zone_title = "Middle Cerebral Artery (MCA) Territory"
        affected_lobes = "Lateral Frontal, Parietal & Superior Temporal Lobes + Basal Ganglia"
        motor_sensory = "Contralateral Face & Arm Weakness > Leg Weakness (Brachiofacial Paresis)"
        speech_vision = "Broca/Wernicke Aphasia (Left Hemisphere) or Hemispatial Neglect (Right)"
        vessel_focus = "M1 / M2 Horizontal Segment"
    elif "ACA" in territory_key:
        # Anterior Medial Strip (Anterior Cerebral Artery)
        pts = np.array(
            [
                [cx - 48, cy - 148],
                [cx, cy - 160],
                [cx, cy + 10],
                [cx - 38, cy + 10],
            ],
            np.int32,
        )
        cv2.fillPoly(overlay, [pts], (56, 189, 248))
        zone_title = "Anterior Cerebral Artery (ACA) Territory"
        affected_lobes = "Medial Frontal & Superior Parietal Lobes (Paracentral Lobule)"
        motor_sensory = "Contralateral Leg & Foot Weakness > Arm/Face Weakness"
        speech_vision = "Abulia (Apathy), Executive Dysfunction, Transcortical Motor Aphasia"
        vessel_focus = "A1 / A2 Pericallosal Segment"
    elif "PCA" in territory_key:
        # Posterior Occipital Wedge (Posterior Cerebral Artery)
        cv2.ellipse(
            overlay,
            (cx, cy),
            (138, 160),
            0,
            52,
            125,
            (168, 85, 247),
            -1,
            cv2.LINE_AA,
        )
        zone_title = "Posterior Cerebral Artery (PCA) Territory"
        affected_lobes = "Occipital Lobe (Primary Visual Cortex), Inferior Temporal Lobe & Thalamus"
        motor_sensory = "Thalamic Contralateral Hemisensory Loss (Dejerine-Roussy Syndrome)"
        speech_vision = "Contralateral Homonymous Hemianopia with Macular Sparing; Alexia without Agraphia"
        vessel_focus = "P1 / P2 Ambient Segment"
    else:
        # Central Brainstem / Cerebellar Posterior Fossa (Vertebrobasilar)
        cv2.circle(overlay, (cx, cy + 55), 54, (245, 158, 11), -1, cv2.LINE_AA)
        zone_title = "Vertebrobasilar / Brainstem & Cerebellar Territory"
        affected_lobes = "Pons, Medulla, Midbrain & Cerebellar Hemispheres"
        motor_sensory = "'Crossed' Cranial Nerve + Contralateral Motor Deficits; Truncal/Gait Ataxia"
        speech_vision = "The '5 Ds': Dizziness (Vertigo), Diplopia, Dysarthria, Dysphagia, Dystaxia"
        vessel_focus = "Basilar & Intracranial Vertebral Arteries"

    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
    # Re-draw ventricles for anatomical clarity
    cv2.ellipse(canvas, (cx - 18, cy - 6), (11, 32), 8, 0, 360, (18, 25, 42), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx + 18, cy - 6), (11, 32), -8, 0, 360, (18, 25, 42), -1, cv2.LINE_AA)

    # --- RIGHT: SCHEMATIC CIRCLE OF WILLIS & HOMUNCULUS DEFICIT INDICATOR ---
    w_cx, w_cy = 640, 210
    # Draw stylized Circle of Willis polygon
    basilar_top = (w_cx, w_cy + 40)
    basilar_bot = (w_cx, w_cy + 125)
    pca_l, pca_r = (w_cx - 75, w_cy + 35), (w_cx + 75, w_cy + 35)
    ica_l, ica_r = (w_cx - 48, w_cy - 15), (w_cx + 48, w_cy - 15)
    mca_l, mca_r = (w_cx - 135, w_cy - 25), (w_cx + 135, w_cy - 25)
    aca_l, aca_r = (w_cx - 22, w_cy - 110), (w_cx + 22, w_cy - 110)

    base_vessel_col = (70, 130, 180)
    # Draw baseline vessels
    vessels = [
        (basilar_bot, basilar_top),
        (basilar_top, pca_l),
        (basilar_top, pca_r),
        (basilar_top, ica_l),
        (basilar_top, ica_r),
        (ica_l, mca_l),
        (ica_r, mca_r),
        (ica_l, aca_l),
        (ica_r, aca_r),
        (aca_l, aca_r),
    ]
    for p1, p2 in vessels:
        cv2.line(canvas, p1, p2, base_vessel_col, 5, cv2.LINE_AA)

    # Highlight active occluded artery with pulsing marker
    if "MCA" in territory_key:
        cv2.line(canvas, ica_l, mca_l, (239, 68, 68), 9, cv2.LINE_AA)
        clot_pt = ((ica_l[0] + mca_l[0]) // 2, (ica_l[1] + mca_l[1]) // 2)
    elif "ACA" in territory_key:
        cv2.line(canvas, ica_l, aca_l, (56, 189, 248), 9, cv2.LINE_AA)
        clot_pt = ((ica_l[0] + aca_l[0]) // 2, (ica_l[1] + aca_l[1]) // 2)
    elif "PCA" in territory_key:
        cv2.line(canvas, basilar_top, pca_l, (168, 85, 247), 9, cv2.LINE_AA)
        clot_pt = ((basilar_top[0] + pca_l[0]) // 2, (basilar_top[1] + pca_l[1]) // 2)
    else:
        cv2.line(canvas, basilar_bot, basilar_top, (245, 158, 11), 10, cv2.LINE_AA)
        clot_pt = (w_cx, (basilar_bot[1] + basilar_top[1]) // 2)

    cv2.circle(canvas, clot_pt, 12, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, clot_pt, 8, (220, 38, 38), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"OCCLUSION SITE: {vessel_focus}",
        (465, 385),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (252, 165, 165),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Simulated Perfusion Deficit: {occlusion_severity_pct}%",
        (465, 412),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (144, 224, 239),
        1,
        cv2.LINE_AA,
    )

    info = {
        "zone_title": zone_title,
        "vessel_focus": vessel_focus,
        "affected_lobes": affected_lobes,
        "motor_sensory": motor_sensory,
        "speech_vision": speech_vision,
        "occlusion_severity_pct": occlusion_severity_pct,
    }
    return canvas, info
