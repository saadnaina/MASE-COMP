"""
High-Energy Neon Cyberpunk-Medical UI Theme for NeuroScan AI (GMU MASE 2026).
Exciting, vibrant glassmorphism with neon cyan/magenta pulses, high-contrast readability, and effortless usability.
"""
import streamlit as st


def apply_medical_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        /* Global App Styling */
        .stApp {
            background: radial-gradient(circle at 10% 10%, #0B132B 0%, #030712 100%);
            color: #F8FAFC;
            font-family: 'Outfit', 'Inter', sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F172A 0%, #070D1D 100%);
            border-right: 2px solid rgba(0, 180, 216, 0.35);
        }

        /* Exciting Glowing Header Banner */
        .neuro-header {
            background: linear-gradient(135deg, #1E1B4B 0%, #0F172A 50%, #042F2E 100%);
            border: 2px solid rgba(0, 180, 216, 0.6);
            border-left: 8px solid #00F2FE;
            border-radius: 16px;
            padding: 24px 30px;
            margin-bottom: 20px;
            box-shadow: 0 0 30px rgba(0, 242, 254, 0.2), inset 0 0 15px rgba(0, 242, 254, 0.1);
            animation: pulse-border 4s infinite alternate;
        }

        @keyframes pulse-border {
            0% { border-color: rgba(0, 180, 216, 0.4); box-shadow: 0 0 20px rgba(0, 242, 254, 0.15); }
            100% { border-color: rgba(0, 242, 254, 0.85); box-shadow: 0 0 35px rgba(0, 242, 254, 0.35); }
        }

        .neuro-header h1 {
            margin: 0;
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(90deg, #FFFFFF, #38BDF8, #00F2FE);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }
        .neuro-header p {
            margin: 6px 0 0 0;
            font-size: 1.05rem;
            color: #38BDF8;
            font-weight: 600;
        }

        /* Neon Safety Banner */
        .safety-banner {
            background: linear-gradient(90deg, rgba(220, 38, 38, 0.22) 0%, rgba(127, 29, 29, 0.28) 100%);
            border: 2px solid #EF4444;
            border-left: 8px solid #F87171;
            border-radius: 12px;
            padding: 14px 20px;
            margin-bottom: 22px;
            color: #FCA5A5;
            font-size: 0.92rem;
            line-height: 1.5;
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
        }
        .safety-banner strong {
            color: #FFFFFF;
            font-weight: 700;
        }

        /* Vibrant Metric & KPI Cards */
        .metric-card {
            background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-top: 4px solid #00F2FE;
            border-radius: 14px;
            padding: 20px 22px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            margin-bottom: 14px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .metric-card:hover {
            transform: translateY(-4px);
            border-color: #00F2FE;
            box-shadow: 0 12px 30px rgba(0, 242, 254, 0.25);
        }
        .metric-card-urgent {
            border-top: 4px solid #EF4444;
            border-color: rgba(239, 68, 68, 0.4);
        }
        .metric-card-urgent:hover {
            box-shadow: 0 12px 30px rgba(239, 68, 68, 0.35);
        }
        .metric-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94A3B8;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 2.1rem;
            font-weight: 800;
            color: #FFFFFF;
        }
        .metric-sub {
            font-size: 0.82rem;
            color: #38BDF8;
            font-weight: 600;
            margin-top: 6px;
        }

        /* Glassmorphism Info Panels */
        .info-panel {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 16px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }

        /* Glowing Action Buttons */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #00B4D8 0%, #0077B6 100%);
            border: 1px solid #90E0EF;
            color: white;
            font-weight: 700;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            box-shadow: 0 4px 20px rgba(0, 180, 216, 0.4);
            transition: all 0.3s ease;
        }
        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(0, 242, 254, 0.7);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
