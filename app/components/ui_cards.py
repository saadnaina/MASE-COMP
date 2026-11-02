"""
Reusable UI Components for NeuroScan AI clinical pages.
"""
import streamlit as st
from config import APP_NAME, APP_SUBTITLE, MEDICAL_DISCLAIMER_SHORT


def render_top_banner(page_title: str, page_subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="neuro-header">
            <h1>{APP_NAME} — {page_title}</h1>
            <p>{page_subtitle} | {APP_SUBTITLE}</p>
        </div>
        <div class="safety-banner">
            <strong>CLINICAL SAFETY DIRECTIVE:</strong> {MEDICAL_DISCLAIMER_SHORT}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    title: str, value: str, subtitle: str, is_urgent: bool = False
) -> None:
    card_class = "metric-card metric-card-urgent" if is_urgent else "metric-card"
    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
