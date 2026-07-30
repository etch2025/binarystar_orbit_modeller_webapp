"""
Streamlit front-end for the visual-binary orbit fitter.

Run locally with:   streamlit run app.py
"""
import io

import numpy as np
import pandas as pd
import streamlit as st

from orbitfit_core import run_fit

st.set_page_config(page_title="Visual Binary Orbit Fitter", layout="wide")
st.title("Visual Binary Orbit Fitter")
st.caption("Fit the seven Campbell orbital elements to relative astrometry "
           "via the Thiele-Innes method.")

# ----------------------------------------------------------------------
# Sidebar: all configuration
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Astrometry CSV", type="csv",
        help="Columns: THETA (deg), RHO (arcsec), YEAR, PARALLAX (mas, first row only)")
    target = st.text_input("Target name", "61 Cygni")

    st.header("Period search")
    P_lower = st.number_input("P lower (yr)", value=100.0, min_value=0.01)
    P_upper = st.number_input("P upper (yr)", value=2000.0, min_value=1.0)
    n_P_grid = st.slider("Grid points", 10, 500, 100,
                         help="Number of trial periods. More = finer map, slower.")
    n_restarts_per_P = st.slider("Restarts per period", 1, 30, 10,
                                 help="Random restarts of the 6 free elements at each P.")
    P_grid_log = st.checkbox("Log-spaced period grid", value=False)

    st.header("Acceptance")
    accept_factor = st.number_input("Accept factor (× best cost)", value=1.5,
                                    min_value=1.0, step=0.1)

    st.header("Mass constraint (optional)")
    use_mass = st.checkbox("Constrain by spectroscopic total mass", value=True)
    m1 = st.number_input("m1 guess (M☉)", value=0.70, min_value=0.0,
                         disabled=not use_mass)
    m2 = st.number_input("m2 guess (M☉)", value=0.64, min_value=0.0,
                         disabled=not use_mass)
    m_total_frac_accept = st.slider("Mass tolerance (fractional)", 0.01, 0.5, 0.10,
                                    disabled=not use_mass)

    st.header("Reproducibility")
    use_seed = st.checkbox("Fix random seed", value=True)
    seed = st.number_input("Seed", value=42, step=1, disabled=not use_seed)

    run = st.button("Fit orbit", type="primary", use_container_width=True)

# ----------------------------------------------------------------------
# Main panel
# ----------------------------------------------------------------------
if not uploaded:
    st.info("Upload an astrometry CSV to begin. Expected columns: "
            "`THETA, RHO, YEAR, PARALLAX` (parallax on the first data row only).")
    st.subheader("Example format")
    st.code("THETA,RHO,YEAR,PARALLAX\n"
            "149.8,30.629,2000,285.9949\n"
            "149.9,30.692,2001\n"
            "150.1,30.756,2002", language="text")
    st.stop()

# Preview the upload
raw = uploaded.getvalue()
try:
    st.subheader("Uploaded data")
    st.dataframe(pd.read_csv(io.BytesIO(raw)), height=200, use_container_width=True)
except Exception:
    st.warning("Could not preview the CSV, but the fitter will still try to read it.")

if run:
    bar = st.progress(0.0, text="Fitting orbits…")
    try:
        result = run_fit(
            io.BytesIO(raw),
            target=target,
            m1_guess=m1 if use_mass else None,
            m2_guess=m2 if use_mass else None,
            P_lower=P_lower, P_upper=P_upper,
            n_P_grid=int(n_P_grid), n_restarts_per_P=int(n_restarts_per_P),
            P_grid_log=P_grid_log, accept_factor=accept_factor,
            m_total_frac_accept=m_total_frac_accept,
            seed=int(seed) if use_seed else None,
            progress=lambda f: bar.progress(f, text=f"Fitting orbits… {f*100:.0f}%"),
        )
    except Exception as exc:
        bar.empty()
        st.error(f"Fit failed: {exc}")
        st.stop()
    bar.empty()

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accepted / total", f"{result.n_accept} / {result.n_total}")
    c2.metric("Distance (pc)", f"{result.d_pc:.2f}")
    c3.metric("Best P (yr)", f"{result.best_accept_fit[0]:.1f}")
    c4.metric("Best e", f"{result.best_accept_fit[2]:.3f}")

    if result.n_accept == 0:
        st.warning("No orbits passed the cost + mass constraints. "
                   "Showing the lowest-cost orbit instead — try widening the "
                   "accept factor or mass tolerance.")

    st.subheader("Best-fit orbit")
    st.pyplot(result.orbit_fig)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Cost vs period")
        st.pyplot(result.period_fig)
    with col_b:
        st.subheader("Cost vs eccentricity")
        st.pyplot(result.eccent_fig)

    # Accepted-family element ranges
    if result.ranges:
        st.subheader("Accepted-family element ranges")
        st.caption("Indicative spread across all accepted orbits, not a formal "
                   "error bar (residuals are unweighted).")
        st.dataframe(
            pd.DataFrame(
                [(k, v[0], v[2], v[1]) for k, v in result.ranges.items()],
                columns=["Element", "Min", "Median", "Max"],
            ).set_index("Element"),
            use_container_width=True,
        )

    # Downloads
    st.subheader("Downloads")
    d1, d2, d3 = st.columns(3)
    png = io.BytesIO()
    result.orbit_fig.savefig(png, format="png", dpi=200, bbox_inches="tight")
    d1.download_button("Best-fit orbit (PNG)", png.getvalue(),
                       file_name=f"orbit_fit_{target}.png", mime="image/png")

    all_csv = pd.DataFrame(
        result.fitted_values,
        columns=["P", "T", "e", "a_arcsec", "i_deg", "Omega_deg",
                 "omega_deg", "M_total", "cost", "R2"],
    ).to_csv(index=False)
    d2.download_button("All fits (CSV)", all_csv,
                       file_name=f"all_fits_{target}.csv", mime="text/csv")

    d3.download_button("Run log (TXT)", result.log_text,
                       file_name=f"logfile_{target}.txt", mime="text/plain")
