"""
Streamlit front-end for the visual-binary orbit fitter.

Run locally with:   streamlit run app.py
"""
import hashlib
import io
import zipfile
from urllib.parse import quote_plus
import os

import pandas as pd
import streamlit as st

from orbitfit_core import run_fit

EXAMPLE_DIR = "examples"

# ----------------------------------------------------------------------
# Example datasets and their recommended fit settings.
#
# Shared across all examples: 20 grid points, 5 restarts, accept factor 1.5,
# mass fractional acceptance 0.25, mass constraint on.
# Period search differs: 61 Cygni is a long-period system, the other three
# are short-period.
# Mass guesses come from examples/mass_guess.txt.
# ----------------------------------------------------------------------
COMMON_PRESET = {
    "n_P_grid": 20,
    "n_restarts_per_P": 5,
    "accept_factor": 1.5,
    "m_total_frac_accept": 0.25,
    "use_mass": True,
    "P_grid_log": False,
}

EXAMPLES = {
    "61 Cygni (HD 201091)": {
        "file": "61cyg.csv", "target": "61 Cygni (HD 201091)",
        "spectral": "K5V + K7V",
        "m1": 0.70, "m2": 0.64,
        "P_lower": 100.0, "P_upper": 2000.0,
    },
    "Kruger 60 (DO Cep)": {
        "file": "kr60.csv", "target": "Kruger 60 (DO Cep)",
        "spectral": "M3V + M4V",
        "m1": 0.30, "m2": 0.20,
        "P_lower": 0.1, "P_upper": 120.0,
    },
    "70 Ophiuchi": {
        "file": "70oph.csv", "target": "70 Ophiuchi",
        "spectral": "K0V + K4V",
        "m1": 0.86, "m2": 0.72,
        "P_lower": 0.1, "P_upper": 120.0,
    },
    "Sirius (Alp CMa)": {
        "file": "sirius.csv", "target": "Sirius (Alp CMa)",
        "spectral": "A1V + DA1",
        "m1": 2.17, "m2": 1.00,
        "P_lower": 0.1, "P_upper": 120.0,
    },
}

# Map bundled filename -> example key, so an *uploaded* copy of an example
# file also triggers its preset.
FILENAME_TO_EXAMPLE = {v["file"].lower(): k for k, v in EXAMPLES.items()}

UPLOAD_CHOICE = "Upload my own"

# ----------------------------------------------------------------------
# Session-state defaults. Every sidebar widget is bound by `key`, so presets
# work by writing into session_state; the user can still edit freely after.
# ----------------------------------------------------------------------
DEFAULTS = {
    "target": "61 Cygni (HD 201091)",
    "P_lower": 100.0,
    "P_upper": 2000.0,
    "n_P_grid": 20,
    "n_restarts_per_P": 5,
    "P_grid_log": False,
    "accept_factor": 1.5,
    "use_mass": True,
    "m1": 0.70,
    "m2": 0.64,
    "m_total_frac_accept": 0.25,
    "data_choice": "61 Cygni (HD 201091)",
}


# ----------------------------------------------------------------------
# Shareable URLs
#
# Every setting that defines a model is mirrored into the browser query
# string, so a URL fully reproduces an orbit model. Because the fit is
# deterministic for a fixed seed, opening a shared link and pressing
# "Fit orbit" reproduces the identical model.
#
# Caveat: an *uploaded* CSV cannot be embedded in a URL. Links are fully
# self-contained only for the bundled example datasets; for uploads the
# settings travel but the recipient must supply the same file.
# ----------------------------------------------------------------------
def _as_bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# param name -> (coercion function, short URL key)
PARAM_SPEC = {
    "data_choice":         (str, "d"),
    "target":              (str, "name"),
    "P_lower":             (float, "p0"),
    "P_upper":             (float, "p1"),
    "n_P_grid":            (int, "grid"),
    "n_restarts_per_P":    (int, "rs"),
    "P_grid_log":          (_as_bool, "plog"),
    "accept_factor":       (float, "af"),
    "use_mass":            (_as_bool, "mc"),
    "m1":                  (float, "m1"),
    "m2":                  (float, "m2"),
    "m_total_frac_accept": (float, "mtol"),
}

# Fixed internal seed, used for every fit so results stay reproducible
# without exposing a seed control in the UI.
FIT_SEED = 42


def _hydrate_from_url():
    """
    Merge query-string values over DEFAULTS, then seed session_state.

    Runs before any widget is instantiated. Uses setdefault, so on later
    reruns (when session_state already holds the user's edits) the URL is
    ignored and cannot clobber in-progress changes.
    """
    qp = st.query_params
    effective = dict(DEFAULTS)
    for name, (cast, key) in PARAM_SPEC.items():
        if key in qp:
            try:
                effective[name] = cast(qp[key])
            except (TypeError, ValueError):
                pass  # ignore malformed params rather than crashing
    # Guard against a stale/unknown dataset name in a shared link.
    if effective["data_choice"] not in EXAMPLES and \
            effective["data_choice"] != UPLOAD_CHOICE:
        effective["data_choice"] = UPLOAD_CHOICE
    for k, v in effective.items():
        st.session_state.setdefault(k, v)


def _write_url():
    """Mirror the current settings into the browser query string."""
    st.query_params.clear()
    for name, (_cast, key) in PARAM_SPEC.items():
        v = st.session_state.get(name, DEFAULTS.get(name))
        if isinstance(v, bool):
            v = int(v)
        st.query_params[key] = str(v)


def _share_url():
    """Best-effort absolute URL for the current model."""
    qs = "&".join(f"{k}={quote_plus(v)}" for k, v in st.query_params.to_dict().items())
    try:
        h = st.context.headers or {}
        host = h.get("Host") or h.get("host")
        proto = (h.get("X-Forwarded-Proto") or h.get("x-forwarded-proto")
                 or ("http" if host and host.split(":")[0]
                     in ("localhost", "127.0.0.1") else "https"))
        if host:
            return f"{proto}://{host}/?{qs}"
    except Exception:
        pass
    return f"?{qs}"


_hydrate_from_url()
st.session_state.setdefault("_preset_applied_for_upload", None)


def apply_preset(name):
    """Write an example's recommended settings into session_state."""
    p = EXAMPLES[name]
    st.session_state["target"] = p["target"]
    st.session_state["P_lower"] = float(p["P_lower"])
    st.session_state["P_upper"] = float(p["P_upper"])
    st.session_state["m1"] = float(p["m1"])
    st.session_state["m2"] = float(p["m2"])
    for k, v in COMMON_PRESET.items():
        st.session_state[k] = v


def _on_data_choice_change():
    """Callback: applying a preset here is safe (runs before the rerun)."""
    choice = st.session_state["data_choice"]
    if choice in EXAMPLES:
        apply_preset(choice)
        st.session_state["_preset_applied_for_upload"] = None


st.set_page_config(page_title="Binary Star Orbit Modeller", layout="wide")
st.title("Binary Star Orbit Modeller")
st.caption("Fit the seven Campbell orbital elements to relative astrometry "
           "via the Thiele-Innes method.")

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Data")
    st.selectbox(
        "Dataset",
        [UPLOAD_CHOICE] + list(EXAMPLES),
        key="data_choice",
        on_change=_on_data_choice_change,
        help="Picking an example loads its data and its recommended fit "
             "settings. You can still change any setting afterwards.",
    )

    uploaded = None
    if st.session_state["data_choice"] == UPLOAD_CHOICE:
        uploaded = st.file_uploader(
            "Astrometry CSV", type="csv",
            help="Columns: THETA (deg), RHO (arcsec), YEAR, "
                 "PARALLAX (mas, first row only)")
        # If the upload is a copy of a bundled example, apply its preset once.
        if uploaded is not None:
            key = os.path.basename(uploaded.name).lower()
            match = FILENAME_TO_EXAMPLE.get(key)
            if match and st.session_state["_preset_applied_for_upload"] != uploaded.name:
                apply_preset(match)
                st.session_state["_preset_applied_for_upload"] = uploaded.name
                st.rerun()

    st.text_input("Target name", key="target")

    st.header("Period search")
    st.number_input("P lower (yr)", key="P_lower",
                    min_value=0.001, step=0.1, format="%.3f")
    st.number_input("P upper (yr)", key="P_upper",
                    min_value=1.0, step=10.0, format="%.3f")
    st.slider("Grid points", 10, 500, key="n_P_grid",
              help="Number of trial periods. More = finer map, slower.")
    st.slider("Restarts per period", 1, 30, key="n_restarts_per_P",
              help="Random restarts of the 6 free elements at each P.")
    st.checkbox("Log-spaced period grid", key="P_grid_log")

    st.header("Acceptance")
    st.number_input("Accept factor (x best cost)", key="accept_factor",
                    min_value=1.0, step=0.1)

    st.header("Mass constraint (optional)")
    st.checkbox("Constrain by spectroscopic total mass", key="use_mass")
    _mass_off = not st.session_state["use_mass"]
    st.number_input("m1 guess (Msun)", key="m1", min_value=0.0, step=0.01,
                    disabled=_mass_off)
    st.number_input("m2 guess (Msun)", key="m2", min_value=0.0, step=0.01,
                    disabled=_mass_off)
    st.slider("Mass tolerance (fractional)", 0.01, 0.5,
              key="m_total_frac_accept", disabled=_mass_off)

    if st.session_state["data_choice"] in EXAMPLES:
        if st.button("Reset to recommended settings", use_container_width=True):
            apply_preset(st.session_state["data_choice"])
            st.rerun()

    run = st.button("Fit orbit", type="primary", use_container_width=True)

# ----------------------------------------------------------------------
# Resolve the data source
# ----------------------------------------------------------------------
choice = st.session_state["data_choice"]
raw = None
source_label = None

if choice in EXAMPLES:
    path = os.path.join(EXAMPLE_DIR, EXAMPLES[choice]["file"])
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        source_label = f"{choice}  ({EXAMPLES[choice]['spectral']})"
    except FileNotFoundError:
        st.error(f"Example file not found: {path}. Make sure the "
                 f"`{EXAMPLE_DIR}/` folder ships alongside app.py.")
        st.stop()
elif uploaded is not None:
    raw = uploaded.getvalue()
    source_label = uploaded.name

if raw is None:
    # No data selected right now. But if a previous fit is still in memory,
    # fall through and keep showing it instead of wiping the screen — only
    # show the welcome/help panel when there is genuinely nothing to display.
    if st.session_state.get("result") is None:
        st.info("Choose an example dataset in the sidebar, or upload your own "
                "astrometry CSV.")
        st.subheader("Example datasets included")
        st.dataframe(
            pd.DataFrame([
                {"System": k, "Components": v["spectral"], "File": v["file"],
                 "m1 (Msun)": v["m1"], "m2 (Msun)": v["m2"],
                 "Period search (yr)": f"{v['P_lower']:g} - {v['P_upper']:g}"}
                for k, v in EXAMPLES.items()
            ]).set_index("System"),
            use_container_width=True,
        )
        st.subheader("Expected CSV format")
        st.code("THETA,RHO,YEAR,PARALLAX\n"
                "149.8,30.629,2000,285.9949\n"
                "149.9,30.692,2001\n"
                "150.1,30.756,2002", language="text")
        st.stop()

# ----------------------------------------------------------------------
# Main panel
# ----------------------------------------------------------------------
if raw is not None:
    st.subheader(f"Data: {source_label}")
    if choice in EXAMPLES:
        st.caption("Recommended fit settings for this system have been "
                   "applied. Adjust anything in the sidebar and re-run.")
    try:
        st.dataframe(pd.read_csv(io.BytesIO(raw)), height=210,
                     use_container_width=True)
    except Exception:
        st.warning("Could not preview the CSV, but the fitter will still try "
                   "to read it.")
else:
    st.info("No dataset is currently selected — showing the most recent fit "
            "below. Pick a dataset or upload a CSV to run a new one.")

def _settings_signature():
    """Identity of the model the current sidebar settings would produce."""
    return (
        hashlib.md5(raw).hexdigest() if raw is not None else None,
        st.session_state["data_choice"], st.session_state["target"],
        st.session_state["P_lower"], st.session_state["P_upper"],
        int(st.session_state["n_P_grid"]),
        int(st.session_state["n_restarts_per_P"]),
        st.session_state["P_grid_log"], st.session_state["accept_factor"],
        st.session_state["use_mass"], st.session_state["m1"],
        st.session_state["m2"], st.session_state["m_total_frac_accept"],
    )


if run and raw is None:
    st.warning("Select a dataset or upload a CSV before fitting.")
    run = False

if run:
    # Clicking "Fit orbit" resets everything and recomputes from scratch.
    st.session_state["result"] = None
    st.session_state["fit_meta"] = None

    # Mirror the exact settings of this model into the URL before fitting,
    # so the address bar is a permalink to what is being computed.
    _write_url()
    bar = st.progress(0.0, text="Fitting orbits...")
    try:
        result = run_fit(
            io.BytesIO(raw),
            target=st.session_state["target"],
            m1_guess=st.session_state["m1"] if st.session_state["use_mass"] else None,
            m2_guess=st.session_state["m2"] if st.session_state["use_mass"] else None,
            P_lower=st.session_state["P_lower"],
            P_upper=st.session_state["P_upper"],
            n_P_grid=int(st.session_state["n_P_grid"]),
            n_restarts_per_P=int(st.session_state["n_restarts_per_P"]),
            P_grid_log=st.session_state["P_grid_log"],
            accept_factor=st.session_state["accept_factor"],
            m_total_frac_accept=st.session_state["m_total_frac_accept"],
            seed=FIT_SEED,
            progress=lambda f: bar.progress(
                f, text=f"Fitting orbits... {f * 100:.0f}%"),
        )
    except Exception as exc:
        bar.empty()
        st.error(f"Fit failed: {exc}")
        st.stop()
    bar.empty()

    # Persist the result together with a snapshot of the settings that
    # produced it, so later sidebar edits cannot mislabel these outputs.
    st.session_state["result"] = result
    st.session_state["fit_meta"] = {
        "target": st.session_state["target"],
        "data_choice": st.session_state["data_choice"],
        "source_label": source_label,
        "n_restarts_per_P": int(st.session_state["n_restarts_per_P"]),
        "signature": _settings_signature(),
        "share_url": _share_url(),
    }

# ----------------------------------------------------------------------
# Render from session_state, so changing a setting does NOT discard the
# fits, plots, or element ranges. Only the "Fit orbit" button recomputes.
# ----------------------------------------------------------------------
result = st.session_state.get("result")
meta = st.session_state.get("fit_meta") or {}

if result is not None:
    if meta.get("signature") and meta["signature"] != _settings_signature():
        st.info("Sidebar settings have changed since this fit. The results "
                "below are from the previous run - click **Fit orbit** to "
                "recompute with the new settings.")

    st.caption(f"Showing fit for **{meta.get('source_label', 'data')}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accepted / total", f"{result.n_accept} / {result.n_total}")
    c2.metric("Distance (pc)", f"{result.d_pc:.2f}")
    c3.metric("Best P (yr)", f"{result.best_accept_fit[0]:.1f}")
    c4.metric("Best e", f"{result.best_accept_fit[2]:.3f}")

    if result.n_accept == 0:
        st.warning("No orbits passed the cost + mass constraints. Showing the "
                   "lowest-cost orbit instead - try widening the accept factor "
                   "or the mass tolerance.")

    st.subheader("Best-fit orbit")
    st.pyplot(result.orbit_fig)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader("Cost vs period")
        st.pyplot(result.period_fig)
    with col_b:
        st.subheader("Cost vs semi-major axis")
        st.pyplot(result.sma_fig)
    with col_c:
        st.subheader("Cost vs eccentricity")
        st.pyplot(result.eccent_fig)

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

    st.subheader("Share this model")
    if meta.get("data_choice") == UPLOAD_CHOICE:
        st.info("This link carries the fit settings, but an uploaded CSV "
                "cannot be embedded in a URL — the recipient will need to "
                "upload the same file. Links to the bundled example datasets "
                "are fully self-contained.")
    st.caption("This link points to the fit shown above.")
    st.code(meta.get("share_url") or _share_url(), language=None)

    st.subheader("Downloads")
    tag = meta.get("target", "orbit").replace(" ", "_")
    d1, d2, d3, d4 = st.columns(4)
    png = io.BytesIO()
    result.orbit_fig.savefig(png, format="png", dpi=200, bbox_inches="tight")
    d1.download_button("Best-fit orbit (PNG)", png.getvalue(),
                       file_name=result.orbit_filename, mime="image/png")

    # Bundle the three cost-diagnostic plots (period, semi-major axis,
    # eccentricity) into a single zip for one-click download. Filenames
    # inside the zip match orbit3.py's naming scheme exactly.
    cost_plots = {
        result.period_filename: result.period_fig,
        result.sma_filename: result.sma_fig,
        result.eccent_filename: result.eccent_fig,
    }
    cost_zip = io.BytesIO()
    with zipfile.ZipFile(cost_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, fig in cost_plots.items():
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            zf.writestr(fname, buf.getvalue())
    d2.download_button("All cost plots (ZIP)", cost_zip.getvalue(),
                       file_name=f"cost_plots_{tag}.zip", mime="application/zip")

    all_csv = pd.DataFrame(
        result.fitted_values,
        columns=["P", "T", "e", "a_arcsec", "i_deg", "Omega_deg",
                 "omega_deg", "M_total", "cost", "R2"],
    ).to_csv(index=False)
    d3.download_button("All fits (CSV)", all_csv,
                       file_name=f"all_fits_{tag}.csv", mime="text/csv")

    d4.download_button("Run log (TXT)", result.log_text,
                       file_name=result.log_filename, mime="text/plain")