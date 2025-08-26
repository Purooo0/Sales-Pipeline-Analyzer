"""
Streamlit UI for the Sales Pipeline Analyzer.

Flow:
1) Upload Excel -> 2) Show columns -> 3) Select columns -> 4) Map to canonical -> 5) Run analyses

Visual theme uses a single Telkomsel red accent for a professional, consistent look.
"""
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from processing import (
    CANONICAL_COLS,
    auto_map_columns,
    preprocess,
    run_bagian1,
    run_bagian2,
    idr_label,
    LOB_MAPPING,
)

# ---- Theme constants ----
TELKOMSEL_RED: str = "#e60000"
NEUTRAL_GRAY: str = "#d9d9d9"

# ---- Helpers for UI rendering (defined early to avoid NameError) ----
def _render_top_am_section(df_am: pd.DataFrame, metric: str, xlabel: str, key_suffix: str):
    """Render the Top 5 AM controls and per-LoB tabs for a given metric.

    Parameters
    ----------
    df_am : pd.DataFrame
        Aggregated AM performance dataframe produced by `run_bagian2()`.
    metric : str
        One of {"conversion_rate", "total_cv", "total"}.
    xlabel : str
        X-axis label for the bars.
    key_suffix : str
        Suffix for Streamlit widget keys to avoid collisions.
    """
    # Controls: pick LoB(s), years, Top N
    lob_options = sorted(df_am["LoB"].dropna().unique().tolist())
    c1, c2, c3, c4 = st.columns([2, 2, 1, 2])
    with c1:
        lob_single = st.selectbox("LoB", options=lob_options, key=f"lob_{key_suffix}")
    # union of years for all lobs
    years_all = sorted(df_am["Year"].dropna().unique().tolist())
    with c2:
        years_sel = st.multiselect("Years", options=years_all, default=years_all, key=f"years_{key_suffix}")
    with c3:
        topn = st.slider("Top N", min_value=3, max_value=10, value=5, step=1, key=f"topn_{key_suffix}")
    with c4:
        show_all = st.checkbox("Tampilkan semua LoB", value=False, key=f"all_{key_suffix}")

    if not years_sel:
        st.info("Pilih minimal 1 tahun.")
        return

    lobs_to_show = lob_options if show_all else [lob_single]
    for lob in lobs_to_show:
        st.markdown(f"#### {lob}")
        _render_top_am_for_lob(df_am, lob, years_sel, topn, metric, xlabel)

def _render_top_am_for_lob(df_am: pd.DataFrame, lob: str, years_sel: List[int], topn: int, metric: str, xlabel: str):
    """Render Top N AM bars for a single LoB across selected years."""
    tabs = st.tabs([str(y) for y in years_sel])
    for i, y in enumerate(years_sel):
        with tabs[i]:
            sub = df_am[(df_am["LoB"] == lob) & (df_am["Year"] == y)].copy()
            if sub.empty:
                st.warning("Tidak ada data untuk kombinasi ini.")
                continue
            sub = sub.sort_values(by=metric, ascending=False).head(topn)
            fig, ax = plt.subplots(figsize=(9, max(3.5, 0.5 * len(sub) + 2)))
            ax.barh(sub["am"], sub[metric], color=TELKOMSEL_RED)
            ax.invert_yaxis()
            # Labels on bars
            for j, v in enumerate(sub[metric].tolist()):
                label = _fmt_metric(v, metric)
                ax.text(v, j, f"  {label}", va="center", ha="left", fontsize=9, color="#333333")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Account Manager")
            ax.set_title(f"Top {len(sub)} AM - {y}")
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            st.pyplot(fig)

def _fmt_metric(v, metric: str) -> str:
    """Format metric values for in-bar labels."""
    if pd.isna(v):
        return ""
    if metric == "conversion_rate":
        return f"{v:.1f}%"
    if metric == "total_cv":
        return idr_label(v)
    if metric == "total":
        return f"{int(v)}"
    return str(v)

# Simple professional table styling with Telkomsel red accent (minimal colors)
def style_df(df: pd.DataFrame):
    """Return a styled dataframe with subtle Telkomsel-red accents (no gradients)."""
    if df is None or df.empty:
        return df
    styler = df.style.set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#ffe5e5"),
            ("color", "#7f0000"),
            ("font-weight", "600"),
            ("border-bottom", "1px solid #f0b3b3"),
        ]},
        {"selector": "td", "props": [
            ("border-bottom", "1px solid #f0f0f0"),
        ]},
    ]).set_properties(**{"text-align": "left"})
    return styler

st.set_page_config(page_title="Sales Pipeline Analyzer", layout="wide")

st.title("Sales Pipeline Analyzer")

st.sidebar.header("1) Upload Excel")
uploaded = st.sidebar.file_uploader("Upload .xlsx", type=["xlsx"])

@st.cache_data(show_spinner=False)
def load_excel(file) -> pd.DataFrame:
    """Load the first sheet of the uploaded Excel file."""
    return pd.read_excel(file, sheet_name=0, engine="openpyxl")

def dependency_sets() -> Dict[str, List[str]]:
    """Return minimal per-output dependencies to drive warnings/skips."""
    return {
        # Bagian 1
        "b1_summary": ["Industry Segment", "Schedule Amount", "Stage", "Opportunity Name", "Created Date", "Close Date"],
        "b1_pie": ["Pilar", "Schedule Amount", "Stage"],
        "b1_quarter_bars": ["Schedule Date", "Schedule Amount", "Stage"],
        "b1_pivot": ["Schedule Date", "Product Type", "Stage", "Opportunity Name"],
        "b1_top_bottom": ["Close Date (Year)", "Opportunity Name", "Account Name", "Close Date", "Schedule Amount", "Stage"],
        "b1_open_pipeline": ["Stage", "Close Date", "Schedule Amount", "Opportunity Name", "Account Name"],
        # Bagian 2
        "b2_core": ["Stage", "Created Date", "Close Date", "Industry Segment"],
        "b2_stage_close": ["Last Stage Change Date", "Close Date", "Pilar", "Industry Segment"],
        "b2_se": ["Last Stage Change Date", "Close Date", "Opportunity Owner"],
        "b2_top5_am": ["Close Date", "AM Name", "Opportunity Name", "Stage", "Schedule Amount"],
    }

def missing_for(deps: List[str], available_cols: List[str]) -> List[str]:
    """Return missing dependency columns from a list of available columns."""
    return [c for c in deps if c not in available_cols]

# Step 2: read + show columns
if uploaded:
    df_raw = load_excel(uploaded)
    st.subheader("Preview Data")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.subheader("Columns Detected")
    all_cols = list(df_raw.columns)
    st.write(", ".join(map(str, all_cols)))

    # Step 3: select columns to use
    st.sidebar.header("2) Select Columns to Use")
    selected_cols = st.sidebar.multiselect(
        "Pick columns to include for processing",
        options=all_cols,
        default=all_cols,
    )

    # Step 4: mapping UI
    st.sidebar.header("3) Column Mapping (only if needed)")
    auto_map = auto_map_columns(selected_cols)
    mapping: Dict[str, Optional[str]] = {}
    with st.sidebar.expander("Map to Canonical Names", expanded=False):
        for canon in CANONICAL_COLS:
            opts = [None] + selected_cols
            prefill = auto_map.get(canon)
            choice = st.selectbox(f"{canon}", options=opts, index=(opts.index(prefill) if prefill in opts else 0), key=f"map_{canon}")
            mapping[canon] = choice

    # Industry Segment filter (mapped to standardized 8 LoB)
    st.sidebar.header("4) Filters")
    seg_col = mapping.get("Industry Segment") or ("Industry Segment" if "Industry Segment" in df_raw.columns else None)
    selected_lobs: List[str] = []
    if seg_col:
        raw_seg = df_raw[seg_col].astype(str).str.strip().str.lower()
        mapped_lob = raw_seg.map(LOB_MAPPING)
        lob_options = sorted(pd.Series(mapped_lob).dropna().unique().tolist())
        selected_lobs = st.sidebar.multiselect("Industry Segment (mapped)", options=lob_options, default=lob_options)

    st.sidebar.header("5) Run (Per Fitur)")
    feature_options = [
        "Summary",
        "Pie Charts",
        "Quarterly Bars",
        "Opportunity Count Table",
        "Top/Bottom 5 Closed Won",
        "Open Pipeline by Close Year",
        "Avg Sales Cycle per LoB",
        "Avg Stage->Close per LoB & Product",
        "Avg Stage->Close per SE",
        "Win Rate per LoB",
        "Top 5 AM: Conversion Rate",
        "Top 5 AM: Total CV",
        "Top 5 AM: Total Deals",
    ]
    selected_features = st.sidebar.multiselect(
        "Pilih fitur yang ingin ditampilkan",
        options=feature_options,
        default=["Summary", "Pie Charts", "Quarterly Bars"],
    )
    btn_run = st.sidebar.button("Run")

    # Preprocess with mapping before running analyses
    # Keep only selected columns first
    df_sel = df_raw[selected_cols].copy()
    df = preprocess(df_sel, mapping)
    available = list(df.columns)

    # Dependency-driven notices
    deps = dependency_sets()

    # Run analyses and cache results when user clicks Run
    if btn_run:
        # Apply mapped LoB filter early so both bagian1 & bagian2 konsisten
        df_filtered = df.copy()
        if "Industry Segment" in df_filtered.columns and selected_lobs:
            seg_series = df_filtered["Industry Segment"].astype(str).str.strip().str.lower()
            lob_series = seg_series.map(LOB_MAPPING)
            df_filtered = df_filtered[lob_series.isin(selected_lobs)].copy()

        # bagian1 already supports industry_filter, but we filtered by mapped LoB above -> pass None to avoid double filtering
        res1 = run_bagian1(df_filtered, industry_filter=None)
        res2 = run_bagian2(df_filtered)
        st.session_state["results"] = {"res1": res1, "res2": res2}

    # Use cached results if available
    results = st.session_state.get("results")
    if not results:
        st.info("Klik Run untuk menjalankan analisis.")
        st.stop()

    res1 = results["res1"]
    res2 = results["res2"]

    # Show accumulated warnings up-front
    for w in res1.get("warnings", []) + res2.get("warnings", []):
        st.warning(w)

    # ----- Bagian 1 features -----
    if "Summary" in selected_features and "summary" in res1:
        st.markdown("### Summary")
        s = res1["summary"]
        st.write(f"Total Pipeline: {s['total_opps']} opps | CV IDR {s['total_cv_bn']} Bn")
        st.write(f"Closed Won: {s['won_opps']} opps | CV IDR {s['won_cv_bn']} Bn")
        st.write(f"Conversion Rate: {s['conversion_rate']}%")

    if "Pie Charts" in selected_features and "figures" in res1:
        if "pie_all" in res1["figures"]:
            st.markdown("### CV by Product Type (All Stage)")
            st.pyplot(res1["figures"]["pie_all"])
        if "pie_won" in res1["figures"]:
            st.markdown("### CV by Product Type (Won Only)")
            st.pyplot(res1["figures"]["pie_won"])

    if "Quarterly Bars" in selected_features and "figures" in res1 and "bars_quarterly" in res1["figures"]:
        st.markdown("### CV per Quarter - All Stage vs Won Only")
        st.pyplot(res1["figures"]["bars_quarterly"])

    if "Opportunity Count Table" in selected_features and "pivot_table" in res1:
        st.markdown("### Opportunity Count Table (Year-Quarter by Product Type & Stage)")
        st.dataframe(style_df(res1["pivot_table"]), use_container_width=True)

    if "Top/Bottom 5 Closed Won" in selected_features:
        if "top5" in res1 and res1["top5"]:
            st.markdown("### Top 5 Closed Won Opportunities per Year")
            for y, t in res1["top5"].items():
                st.markdown(f"**Year {y}")
                st.dataframe(style_df(t), use_container_width=True)
        if "bottom5" in res1 and res1["bottom5"]:
            st.markdown("### Bottom 5 Closed Won Opportunities per Year")
            for y, t in res1["bottom5"].items():
                st.markdown(f"**Year {y}")
                st.dataframe(style_df(t), use_container_width=True)

    if "Open Pipeline by Close Year" in selected_features and "open_pipeline" in res1 and res1["open_pipeline"]:
        st.markdown("### Open Opportunities per Close Year")
        for y, t in res1["open_pipeline"].items():
            st.markdown(f"**Close Year {y}")
            st.dataframe(style_df(t), use_container_width=True)

    # ----- Bagian 2 features -----
    if "Avg Sales Cycle per LoB" in selected_features:
        if "tables" in res2 and "cycle_per_lob" in res2["tables"]:
            st.markdown("### Average Sales Cycle Time per LoB (table)")
            st.dataframe(style_df(res2["tables"]["cycle_per_lob"]), use_container_width=True)
        if "figures" in res2 and "cycle_per_lob" in res2["figures"]:
            st.pyplot(res2["figures"]["cycle_per_lob"])

    if "Avg Stage->Close per LoB & Product" in selected_features and "figures" in res2 and "stage_to_close_lob_product" in res2["figures"]:
        st.markdown("### Avg Stage to Close Time per LoB and Product Type")
        st.pyplot(res2["figures"]["stage_to_close_lob_product"])

    if "Avg Stage->Close per SE" in selected_features and "figures" in res2 and "stage_to_close_per_se" in res2["figures"]:
        st.markdown("### Avg Stage to Close Time per Opportunity Owner (SE)")
        st.pyplot(res2["figures"]["stage_to_close_per_se"])

    if "Win Rate per LoB" in selected_features:
        if "tables" in res2 and "win_rate_per_lob" in res2["tables"]:
            st.markdown("### Win Rate per LoB (table)")
            st.dataframe(style_df(res2["tables"]["win_rate_per_lob"]), use_container_width=True)
        if "figures" in res2 and "win_rate_per_lob" in res2["figures"]:
            st.pyplot(res2["figures"]["win_rate_per_lob"])

    if "Top 5 AM: Conversion Rate" in selected_features:
        st.markdown("### Top 5 AM: Conversion Rate")
        df_am = res2.get("tables", {}).get("top5_am_df")
        if df_am is not None and not df_am.empty:
            _render_top_am_section(df_am, metric="conversion_rate", xlabel="Conversion Rate (%)", key_suffix="conv")
        elif "figures" in res2 and "top5_am_conversion" in res2["figures"]:
            st.pyplot(res2["figures"]["top5_am_conversion"])

    if "Top 5 AM: Total CV" in selected_features:
        st.markdown("### Top 5 AM: Total CV")
        df_am = res2.get("tables", {}).get("top5_am_df")
        if df_am is not None and not df_am.empty:
            _render_top_am_section(df_am, metric="total_cv", xlabel="Total Contract Value (Rupiah)", key_suffix="cv")
        elif "figures" in res2 and "top5_am_total_cv" in res2["figures"]:
            st.pyplot(res2["figures"]["top5_am_total_cv"])

    if "Top 5 AM: Total Deals" in selected_features:
        st.markdown("### Top 5 AM: Total Deals")
        df_am = res2.get("tables", {}).get("top5_am_df")
        if df_am is not None and not df_am.empty:
            _render_top_am_section(df_am, metric="total", xlabel="Total Deals (Opportunity Count)", key_suffix="deals")
        elif "figures" in res2 and "top5_am_total" in res2["figures"]:
            st.pyplot(res2["figures"]["top5_am_total"])

else:
    st.info("Upload file Excel (.xlsx) untuk memulai.")