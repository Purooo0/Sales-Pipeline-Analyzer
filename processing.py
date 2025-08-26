"""
processing.py
Core analysis logic for the Sales Pipeline Analyzer.

Provides data preprocessing and analysis functions used by the Streamlit UI
in `app.py`. Charts follow a minimal Telkomsel-red theme for a professional look.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

sns.set(style="whitegrid")
# Theme colors (single accent to keep visuals consistent and professional)
TELKOMSEL_RED = "#e60000"
NEUTRAL_GRAY = "#d9d9d9"

# Consistent minimal palette: Telkomsel Red primary
sns.set_palette([TELKOMSEL_RED])

# ---- Canonical columns used internally ----
CANONICAL_COLS = [
    "Opportunity Name",
    "Account Name",
    "Stage",
    "Schedule Amount",
    "Schedule Date",
    "Created Date",
    "Close Date",
    "Last Stage Change Date",
    "Last Modified Date",
    "Industry Segment",
    "Pilar",
    "AM Name",
    "Amount",
    "Opportunity Owner",
    "Close Date (Year)",
]

# ---- Common aliases for auto-mapping ----
ALIASES = {
    "Opportunity Name": ["opportunity", "opportunity name", "opp name"],
    "Account Name": ["account", "account name", "customer", "company"],
    "Stage": ["stage", "deal stage", "status"],
    "Schedule Amount": [
        "schedule amount",
        "amount",
        "contract value",
        "cv",
        "value",
        "deal size",
        "total amount",
    ],
    "Schedule Date": ["schedule date", "forecast date"],
    "Created Date": ["created date", "creation date", "created"],
    "Close Date": ["close date", "closed_date", "closing date"],
    "Last Stage Change Date": ["last stage change date", "last_stage_date"],
    "Last Modified Date": ["last modified date", "modified date", "update date"],
    "Industry Segment": [
        "industry segment",
        "industry",
        "segment",
        "lob",
        "line of business",
    ],
    "Pilar": ["pilar", "pillar", "product pillar"],
    "AM Name": ["am name", "account manager", "am", "owner"],
    "Amount": ["amount", "cv", "contract value", "value"],
    "Opportunity Owner": ["opportunity owner", "se", "sales engineer", "owner"],
    "Close Date (Year)": ["close date (year)", "close year", "year closed"],
}

# ---- LoB mapping from your notebook (bagian2) ----
LOB_MAPPING = {
    "agriculture": "AGRICULTURE",
    "ict, transportation": "LOGISTIC & TRANSPORTATION",
    "logisctic & transportation": "LOGISTIC & TRANSPORTATION",
    "manufacturing, logistic, transportation": "LOGISTIC & TRANSPORTATION",
    "construction infra": "MANUFACTURING",
    "ict, manufacturing": "MANUFACTURING",
    "manufacturing": "MANUFACTURING",
    "manufacturing & retail": "MANUFACTURING",
    "manufacturing, agriculture & logistic": "MANUFACTURING",
    "manufacturing, construction infra": "MANUFACTURING",
    "manufacturing, retail": "MANUFACTURING",
    "private conglomeration, manufacturing": "MANUFACTURING",
    "energy": "MINING & ENERGY",
    "energy, manufacturing": "MINING & ENERGY",
    "mining & energy": "MINING & ENERGY",
    "resource": "MINING & ENERGY",
    "resource, energy": "MINING & ENERGY",
    "resource, energy, hospitality": "MINING & ENERGY",
    "oil & gas": "OIL & GAS",
    "education": "GOVERNMENT & EDUCATION",
    "government": "GOVERNMENT & EDUCATION",
    "government & education": "GOVERNMENT & EDUCATION",
    "government & public services": "GOVERNMENT & EDUCATION",
    "government, education": "GOVERNMENT & EDUCATION",
    "government, utility": "GOVERNMENT & EDUCATION",
    "government, utility, construction infra": "GOVERNMENT & EDUCATION",
    "hospitality & healthcare": "GOVERNMENT & EDUCATION",
    "hospitality & private conglomeration": "GOVERNMENT & EDUCATION",
    "ict": "GOVERNMENT & EDUCATION",
    "ict, digital creative": "GOVERNMENT & EDUCATION",
    "ict, healthcare": "GOVERNMENT & EDUCATION",
    "ict, media": "GOVERNMENT & EDUCATION",
    "media, telecommunication & property": "GOVERNMENT & EDUCATION",
    "financial": "FINANCIAL",
    "financial, general telco, private conglomeration": "FINANCIAL",
    "fsi banking": "FINANCIAL",
    "fsi non bank, retail & digital native": "FINANCIAL",
    "ict, financial": "FINANCIAL",
    "ict, financial tech": "FINANCIAL",
    "ict, general telco": "FINANCIAL",
    "ict & retail": "RETAIL",
    "ict, distribution, retail": "RETAIL",
    "ict, retail": "RETAIL",
    "ict, retail, education, healthcare": "RETAIL",
    "retail": "RETAIL",
    "retail & private conglomeration": "RETAIL",
    "retail, conglomeration": "RETAIL",
    "retail, private conglomeration": "RETAIL",
}

# ---- Helpers ----
def auto_map_columns(found_cols: List[str]) -> Dict[str, Optional[str]]:
    """Return a best-effort mapping from available columns to canonical names.

    Parameters
    ----------
    found_cols : list[str]
        Columns found in the uploaded dataset.

    Returns
    -------
    dict[str, Optional[str]]
        Mapping from canonical name to source column (or None if not found).
    """
    mapping: Dict[str, Optional[str]] = {c: None for c in CANONICAL_COLS}
    normalized = {c.lower().strip(): c for c in found_cols}
    # exact name first
    for canon in CANONICAL_COLS:
        key = canon.lower()
        if key in normalized:
            mapping[canon] = normalized[key]
    # alias pass
    for canon, aliases in ALIASES.items():
        if mapping.get(canon):
            continue
        for alias in aliases:
            if alias.lower() in normalized:
                mapping[canon] = normalized[alias.lower()]
                break
    return mapping

def classify_product_type(pilar: Optional[str]) -> str:
    """Classify product pillar into a simplified product type."""
    if pd.isna(pilar):
        return "Unknown"
    p = str(pilar).lower()
    if "advanced connectivity" in p or "fixed connectivity & iptv" in p or "network infra" in p:
        return "Connectivity"
    elif (
        "customer engagement and experience" in p
        or "enterprise iot industrial" in p
        or "fleet" in p
        or "iot industrial solution (horizontal)" in p
        or "iot industrial solution (vertical)" in p
        or "iot security, ai and analytic" in p
        or "manufacturing, logistic & transport" in p
        or "mobile security & emerging solution" in p
        or "ucc and business productivity" in p
    ):
        return "Solution"
    else:
        return "Other"

def classify_year_period(date: Optional[pd.Timestamp]) -> str:
    """Bucket a date into year/period labels used for charts."""
    if pd.isna(date):
        return "Unknown"
    if date.year == 2023:
        return "2023"
    elif date.year == 2024:
        return "2024"
    elif date.year == 2025 and date.month <= 6:
        return "H1 2025"
    else:
        return "Other"

def idr_label(value: float) -> str:
    """Format a numeric value into human-readable Rupiah units (Bn/Mio)."""
    if value >= 1e9:
        return f"{value / 1e9:.2f} Bn"
    elif value >= 1e6:
        return f"{value / 1e6:.2f} Mio"
    else:
        return f"{value:,.0f}"

# ---- Preprocess shared (normalize columns and types) ----
def preprocess(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Normalize and enrich the dataset according to the provided column mapping.

    - Renames mapped columns to canonical names
    - Normalizes industry segment (lower/strip)
    - Parses dates and derives Year/Quarter fields for scheduling
    - Derives product type from Pilar (if present)
    """
    out = df.copy()
    # rename to canonical if mapped
    rename_map = {v: k for k, v in mapping.items() if v}
    out = out.rename(columns=rename_map)

    # lower/strip industry
    if "Industry Segment" in out.columns:
        out["Industry Segment"] = out["Industry Segment"].astype(str).str.strip().str.lower()

    # parse datetimes if present
    for col in ["Created Date", "Close Date", "Schedule Date", "Last Modified Date", "Last Stage Change Date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", dayfirst=True)

    # derive year if missing but Close Date exists
    if "Close Date (Year)" not in out.columns and "Close Date" in out.columns:
        out["Close Date (Year)"] = out["Close Date"].dt.year

    # product type from Pilar
    if "Pilar" in out.columns:
        out["Product Type"] = out["Pilar"].apply(classify_product_type)

    # quarter/year from Schedule Date for bagian1 charts
    if "Schedule Date" in out.columns:
        out["Year"] = out["Schedule Date"].dt.year
        out["Quarter"] = "Q" + out["Schedule Date"].dt.quarter.astype("Int64").astype(str)
        out["Year Category"] = out["Schedule Date"].apply(classify_year_period)

    return out

# ---- Bagian 1 (preserving your logic) ----
def run_bagian1(
    df: pd.DataFrame,
    industry_filter: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Compute Bagian 1 analyses and figures.

    Returns a dict with keys like: summary, figures, pivot_table, top5, bottom5,
    open_pipeline, warnings.
    """
    res: Dict[str, object] = {}
    if "Industry Segment" not in df.columns or "Schedule Amount" not in df.columns:
        res["warnings"] = ["Bagian 1 membutuhkan minimal kolom: Industry Segment, Schedule Amount."]
        return res

    # Filter segments if provided
    df_seg = df.copy()
    if industry_filter:
        lowered = [s.strip().lower() for s in industry_filter]
        df_seg = df_seg[df_seg["Industry Segment"].isin(lowered)].copy()
    # Drop NA in key fields like original
    drop_keys = [c for c in ["Industry Segment", "Schedule Amount", "Created Date", "Close Date"] if c in df_seg.columns]
    if drop_keys:
        df_seg = df_seg.dropna(subset=drop_keys)

    # Product type (already created in preprocess if Pilar exists)
    if "Pilar" in df_seg.columns and "Product Type" not in df_seg.columns:
        df_seg["Product Type"] = df_seg["Pilar"].apply(classify_product_type)

    # Summary stats
    total_cv = round(df_seg.get("Schedule Amount", pd.Series(dtype=float)).sum() / 1e9, 2)
    won_df = df_seg[df_seg.get("Stage", pd.Series(dtype=object)) == "Closed Won"].copy()
    won_cv_total = round(won_df.get("Schedule Amount", pd.Series(dtype=float)).sum() / 1e9, 2)
    if {"Opportunity Name", "Stage"}.issubset(df_seg.columns):
        opps_all = df_seg[["Opportunity Name", "Stage"]].drop_duplicates()
        total_opps = opps_all["Opportunity Name"].nunique()
        won_opps = opps_all[opps_all["Stage"] == "Closed Won"]["Opportunity Name"].nunique()
        conversion_rate = round((won_opps / total_opps) * 100, 2) if total_opps else 0.0
    else:
        total_opps = won_opps = 0
        conversion_rate = 0.0
    res["summary"] = {
        "total_cv_bn": total_cv,
        "won_cv_bn": won_cv_total,
        "total_opps": total_opps,
        "won_opps": won_opps,
        "conversion_rate": conversion_rate,
    }

    # Pie charts depend on Product Type, Schedule Amount
    figs: Dict[str, plt.Figure] = {}
    if {"Product Type", "Schedule Amount"}.issubset(df_seg.columns):
        cv_total = df_seg.groupby("Product Type")["Schedule Amount"].sum()
        cv_won = won_df.groupby("Product Type")["Schedule Amount"].sum()
        cv_summary = pd.DataFrame({"CV_Total": cv_total, "CV_Won": cv_won}).fillna(0).reset_index()
        cv_summary["CV_Total_Bn"] = (cv_summary["CV_Total"] / 1e9).round(2)
        cv_summary["CV_Won_Bn"] = (cv_summary["CV_Won"] / 1e9).round(2)
        # neutral greys to avoid too many colors while keeping categories distinguishable
        colors = plt.cm.Greys(np.linspace(0.4, 0.9, len(cv_summary)))

        # All stages pie
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.pie(cv_summary["CV_Total"], labels=[f"{pt} ({val:.2f}Bn)" for pt, val in zip(cv_summary["Product Type"], cv_summary["CV_Total_Bn"])],
                autopct="%1.1f%%", startangle=140, colors=colors)
        ax1.set_title("CV by Product Type (All Stage)")
        ax1.axis("equal")
        figs["pie_all"] = fig1

        # Won only pie
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.pie(cv_summary["CV_Won"], labels=[f"{pt} ({val:.2f}Bn)" for pt, val in zip(cv_summary["Product Type"], cv_summary["CV_Won_Bn"])],
                autopct="%1.1f%%", startangle=140, colors=colors)
        ax2.set_title("CV by Product Type (Won Only)")
        ax2.axis("equal")
        figs["pie_won"] = fig2
    else:
        res.setdefault("warnings", []).append("Lewati Pie Chart: membutuhkan Pilar→Product Type dan Schedule Amount.")

    # Quarterly bars depend on Year, Quarter, Schedule Amount, Stage
    if {"Year", "Quarter", "Schedule Amount"}.issubset(df_seg.columns):
        quarterly = df_seg.groupby(["Year", "Quarter"])["Schedule Amount"].sum().reset_index()
        won_quarterly = won_df.groupby(["Year", "Quarter"])["Schedule Amount"].sum().reset_index()
        merged_q = pd.merge(quarterly, won_quarterly, on=["Year", "Quarter"], how="left", suffixes=("_Total", "_Won")).fillna(0)
        merged_q["Label"] = merged_q["Year"].astype(str) + " " + merged_q["Quarter"]
        merged_q["CV_Total_Bn"] = (merged_q["Schedule Amount_Total"] / 1e9).round(2)
        merged_q["CV_Won_Bn"] = (merged_q["Schedule Amount_Won"] / 1e9).round(2)

        fig3, ax = plt.subplots(figsize=(16, 7))
        bar_width = 0.35
        x = np.arange(len(merged_q))
        color_all = NEUTRAL_GRAY
        color_won = TELKOMSEL_RED
        ax.bar(x - bar_width/2, merged_q["CV_Total_Bn"], width=bar_width, label="All Stage", color=color_all)
        ax.bar(x + bar_width/2, merged_q["CV_Won_Bn"], width=bar_width, label="Won Only", color=color_won)
        for i in x:
            ax.text(i - bar_width/2, merged_q["CV_Total_Bn"].iloc[i] + 0.05, f'{merged_q["CV_Total_Bn"].iloc[i]:.1f}Bn', ha="center", fontsize=9)
            ax.text(i + bar_width/2, merged_q["CV_Won_Bn"].iloc[i] + 0.05, f'{merged_q["CV_Won_Bn"].iloc[i]:.1f}Bn', ha="center", fontsize=9)
        for i in range(len(x)-1):
            if merged_q["Year"].iloc[i] != merged_q["Year"].iloc[i+1]:
                ax.axvline(x=i+0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x, merged_q["Label"], rotation=45)
        ax.set_ylabel("CV (Bn)")
        ax.set_title("CV per Quarter - All Stage vs Won Only")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig3.tight_layout()
        figs["bars_quarterly"] = fig3
    else:
        res.setdefault("warnings", []).append("Lewati Bar Chart Quarter: membutuhkan Schedule Date→(Year, Quarter) dan Schedule Amount.")

    # Heatmap-like pivot as table (opportunity count)
    if {"Year", "Quarter", "Product Type", "Stage", "Opportunity Name"}.issubset(df_seg.columns):
        df_seg["Stage Category"] = df_seg["Stage"].apply(lambda x: "Won Only" if x == "Closed Won" else "All Stage")
        pivot = df_seg.groupby(["Year", "Quarter", "Product Type", "Stage Category"])["Opportunity Name"].nunique().reset_index()
        pivot_table = pivot.pivot_table(
            index=["Year", "Quarter"], columns=["Product Type", "Stage Category"], values="Opportunity Name", fill_value=0
        ).astype(int)
        pivot_table.columns = [" ".join(col).strip() for col in pivot_table.columns.values]
        pivot_table = pivot_table.reset_index()
        res["pivot_table"] = pivot_table
    else:
        res.setdefault("warnings", []).append("Lewati Tabel Opportunity Count: membutuhkan Year, Quarter, Product Type, Stage, Opportunity Name.")

    # Top 5 / Bottom 5 per Close Year
    if {"Close Date (Year)", "Opportunity Name", "Account Name", "Close Date", "Schedule Amount"}.issubset(df_seg.columns):
        won = df_seg[df_seg["Stage"] == "Closed Won"].copy()
        won["Close Year Category"] = won["Close Date (Year)"].astype(str)
        tops: Dict[str, pd.DataFrame] = {}
        bottoms: Dict[str, pd.DataFrame] = {}
        for year_cat in sorted(won["Close Year Category"].dropna().unique()):
            yearly = won[won["Close Year Category"] == year_cat]
            grp = yearly.groupby(["Opportunity Name", "Account Name", "Close Date"]).agg(CV=("Schedule Amount", "sum")).reset_index()
            top5 = grp.sort_values("CV", ascending=False).head(5).copy()
            bottom5 = grp.sort_values("CV", ascending=True).head(5).copy()
            top5["CV (Rupiah)"] = top5["CV"].apply(idr_label)
            bottom5["CV (Rupiah)"] = bottom5["CV"].apply(idr_label)
            tops[year_cat] = top5.drop(columns="CV").reset_index(drop=True)
            bottoms[year_cat] = bottom5.drop(columns="CV").reset_index(drop=True)
        res["top5"] = tops
        res["bottom5"] = bottoms
    else:
        res.setdefault("warnings", []).append("Lewati Top/Bottom 5: membutuhkan Close Date (Year), Opportunity Name, Account Name, Close Date, Schedule Amount.")

    # Open pipeline by Close Year
    if {"Stage", "Close Date", "Schedule Amount", "Opportunity Name", "Account Name"}.issubset(df_seg.columns):
        now = pd.Timestamp.now()
        open_df = df_seg[(df_seg["Stage"] != "Closed Won") & (df_seg["Close Date"] >= now)].copy()
        open_df["Close Year"] = open_df["Close Date"].dt.year.astype(str)
        open_df["CV (Rupiah)"] = open_df["Schedule Amount"].apply(idr_label)
        per_year: Dict[str, pd.DataFrame] = {}
        for year in sorted(open_df["Close Year"].unique()):
            yearly = open_df[open_df["Close Year"] == year]
            cols = ["Opportunity Name", "Account Name", "Close Date", "CV (Rupiah)"]
            if "Product Type" in yearly.columns:
                cols.insert(3, "Product Type")
            per_year[year] = yearly[cols].reset_index(drop=True)
        res["open_pipeline"] = per_year
    else:
        res.setdefault("warnings", []).append("Lewati Open Pipeline: membutuhkan Stage, Close Date, Schedule Amount, Opportunity Name, Account Name.")

    res["figures"] = figs
    return res

# ---- Bagian 2 (preserving your logic) ----
def run_bagian2(df: pd.DataFrame) -> Dict[str, object]:
    """Compute Bagian 2 analyses and figures (cycle, stage durations, SE, win rate, Top 5 AM)."""
    res: Dict[str, object] = {}
    figs: Dict[str, plt.Figure] = {}

    # Normalize for bagian2 renaming logic
    df2 = df.copy()
    # Create df columns expected by bagian2
    rename_map = {
        "Opportunity Name": "opportunity",
        "Account Name": "account",
        "Stage": "stage",
        "Close Date": "closed_date",
        "Created Date": "created_date",
        "Last Stage Change Date": "last_stage_date",
        "AM Name": "am",
        "Schedule Amount": "cv",
        "Industry Segment": "industry_segment",
        "Pilar": "Pilar",
        "Opportunity Owner": "Opportunity Owner",
    }
    for k, v in rename_map.items():
        if k in df2.columns:
            df2[v] = df2[k]

    # Required checks
    needed = {"stage", "created_date", "closed_date", "industry_segment"}
    missing = [c for c in needed if c not in df2.columns]
    if missing:
        res["warnings"] = [f"Lewati sebagian Bagian 2: kolom wajib tidak ada: {', '.join(missing)}"]
        return res

    # Parse types (already parsed in preprocess but ensure)
    for col in ["created_date", "closed_date", "last_stage_date"]:
        if col in df2.columns:
            df2[col] = pd.to_datetime(df2[col], errors="coerce")

    # LoB mapping
    df2["lob_raw"] = df2["industry_segment"].astype(str).str.strip().str.lower()
    df2["lob"] = df2["lob_raw"].map(LOB_MAPPING)
    df2 = df2[df2["lob"].notna()].copy()

    # 2. Average sales cycle time per LoB (Closed Won only)
    cycle_results = []
    for lob, group in df2[df2["stage"].str.lower() == "closed won"].groupby("lob"):
        group = group.copy()
        group["sales_cycle_days"] = (group["closed_date"] - group["created_date"]).dt.days
        valid = group["sales_cycle_days"] >= 0
        avg_days = group.loc[valid, "sales_cycle_days"].mean()
        cycle_results.append((lob, avg_days, valid.sum(), len(group)))
    df_cycle = pd.DataFrame(cycle_results, columns=["LoB", "Avg Sales Cycle (days)", "Valid Count", "Total Count"])
    df_cycle["% Valid"] = round(100 * df_cycle["Valid Count"] / df_cycle["Total Count"], 1)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_cycle, y="LoB", x="Avg Sales Cycle (days)", hue="LoB", legend=False, palette=[TELKOMSEL_RED], ax=ax1)
    ax1.set_title("Average Sales Cycle Time per LoB")
    ax1.set_xlabel("Average Days")
    ax1.set_ylabel("LoB")
    fig1.tight_layout()
    figs["cycle_per_lob"] = fig1

    # 3. Average Stage -> Close per LoB and Product Type
    df_valid_stage = df2.copy()
    if "last_stage_date" in df_valid_stage.columns:
        df_valid_stage = df_valid_stage[df_valid_stage["closed_date"].notna() & df_valid_stage["last_stage_date"].notna()].copy()
        df_valid_stage["stage_duration"] = (df_valid_stage["closed_date"] - df_valid_stage["last_stage_date"]).dt.days
        df_valid_stage = df_valid_stage[df_valid_stage["stage_duration"] >= 0]
        # Product Type
        if "Pilar" in df_valid_stage.columns:
            df_valid_stage["Product Type"] = df_valid_stage["Pilar"].apply(classify_product_type)
        else:
            df_valid_stage["Product Type"] = "Unknown"

        lob_product = df_valid_stage.groupby(["lob", "Product Type"])["stage_duration"].mean().reset_index()
        lob_product = lob_product.sort_values(by="stage_duration", ascending=False)
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.barplot(data=lob_product, y="lob", x="stage_duration", hue="Product Type", palette=[TELKOMSEL_RED], ax=ax2)
        ax2.set_title("Avg Stage to Close Time per LoB and Product Type")
        ax2.set_xlabel("Average Days")
        ax2.set_ylabel("Industry Segment (LoB)")
        ax2.legend(title="Product Type")
        fig2.tight_layout()
        figs["stage_to_close_lob_product"] = fig2
    else:
        res.setdefault("warnings", []).append("Lewati Stage→Close per LoB/Product: Last Stage Change Date tidak tersedia.")

    # SE (Opportunity Owner)
    if "Opportunity Owner" in df_valid_stage.columns and "stage_duration" in df_valid_stage.columns:
        se_stage = df_valid_stage.groupby("Opportunity Owner")["stage_duration"].mean().reset_index()
        se_stage = se_stage.sort_values(by="stage_duration", ascending=False)
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sns.barplot(data=se_stage, y="Opportunity Owner", x="stage_duration", hue="Opportunity Owner", legend=False, palette=[TELKOMSEL_RED], ax=ax3)
        ax3.set_title("Avg Stage to Close Time per Opportunity Owner (SE)")
        ax3.set_xlabel("Average Days")
        ax3.set_ylabel("Opportunity Owner (SE)")
        fig3.tight_layout()
        figs["stage_to_close_per_se"] = fig3

    # 4. Win rate per LoB
    win_results = []
    for lob, group in df2.groupby("lob"):
        total = len(group)
        won = len(group[group["stage"].str.lower() == "closed won"])
        win_rate = 100 * won / total if total else 0
        win_results.append((lob, total, won, win_rate))
    df_win = pd.DataFrame(win_results, columns=["LoB", "Total Deals", "Closed Won", "Win Rate (%)"]).sort_values(by="Win Rate (%)", ascending=False)
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_win, y="LoB", x="Win Rate (%)", hue="LoB", legend=False, palette=[TELKOMSEL_RED], ax=ax4)
    ax4.set_title("Win Rate per LoB")
    ax4.set_xlabel("Win Rate (%)")
    ax4.set_ylabel("Industry Segment")
    fig4.tight_layout()
    figs["win_rate_per_lob"] = fig4

    # Top 5 AM plots (per LoB per Year)
    if "closed_date" in df2.columns and "am" in df2.columns and "opportunity" in df2.columns:
        df2["year"] = df2["closed_date"].dt.year
        am_list = []
        won_mask = df2["stage"].str.lower() == "closed won"
        for (lob, year), group in df2[won_mask].groupby(["lob", "year"]):
            all_deals = df2[(df2["lob"] == lob) & (df2["year"] == year)].groupby("am")["opportunity"].count()
            perf = group.groupby("am").agg(won=("opportunity", "count"), total_cv=("cv", "sum")).reset_index()
            perf["total"] = perf["am"].map(all_deals)
            perf = perf[perf["total"].notna()].copy()
            perf["conversion_rate"] = 100 * perf["won"] / perf["total"]
            perf["LoB"] = lob
            perf["Year"] = year
            am_list.append(perf)
        if am_list:
            df_am = pd.concat(am_list, ignore_index=True)

            def catplot_top5(sort_by: str, title_suffix: str) -> plt.Figure:
                top5_list = []
                for (lob, year), grp in df_am.groupby(["LoB", "Year"]):
                    top5 = grp.sort_values(by=sort_by, ascending=False).head(5)
                    top5_list.append(top5)
                df_plot = pd.concat(top5_list, ignore_index=True) if top5_list else pd.DataFrame(columns=df_am.columns)
                g = sns.catplot(
                    data=df_plot,
                    x=sort_by,
                    y="am",
                    hue="Year",
                    col="LoB",
                    kind="bar",
                    height=4.8,
                    aspect=2.5,
                    palette={2023: TELKOMSEL_RED, 2024: TELKOMSEL_RED, 2025: TELKOMSEL_RED},
                    sharex=False,
                    sharey=False,
                    col_wrap=2,
                    legend_out=True,
                )
                g.set_titles("Top AM in {col_name}")
                g.set_axis_labels(title_suffix, "Account Manager")
                for ax in g.axes.flatten():
                    for bar in ax.patches:
                        width = bar.get_width()
                        if pd.notna(width):
                            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, _format_label(width, sort_by), va="center", ha="left", fontsize=9)
                    for spine in ["top", "right", "left"]:
                        ax.spines[spine].set_visible(False)
                    ax.grid(axis="y", linestyle="--", alpha=0.3)
                plt.suptitle(f"Top 5 AM based on {title_suffix} per LoB per Year", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                return g.fig

            def _format_label(value, metric):
                if pd.isna(value):
                    return ""
                if metric == "conversion_rate":
                    return f"{value:.1f}%"
                elif metric == "total_cv":
                    return idr_label(value)
                elif metric == "total":
                    return f"{int(value)}"
                return str(value)

            figs["top5_am_conversion"] = catplot_top5("conversion_rate", "Conversion Rate (%)")
            figs["top5_am_total_cv"] = catplot_top5("total_cv", "Total Contract Value (Rupiah)")
            figs["top5_am_total"] = catplot_top5("total", "Total Deals (Opportunity Count)")
            # Expose for Streamlit UI to build custom, readable views
            res.setdefault("tables", {})["top5_am_df"] = df_am.copy()
        else:
            res.setdefault("warnings", []).append("Lewati Top 5 AM: tidak ada data Closed Won yang cukup.")
    else:
        res.setdefault("warnings", []).append("Lewati Top 5 AM: butuh closed_date, am, opportunity.")

    res["figures"] = figs
    # Merge tables so previously added entries (e.g., top5_am_df) are preserved
    tables = res.get("tables", {})
    tables.update({
        "cycle_per_lob": df_cycle,
        "win_rate_per_lob": df_win,
    })
    res["tables"] = tables
    return res