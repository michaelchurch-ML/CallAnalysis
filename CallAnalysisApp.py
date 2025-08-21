#!/usr/bin/env python3
"""
CallAnalysisApp.py — Streamlit app (password gate, two segments, totals & rates)

Security
- Simple password gate using env var MELON_APP_PASSWORD.
  The app will not load until the user enters the correct password.

Features
- Env vars: MELON_APP_PASSWORD, MELON_CLIENT_ID, MELON_CLIENT_SECRET
- Date pickers + Fetch button
- Client Name filter (ALL) and two Segment dropdowns (A & B; either can be None)
- Results table: Segment, Calls, Qualified Calls, Cost, 5-min, 10-min, Missed
- Summary split into two tables that always stay at the bottom:
    • TOTAL — whole-number sums (Cost has 2 decimals)
    • RATES — per-call/percentage metrics (percent columns correctly scaled)

Run:
  export MELON_APP_PASSWORD="your_password"
  export MELON_CLIENT_ID="..."
  export MELON_CLIENT_SECRET="..."
  pip install streamlit requests pandas numpy python-dateutil
  streamlit run CallAnalysisApp.py
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =================== Config ===================
st.set_page_config(page_title="Melon Calls Dashboard", layout="wide")

API_TOKEN_URL = "https://reporting.qmp.ai/oauth/generatetoken?grant_type=client_credentials"
API_REPORT_BASE = "https://reporting.qmp.ai/api/client/download/68980"

CLIENT_ID_ENV = "MELON_CLIENT_ID"
CLIENT_SECRET_ENV = "MELON_CLIENT_SECRET"
APP_PASSWORD_ENV = "MELON_APP_PASSWORD"


# =================== Password Gate ===================
def require_password() -> bool:
    """Block app unless the user enters the correct password from env var MELON_APP_PASSWORD."""
    expected = os.getenv(APP_PASSWORD_ENV, "")
    if not expected:
        st.title("Melon Calls Dashboard")
        st.error(f"Missing password. Set env var {APP_PASSWORD_ENV}.")
        return False

    # If already authed, don't render the page title here to avoid duplicates
    if st.session_state.get("is_authed"):
        with st.sidebar:
            st.caption("🔒 Access granted")
            if st.button("Sign out"):
                for k in ("is_authed", "pw_input"):
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
        return True

    # Not authed: render a single title + login form
    st.title("Melon Calls Dashboard")
    with st.form("login_form", clear_on_submit=False):
        st.subheader("Enter password to continue")
        pw = st.text_input("Password", type="password", key="pw_input")
        ok = st.form_submit_button("Sign in", use_container_width=True)
        if ok:
            if pw == expected:
                st.session_state.is_authed = True
                st.success("Signed in")
                st.rerun()
            else:
                st.error("Incorrect password")
    return False

    if st.session_state.get("is_authed"):
        with st.sidebar:
            st.caption("🔒 Access granted")
            if st.button("Sign out"):
                for k in ("is_authed", "pw_input"):
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
        return True

    with st.form("login_form", clear_on_submit=False):
        st.subheader("Enter password to continue")
        pw = st.text_input("Password", type="password", key="pw_input")
        ok = st.form_submit_button("Sign in", use_container_width=True)
        if ok:
            if pw == expected:
                st.session_state.is_authed = True
                st.success("Signed in")
                st.rerun()
            else:
                st.error("Incorrect password")
    return False


# =================== API Helpers ===================
def _env_creds() -> Tuple[str, str]:
    cid = os.getenv(CLIENT_ID_ENV, "")
    csec = os.getenv(CLIENT_SECRET_ENV, "")
    if not cid or not csec:
        raise RuntimeError(
            f"Missing API credentials. Set env vars {CLIENT_ID_ENV} and {CLIENT_SECRET_ENV}."
        )
    return cid, csec


@st.cache_data(show_spinner=False)
def get_access_token(client_id: str, client_secret: str) -> str:
    r = requests.post(API_TOKEN_URL, auth=(client_id, client_secret), timeout=30)
    r.raise_for_status()
    data = r.json()
    token = data.get("access_token") or data.get("accessToken")
    if not token:
        raise RuntimeError("Did not receive access token from token endpoint")
    return token


@st.cache_data(show_spinner=False)
def fetch_report(token: str, start_date: Optional[date], end_date: Optional[date]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if start_date:
        params["startDate"] = start_date.strftime("%Y-%m-%d")
    if end_date:
        params["endDate"] = end_date.strftime("%Y-%m-%d")
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(API_REPORT_BASE, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def normalize_payload(payload: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str], Any]:
    """Turn API payload into a tidy DataFrame and clean column names."""
    block = payload.get("data", payload)
    columns: List[str] = block.get("columns") or payload.get("columns") or []
    columns = [str(c).strip() for c in columns]

    records = block.get("records") or payload.get("records") or block.get("data") or []

    out_rows: List[Dict[str, Any]] = []
    for rec in records or []:
        row: Dict[str, Any] = {}
        if hasattr(rec, "items"):
            for k, v in rec.items():
                sk = str(k).strip()
                if sk.isdigit():
                    idx = int(sk)
                    name = columns[idx] if idx < len(columns) else sk
                    row[name] = v
                else:
                    row[sk] = v
        out_rows.append(row)

    df = pd.DataFrame(out_rows)
    df.columns = [str(c).strip() for c in df.columns]

    return df, columns, payload.get("lastRefreshedOn")


def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first column in df matching any candidate (case-insensitive, punctuation-insensitive)."""
    if df is None or df.empty:
        return None
    norm = {str(c).lower().replace(" ", "").replace("_", ""): str(c) for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in norm:
            return norm[key]
    for key, orig in norm.items():
        for cand in candidates:
            ckey = cand.lower().replace(" ", "").replace("_", "")
            if ckey in key:
                return orig
    return None


def aggregate(df: pd.DataFrame, client_filter: str, seg_a: str, seg_b: str) -> Tuple[pd.DataFrame, Optional[str]]:
    # Resolve important columns (robust to naming variants)
    calls_col = _resolve_column(df, ["calls", "call_count", "total_calls"])
    qual_col = _resolve_column(df, ["qualified_call", "qualifiedcalls", "qualified"])
    cost_col = _resolve_column(df, ["advertiser_cost", "cost", "spend", "amount"])
    dur_col = _resolve_column(df, ["call_duration", "duration_seconds", "duration", "call_duration_seconds"])
    client_col = _resolve_column(df, ["client_name", "client", "advertiser_name", "customer", "account_name"])

    df_work = df.copy()

    # Filter by Client
    if client_filter != "ALL" and client_col and client_col in df_work.columns:
        df_work = df_work[df_work[client_col].astype(str) == str(client_filter)]

    # Build Segment label from up to two fields
    def _seg_series(field: Optional[str]) -> pd.Series:
        if not field or field == "None" or field not in df_work.columns:
            return pd.Series(["None"] * len(df_work), index=df_work.index, dtype="object")
        return df_work[field].astype(str).fillna("None")

    sA = _seg_series(seg_a)
    sB = _seg_series(seg_b)

    if (seg_b and seg_b != "None" and seg_b in df_work.columns) or (seg_a and seg_a != "None" and seg_a in df_work.columns):
        seg_combined = np.where((sB != "None") & (sA != "None"), sA + " | " + sB,
                                np.where(sA != "None", sA, sB))
    else:
        seg_combined = pd.Series(["None"] * len(df_work), index=df_work.index, dtype="object")

    df_work = df_work.assign(Segment=seg_combined)

    # Numeric series with safe defaults
    calls = pd.to_numeric(df_work.get(calls_col, 0), errors="coerce").fillna(0)
    qual = pd.to_numeric(df_work.get(qual_col, 0), errors="coerce").fillna(0)
    cost = pd.to_numeric(df_work.get(cost_col, 0), errors="coerce").fillna(0.0)
    dur = pd.to_numeric(df_work.get(dur_col, 0), errors="coerce").fillna(0)

    calc = pd.DataFrame(
        {
            "Segment": df_work["Segment"],
            "Calls": calls,
            "Qualified Calls": qual,
            "Cost": cost,
            "5-min": np.where(dur > 299, qual, 0),
            "10-min": np.where(dur > 599, qual, 0),
            "Missed": np.where(dur <= 0, calls, 0),
        }
    )

    agg = calc.groupby("Segment", dropna=False, as_index=False).sum(numeric_only=True)
    agg = agg.sort_values("Calls", ascending=False, ignore_index=True)
    return agg, client_col


# =================== Main App ===================
if not require_password():
    st.stop()

st.title("Melon Calls Dashboard")

# Initialize session state
for key, default in [
    ("df", None),
    ("columns", None),
    ("refreshed", None),
    ("client_options", ["ALL"]),
    ("segment_options", ["None"]),
    ("client_sel", "ALL"),
    ("segment_a_sel", "None"),
    ("segment_b_sel", "None"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Inputs
left, mid, right = st.columns([1, 1, 0.6])
with left:
    start_default = date.today() - timedelta(days=7)
    start_date = st.date_input("Start Date", value=start_default, format="YYYY-MM-DD")
with mid:
    end_default = date.today()
    end_date = st.date_input("End Date", value=end_default, format="YYYY-MM-DD")
with right:
    fetch_clicked = st.button("Fetch data", type="primary", use_container_width=True)

st.divider()

# Fetch
if fetch_clicked:
    try:
        client_id, client_secret = _env_creds()
        with st.spinner("Authenticating…"):
            token = get_access_token(client_id, client_secret)
        with st.spinner("Fetching report…"):
            payload = fetch_report(token, start_date, end_date)

        df, columns, refreshed = normalize_payload(payload)
        if df.empty:
            st.info("No rows returned for the selected dates.")
            st.stop()

        st.session_state.df = df
        st.session_state.columns = columns or df.columns.tolist()
        st.session_state.refreshed = refreshed

        client_col_guess = _resolve_column(
            df, ["client_name", "client", "advertiser_name", "customer", "account_name"]
        )
        client_options = ["ALL"]
        if client_col_guess and client_col_guess in df.columns:
            client_values = df[client_col_guess].dropna().astype(str).unique().tolist()
            client_options += sorted(client_values)
        st.session_state.client_options = client_options

        seg_union = ["None"] + list(dict.fromkeys([str(c) for c in (list(df.columns) + (columns or []))]))
        for key in ("segment_a_sel", "segment_b_sel"):
            if st.session_state.get(key) not in seg_union:
                seg_union.append(st.session_state.get(key, "None"))
        st.session_state.segment_options = seg_union

        st.session_state.client_sel = "ALL"
        st.session_state.segment_a_sel = "None"
        st.session_state.segment_b_sel = "None"

    except Exception as e:
        st.error(f"Error during fetch: {e}")
        st.stop()

# Render
if isinstance(st.session_state.df, pd.DataFrame) and not st.session_state.df.empty:
    st.caption(
        f"Loaded **{len(st.session_state.df):,}** rows · Last refreshed: **{st.session_state.refreshed or 'n/a'}**"
    )

    if st.session_state.client_sel not in st.session_state.client_options:
        st.session_state.client_options = list(st.session_state.client_options) + [st.session_state.client_sel]
    for key in ("segment_a_sel", "segment_b_sel"):
        if st.session_state[key] not in st.session_state.segment_options:
            st.session_state.segment_options = list(st.session_state.segment_options) + [st.session_state[key]]

    def _pin_first(opts, first_label):
        opts_unique, seen = [], set()
        for o in opts:
            if o not in seen:
                seen.add(o); opts_unique.append(o)
        if len(opts_unique)>1 and opts_unique[0]==first_label:
            rest = sorted([o for o in opts_unique[1:] if o!=first_label])
            return [first_label] + rest
        return opts_unique
    st.session_state.client_options = _pin_first(st.session_state.client_options, "ALL")
    st.session_state.segment_options = _pin_first(st.session_state.segment_options, "None")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.selectbox("Client Name", st.session_state.client_options, key="client_sel")
    with c2:
        st.selectbox("Segment A (group by)", st.session_state.segment_options, key="segment_a_sel")
    with c3:
        st.selectbox("Segment B (optional)", st.session_state.segment_options, key="segment_b_sel")

    result, client_col_used = aggregate(
        st.session_state.df, st.session_state.client_sel, st.session_state.segment_a_sel, st.session_state.segment_b_sel
    )

    tot_calls = float(result["Calls"].sum()) if "Calls" in result.columns else 0.0
    tot_qual  = float(result["Qualified Calls"].sum()) if "Qualified Calls" in result.columns else 0.0
    tot_cost  = float(result["Cost"].sum()) if "Cost" in result.columns else 0.0
    tot_5     = float(result["5-min"].sum()) if "5-min" in result.columns else 0.0
    tot_10    = float(result["10-min"].sum()) if "10-min" in result.columns else 0.0
    tot_miss  = float(result["Missed"].sum()) if "Missed" in result.columns else 0.0

    totals_df = pd.DataFrame([{
        "Segment": "TOTAL",
        "Calls": tot_calls,
        "Qualified Calls": tot_qual,
        "Cost": tot_cost,
        "5-min": tot_5,
        "10-min": tot_10,
        "Missed": tot_miss,
    }])

    def rate(n, d): return (n / d) if (d and d != 0) else 0.0

    rates_df = pd.DataFrame([{
        "Segment": "RATES",
        "Calls": tot_calls,
        "Qualified Calls": rate(tot_qual, tot_calls) * 100.0,  # scaled to percent
        "Cost": rate(tot_cost, tot_calls),   # per call
        "5-min": rate(tot_5, tot_calls) * 100.0,
        "10-min": rate(tot_10, tot_calls) * 100.0,
        "Missed": rate(tot_miss, tot_calls) * 100.0,
    }])

    # Main table (sortable) and summaries (fixed at bottom)
    st.subheader("Results")
    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Calls": st.column_config.NumberColumn(format="%.0f"),
            "Qualified Calls": st.column_config.NumberColumn(format="%.0f"),
            "Cost": st.column_config.NumberColumn(format="%.2f"),
            "5-min": st.column_config.NumberColumn(format="%.0f"),
            "10-min": st.column_config.NumberColumn(format="%.0f"),
            "Missed": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Summary — Totals")
        st.dataframe(
            totals_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Calls": st.column_config.NumberColumn(format="%.0f"),
                "Qualified Calls": st.column_config.NumberColumn(format="%.0f"),
                "Cost": st.column_config.NumberColumn(format="%.2f"),
                "5-min": st.column_config.NumberColumn(format="%.0f"),
                "10-min": st.column_config.NumberColumn(format="%.0f"),
                "Missed": st.column_config.NumberColumn(format="%.0f"),
            },
        )
    with c2:
        st.subheader("Summary — Rates")
        st.dataframe(
            rates_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Calls": st.column_config.NumberColumn(format="%.0f"),
                "Qualified Calls": st.column_config.NumberColumn(format="%.2f%%"),
                "Cost": st.column_config.NumberColumn(format="%.2f"),  # per call
                "5-min": st.column_config.NumberColumn(format="%.2f%%"),
                "10-min": st.column_config.NumberColumn(format="%.2f%%"),
                "Missed": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    # CSV download (combined)
    csv_df = pd.concat([result, totals_df, rates_df], ignore_index=True)
    csv = csv_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download table as CSV", csv, file_name="melon_calls_summary.csv", mime="text/csv")

else:
    st.info("Choose dates and click **Fetch data** to load the report.")
