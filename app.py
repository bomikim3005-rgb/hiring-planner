"""Hiring Planner - Streamlit app.

Turns a hiring request (role, headcount, start date, offer) into a plan: the
candidate funnel, recruiter and interviewer load, and a back-planned timeline,
checked against live Bay Area market data, plus internal-mobility options.

Market data comes from the Job Posting Monitor pipeline's dbt models (a separate
project): BigQuery when BQ_PROJECT and GCP_SA_KEY are set, otherwise a local
DuckDB file, JOB_MARKET_DB if set, else ../job-posting-monitor/data/job_market.duckdb.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""
import json
import math
import os
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

LOCAL_DB = Path(os.getenv("JOB_MARKET_DB") or Path(__file__).resolve().parent.parent
                / "job-posting-monitor" / "data" / "job_market.duckdb")
BQ_DATASET = "job_market"

# dbt models the planner reads ({name} is replaced per warehouse).
QUERIES = {
    "postings": """
        select job_id, company, posted_salary_mid, first_seen_at, last_seen_at
        from {dim_postings}""",
    "roles": "select job_id, role from {bridge_posting_roles}",
    "skills": "select skill, role, postings_mentioning from {mart_skill_mentions}",
}
TABLES = ["dim_postings", "bridge_posting_roles", "mart_skill_mentions"]


# ---------------------------------------------------------------- helpers

def role_label(role: str) -> str:
    return role.replace("_", " ").capitalize()


def naive_utc(s: pd.Series) -> pd.Series:
    """Timestamps as tz-naive UTC, whichever warehouse they came from."""
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def ranked_bar(df: pd.DataFrame, value: str, label: str, title: str, fmt: str = ","):
    """Horizontal bar chart sorted by value."""
    if df.empty:
        st.caption("No data yet.")
        return
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{value}:Q", title=title, axis=alt.Axis(format=fmt)),
        y=alt.Y(f"{label}:N", sort="-x", title=None),
        tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip(f"{value}:Q", format=fmt)],
    )
    st.altair_chart(chart, width="stretch")


# ------------------------------------------------------------ market data

@st.cache_data(ttl=900, show_spinner="Loading market data...")
def load_marts():
    """Return ({name: DataFrame}, source label), or (None, None) if not configured."""
    project, key = os.getenv("BQ_PROJECT"), os.getenv("GCP_SA_KEY")
    if project and key:
        from google.cloud import bigquery
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_info(json.loads(key))
        client = bigquery.Client(project=project, credentials=creds)
        names = {t: f"`{project}.{BQ_DATASET}.{t}`" for t in TABLES}
        return {k: client.query(q.format(**names)).to_dataframe(create_bqstorage_client=False)
                for k, q in QUERIES.items()}, "BigQuery"
    if LOCAL_DB.exists():
        import duckdb

        with duckdb.connect(str(LOCAL_DB), read_only=True) as con:
            return {k: con.sql(q.format(**{t: t for t in TABLES})).df()
                    for k, q in QUERIES.items()}, "local DuckDB"
    return None, None


def prepared_marts():
    """Load the marts with timestamps normalized, or show why not and return None."""
    try:
        data, source = load_marts()
    except Exception as e:  # show the kind of failure, never credentials
        st.error(f"Couldn't load market data ({type(e).__name__}). "
                 "Check the BQ_PROJECT and GCP_SA_KEY secrets.")
        return None
    if data is None:
        st.info("Not connected to market data yet. Set BQ_PROJECT and GCP_SA_KEY "
                "for BigQuery, or run Job Posting Monitor's ./pipeline/run.sh locally "
                "(or point JOB_MARKET_DB at its DuckDB file).")
        return None
    if data["postings"].empty:
        st.info("The pipeline hasn't collected any postings yet.")
        return None
    data = {k: v.copy() for k, v in data.items()}
    for col in ("first_seen_at", "last_seen_at"):
        data["postings"][col] = naive_utc(data["postings"][col])
    return data, source


# --------------------------------------------------------------- planner

# Starting assumptions. These are rough rules of thumb, not numbers the pipeline
# measures, so every one is editable in the app.
FUNNEL_DEFAULTS = [  # (stage, share of the previous stage that reaches it)
    ("Applicants", None),
    ("Recruiter screens", 0.20),
    ("Hiring-manager interviews", 0.40),
    ("Final loops", 0.40),
    ("Offers", 0.35),
    ("Accepted offers", 0.80),
]
TIMELINE_DEFAULTS = [  # (phase, days), in order, ending on the start date
    ("Intake + posting", 7),
    ("Sourcing + screening", 21),
    ("Interviews", 14),
    ("Offer + close", 7),
    ("Notice period", 14),
]
HM_INTERVIEW_HOURS = 1


def skill_shares(skills: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    """Skill x role table of the share of each role's postings mentioning the skill."""
    totals = roles.groupby("role").size()
    s = skills.assign(share=skills["postings_mentioning"] / skills["role"].map(totals))
    return s.pivot_table(index="skill", columns="role", values="share", fill_value=0.0)


def cosine(a: pd.Series, b: pd.Series) -> float:
    denom = (a.pow(2).sum() * b.pow(2).sum()) ** 0.5
    return float((a * b).sum() / denom) if denom else 0.0


def funnel_counts(hires: int, rates: list) -> list:
    """Work back from accepted offers to how many people each stage needs."""
    counts = [hires]
    for rate in reversed(rates):  # rates[i] = share moving from stage i to stage i+1
        counts.append(math.ceil(counts[-1] / rate) if rate else 0)
    return counts[::-1]


def back_plan(start: date, phases: list) -> pd.DataFrame:
    """Lay the phases end to end so the last one ends on the start date."""
    rows, end = [], start
    for name, days in reversed(phases):
        rows.append({"phase": name, "start": end - timedelta(days=days), "end": end, "days": days})
        end -= timedelta(days=days)
    return pd.DataFrame(rows[::-1])


def render_planner():
    loaded = prepared_marts()
    if loaded is None:
        return
    data, source = loaded
    postings, roles, skills = data["postings"], data["roles"], data["skills"]
    all_roles = sorted(roles["role"].unique())
    today = date.today()

    st.caption("Turn a hiring request into a plan: the candidate funnel, recruiter load and "
               "timeline it needs, checked against what the Bay Area market is doing right "
               "now, plus which adjacent roles could fill it through internal mobility.")

    # ---- the request
    c1, c2, c3, c4, c5 = st.columns(5)
    role = c1.selectbox("Role", all_roles, format_func=role_label, key="plan_role")
    hires = c2.number_input("Hires", 1, 200, 3, key="plan_hires")
    start = c3.date_input("Target start date", today + timedelta(days=90), min_value=today,
                          key="plan_start")
    role_ids = roles.loc[roles["role"] == role, "job_id"]
    p = postings[postings["job_id"].isin(role_ids)]
    pay = p["posted_salary_mid"].dropna()
    default_offer = int(round(pay.median(), -4)) if len(pay) else 150_000
    offer = c4.number_input("Planned base salary (USD)", 0, 1_000_000, default_offer,
                            step=5_000, key=f"plan_offer_{role}")
    internal_pct = c5.slider("Filled internally", 0, 100, 20, step=5, format="%d%%",
                             key="plan_internal",
                             help="Share of the hires you expect to fill through "
                                  "internal mobility rather than external hiring.")

    with st.expander("Planning assumptions (edit to match your team)"):
        st.caption("Starting values are rough rules of thumb, not measured by the pipeline.")
        left, mid, right = st.columns(3)
        left.markdown("**Funnel: share who move on to each stage**")
        rates = []
        for stage, rate in FUNNEL_DEFAULTS[1:]:
            pct = left.number_input(stage, 1, 100, int(rate * 100), format="%d",
                                    key=f"plan_rate_{stage}", help="Percent of the stage before.")
            rates.append(pct / 100)
        mid.markdown("**Timeline: days per phase**")
        phases = [(name, mid.number_input(name, 0, 180, days, key=f"plan_days_{name}"))
                  for name, days in TIMELINE_DEFAULTS]
        right.markdown("**Team capacity**")
        screens_per_week = right.number_input("Screens per recruiter per week", 1, 60, 12,
                                              key="plan_spw")
        loop_hours = right.number_input("Interviewer hours per final loop", 1, 20, 4,
                                        key="plan_loop_hours")

    internal = round(hires * internal_pct / 100)
    external = hires - internal
    stages = [s for s, _ in FUNNEL_DEFAULTS]
    counts = funnel_counts(external, rates)
    funnel = dict(zip(stages, counts))
    plan = back_plan(start, phases)
    open_by = plan["start"].iloc[0]
    total_days = int(plan["days"].sum())
    sourcing_weeks = dict(phases)["Sourcing + screening"] / 7
    recruiters = (math.ceil(funnel["Recruiter screens"] / (screens_per_week * sourcing_weeks))
                  if sourcing_weeks and external else 0)
    interviewer_hours = (funnel["Hiring-manager interviews"] * HM_INTERVIEW_HOURS
                         + funnel["Final loops"] * loop_hours)

    # ---- headline
    st.subheader("Plan at a glance")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("External hires / internal moves", f"{external} / {internal}")
    m2.metric("Applicants needed", f"{funnel['Applicants']:,}")
    m3.metric("Open the req by", f"{open_by:%b} {open_by.day}",
              delta=f"{(open_by - today).days} days from today", delta_color="off")
    m4.metric("Recruiters needed", recruiters,
              help=f"Recruiter screens ÷ ({screens_per_week} a week × "
                   f"{sourcing_weeks:.1f} weeks of sourcing).")
    m5.metric("Interviewer hours", f"{interviewer_hours:,}",
              help=f"{HM_INTERVIEW_HOURS} h per hiring-manager interview + "
                   f"{loop_hours} h per final loop.")
    if open_by < today:
        earliest = today + timedelta(days=total_days)
        st.warning(f"Behind schedule: a {start:%b} {start.day} start needed the req open "
                   f"{(today - open_by).days} days ago. Opening today, the earliest "
                   f"realistic start is **{earliest:%b} {earliest.day}, {earliest.year}**, "
                   "unless you shorten a phase.")

    # ---- market + pay
    updated = postings["last_seen_at"].max()
    recent = p[p["first_seen_at"] >= updated - timedelta(days=7)]
    left, right = st.columns(2)
    with left:
        st.subheader("Competition for this talent")
        a, b, c = st.columns(3)
        a.metric("Open postings tracked", f"{len(p):,}")
        b.metric("New in last 7 days", f"{len(recent):,}")
        companies = p["company"].dropna()
        top5 = companies.value_counts().head(5).sum() / len(companies) if len(companies) else 0
        c.metric("Employers hiring", f"{companies.nunique():,}",
                 help=f"The 5 busiest employers hold {top5:.0%} of these postings.")
        top = (companies.value_counts().head(8)
               .rename_axis("company").reset_index(name="postings"))
        ranked_bar(top, "postings", "company", f"{role_label(role)} postings")
        st.caption("The employers you're most likely competing with for candidates, and "
                   "where experienced candidates may be coming from.")
    with right:
        st.subheader("Pay positioning")
        if len(pay) < 3:
            st.caption("Too few employer-posted salaries for this role yet.")
            pct_rank = None
        else:
            q25, q50, q75 = pay.quantile([0.25, 0.5, 0.75])
            pct_rank = (pay < offer).mean()
            a, b, c = st.columns(3)
            a.metric("Market 25th pct", f"${q25:,.0f}")
            b.metric("Market median", f"${q50:,.0f}")
            c.metric("Your offer ranks", f"{pct_rank:.0%}",
                     help="Share of employer-posted salary midpoints below your offer.")
            hist = alt.Chart(pd.DataFrame({"salary": pay})).mark_bar(opacity=0.7).encode(
                x=alt.X("salary:Q", bin=alt.Bin(maxbins=12), title="Posted salary midpoint",
                        axis=alt.Axis(format="$,.0f")),
                y=alt.Y("count():Q", title="Postings"),
            )
            rule = alt.Chart(pd.DataFrame({"offer": [offer]})).mark_rule(
                color="#d62728", strokeWidth=2).encode(
                x="offer:Q", tooltip=[alt.Tooltip("offer:Q", title="Your offer", format="$,.0f")])
            st.altair_chart(hist + rule, width="stretch")
            if pct_rank < 0.25:
                st.warning("Your offer is in the bottom quarter of posted pay. Expect fewer "
                           "applicants and a lower offer-accept rate than assumed.")
            elif pct_rank < 0.5:
                st.info("Your offer is below the market median. Strong close plans "
                        "(mission, growth, flexibility) will matter.")
            else:
                st.success("Your offer is at or above the market median.")
            st.caption(f"Based on {len(pay)} employer-posted ranges (Adzuna estimates excluded). "
                       "Base salary only; equity and bonus aren't in the data.")

    # ---- funnel + timeline
    left, right = st.columns(2)
    with left:
        st.subheader("Candidate funnel for external hires")
        fdf = pd.DataFrame({"stage": stages, "people": counts})
        chart = alt.Chart(fdf).mark_bar().encode(
            x=alt.X("people:Q", title="People needed"),
            y=alt.Y("stage:N", sort=stages, title=None),
            tooltip=["stage", alt.Tooltip("people:Q", format=",")],
        ) + alt.Chart(fdf).mark_text(align="left", dx=4).encode(
            x="people:Q", y=alt.Y("stage:N", sort=stages), text=alt.Text("people:Q", format=","))
        st.altair_chart(chart, width="stretch")
        st.caption("Worked back from accepted offers using the conversion rates in "
                   "Planning assumptions.")
    with right:
        st.subheader("Back-planned timeline")
        gantt = alt.Chart(plan.assign(start=pd.to_datetime(plan["start"]),
                                      end=pd.to_datetime(plan["end"]))).mark_bar().encode(
            x=alt.X("start:T", title=None, axis=alt.Axis(format="%b %d")),
            x2="end:T",
            y=alt.Y("phase:N", sort=list(plan["phase"]), title=None),
            color=alt.Color("phase:N", legend=None, sort=list(plan["phase"])),
            tooltip=["phase", alt.Tooltip("start:T", format="%b %d"),
                     alt.Tooltip("end:T", format="%b %d"), "days"],
        )
        today_rule = alt.Chart(pd.DataFrame({"today": [pd.Timestamp(today)]})).mark_rule(
            strokeDash=[4, 4]).encode(x="today:T", tooltip=[alt.Tooltip("today:T", format="%b %d")])
        st.altair_chart(gantt + today_rule, width="stretch")
        st.caption(f"{total_days} days from intake to start. The dashed line is today.")

    # ---- internal mobility
    st.subheader("Internal mobility: who could grow into this role")
    shares = skill_shares(skills, roles)
    others = [r for r in all_roles if r != role]
    if role not in shares.columns or not others:
        st.caption("Not enough skill data for this role yet.")
        mobility = []
    else:
        target = shares[role]
        mobility = []
        for other in others:
            src = shares[other] if other in shares.columns else target * 0
            gap = (target - src).sort_values(ascending=False)
            mobility.append({
                "role": other,
                "similarity": cosine(target, src),
                "build": [s for s, g in gap.items() if g > 0][:4],
                "shared": [s for s in target.index if target[s] > 0 and src[s] > 0][:5],
            })
        mobility.sort(key=lambda m: m["similarity"], reverse=True)
        cols = st.columns(len(mobility))
        for col, m in zip(cols, mobility):
            with col:
                st.metric(f"From {role_label(m['role'])}", f"{m['similarity']:.0%} skill overlap")
                st.markdown("**Skills to build:** " + (", ".join(m["build"]) or "none"))
                st.markdown("**Already shared:** " + (", ".join(m["shared"]) or "none"))
        if internal:
            best = mobility[0]
            st.info(f"For the {internal} internal move{'s' if internal > 1 else ''}, start with "
                    f"**{role_label(best['role'])}** employees, the closest skill match. A "
                    "development plan could focus on "
                    f"{', '.join(best['build'][:3]) or 'role-specific domain knowledge'}.")
    st.caption("Overlap compares how often each skill appears in each role's job postings "
               "(cosine similarity). It describes the roles, not individual employees, and "
               "skill shares are a lower bound because Adzuna returns only a description "
               "snippet.")

    # ---- export
    lines = [
        f"# Hiring plan: {hires} × {role_label(role)}",
        f"Target start: {start:%b} {start.day}, {start.year} · planned base ${offer:,.0f}",
        "",
        f"- External hires: {external} · internal moves: {internal}",
        f"- Open the req by {open_by:%b} {open_by.day}, {open_by.year} ({total_days} days to start)",
        f"- Recruiters needed: {recruiters} · interviewer hours: {interviewer_hours:,}",
        "",
        "## Funnel",
        *[f"- {s}: {n:,}" for s, n in zip(stages, counts)],
        "",
        "## Timeline",
        *[f"- {r.phase}: {r.start:%b %d} – {r.end:%b %d} ({r.days} days)"
          for r in plan.itertuples()],
        "",
        "## Market",
        f"- {len(p):,} postings tracked, {len(recent):,} new in the last 7 days, "
        f"{p['company'].nunique():,} employers",
    ]
    if pct_rank is not None:
        lines.append(f"- Offer is above {pct_rank:.0%} of {len(pay)} employer-posted salaries")
    for m in mobility:
        lines.append(f"- Internal source {role_label(m['role'])}: {m['similarity']:.0%} skill "
                     f"overlap; build {', '.join(m['build']) or 'n/a'}")
    lines += ["", f"_Market data: {source}, Adzuna postings, San Francisco + 80 km._"]
    st.download_button("Download plan (Markdown)", "\n".join(lines),
                       f"hiring_plan_{role}.md", "text/markdown", key="plan_md")


# ------------------------------------------------------------------- page

st.set_page_config(page_title="Hiring Planner", layout="wide")
st.title("Hiring Planner")
render_planner()
