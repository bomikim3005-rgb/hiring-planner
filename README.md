# Hiring Planner

A Streamlit app for talent acquisition and internal mobility teams. It turns a hiring request into a plan and checks it against live San Francisco Bay Area market data for data analyst, data scientist and analytics engineer roles.

## What it does

Enter a role, the number of hires, a target start date, the planned base salary and the share to fill internally. The app returns:

- **Candidate funnel.** Applicants, screens, interviews, final loops and offers needed, worked back from accepted offers.
- **Team load.** Recruiters needed for the sourcing window, and total interviewer hours.
- **Back-planned timeline.** When the req must open to hit the start date. If that date has passed, it warns and shows the earliest realistic start.
- **Market check.** Competing employers and posting volume for the role, and where the offer ranks among employer-posted salaries.
- **Internal mobility.** Skill overlap between the target role and the other roles (cosine similarity of skill-mention shares in job postings), the skills to build, and which role to source internal moves from.
- **Export.** The plan as a Markdown file.

## Data

Market data comes from the dbt models built by **Job Posting Monitor**, a separate project that collects Adzuna postings every 6 hours into BigQuery. The planner reads `dim_postings`, `bridge_posting_roles` and `mart_skill_mentions`.

- **BigQuery:** set `BQ_PROJECT` and `GCP_SA_KEY` (a read-only service account's JSON key).
- **Locally:** it reads `../job-posting-monitor/data/job_market.duckdb` by default, or the file named in `JOB_MARKET_DB`.

## Limitations

- **Planning assumptions are editable defaults, not measurements.** Funnel conversion rates, phase lengths and team capacity start from rough rules of thumb. Replace them with your team's own numbers.
- **Salary benchmarks use employer-posted base pay only.** Adzuna's estimates are excluded, samples can be small, and equity and bonus aren't included.
- **Skill overlap describes roles, not people.** It compares how often skills appear in each role's postings. Adzuna returns only a description snippet, so skill shares are a lower bound.

## Run it locally

Requires Python 3.10+ (the local DuckDB file needs a recent DuckDB, which doesn't support macOS's built-in Python 3.9).

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Create a Streamlit Community Cloud app from this repository with main file `app.py`, and add `BQ_PROJECT` and `GCP_SA_KEY` as secrets (`GCP_SA_KEY` wrapped in `'''` triple quotes).
