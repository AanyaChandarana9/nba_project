"""
Revised analysis for "Pathway vs. Performance" (prep-to-pro vs one-and-done).
Reads draft_table.csv and player_seasons.csv (from build_dataset_bbref.py) and
writes every number, table and figure used in the revised manuscript.

Usage:  python3 run_analysis.py <draft_table.csv> <player_seasons.csv> <output_dir>
"""
import sys, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

CSV, SEASONS_CSV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20261107)
B = 10_000
DATA_THROUGH = 2026          # last completed season = 2025-26 (season ending year)
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#6b6b6b"
LABEL = {"hs": "Prep-to-pro", "fr": "One-and-done"}
R = {}

# ---------------------------------------------------------------- cleaning
raw = pd.read_csv(CSV)
R["raw_rows"] = int(len(raw))
d = raw.sort_values("SEASON").drop_duplicates("id", keep="last").copy()
R["duplicate_rows_removed"] = int(len(raw) - len(d))
# Not one-and-done under the paper's definition (one season of U.S. college
# basketball before the draft used): Mark Eaton (junior college, then UCLA);
# Phil Walker (re-drafted a year after his first draft, i.e. after more college);
# Jason Richardson (two seasons at Michigan State, 1999-2001).
NOT_OAD = ["Mark Eaton", "Phil Walker", "Jason Richardson"]
R["reclassified_out"] = NOT_OAD
d = d[~((d["class"] == "fr") & d.full_name.isin(NOT_OAD))].copy()

d["first_year"] = d.FIRST_SEASON_ID.str[:4].astype(int) + 1   # 1995-96 -> 1996
d["last_year"] = d.LAST_SEASON_ID.str[:4].astype(int) + 1
d["seasons"] = d.last_year - d.first_year + 1                  # seasons spanned
d["event"] = (~d.is_active).astype(int)                        # 1 = career over
d["MPG"] = d.MIN / d.GP
d["PPG"] = d.PTS / d.GP
d["GS_share"] = d.GS / d.GP
d["allstar_any"] = (d.all_star_count > 0).astype(int)
d["title_any"] = (d.championship_count > 0).astype(int)
d["era"] = np.where(d.SEASON <= 2005, "pre", "rule")
d["lottery"] = (d.OVERALL_PICK <= 14).astype(int)

hs, fr = d[d["class"] == "hs"], d[d["class"] == "fr"]
R["n"] = {"hs": int(len(hs)), "fr": int(len(fr)),
          "hs_retired": int((~hs.is_active).sum()), "fr_retired": int((~fr.is_active).sum()),
          "fr_pre2006": int((fr.SEASON <= 2005).sum()), "fr_rule": int((fr.SEASON >= 2006).sum()),
          "fr_active": int(fr.is_active.sum()), "hs_active": int(hs.is_active.sum())}
R["draft_years"] = {"hs": [int(hs.SEASON.min()), int(hs.SEASON.max())],
                    "fr": [int(fr.SEASON.min()), int(fr.SEASON.max())]}

# ---------------------------------------------------------------- helpers
def boot_diff(a, b, stat=np.mean):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ia = RNG.integers(0, len(a), (B, len(a)))
    ib = RNG.integers(0, len(b), (B, len(b)))
    diffs = stat(a[ia], axis=1) - stat(b[ib], axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"a": float(stat(a)), "b": float(stat(b)), "diff": float(stat(a) - stat(b)),
            "lo": float(lo), "hi": float(hi), "sig": bool(lo > 0 or hi < 0),
            "na": int(len(a)), "nb": int(len(b))}

def boot_sd_ratio(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ia = RNG.integers(0, len(a), (B, len(a)))
    ib = RNG.integers(0, len(b), (B, len(b)))
    r = a[ia].std(axis=1, ddof=1) / b[ib].std(axis=1, ddof=1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return {"sd_a": float(a.std(ddof=1)), "sd_b": float(b.std(ddof=1)),
            "ratio": float(a.std(ddof=1) / b.std(ddof=1)), "lo": float(lo), "hi": float(hi),
            "sig": bool(lo > 1 or hi < 1)}

def table(rows, A, Bdf):
    out = {}
    for key, col in rows:
        out[key] = boot_diff(A[col], Bdf[col])
    return out

# --------------------------------------------- A. full-sample description
R["A_full"] = {
    "draft_pick": boot_diff(hs.OVERALL_PICK, fr.OVERALL_PICK),
    "first_age": boot_diff(hs.FIRST_SEASON_PLAYER_AGE, fr.FIRST_SEASON_PLAYER_AGE),
    "mpg": boot_diff(hs.MPG, fr.MPG),
    "ppg": boot_diff(hs.PPG, fr.PPG),
    "gs_share": boot_diff(hs.GS_share, fr.GS_share),
}
# retired-only cumulative measures (kept for continuity with v1, flagged as biased)
hr, frr = hs[~hs.is_active], fr[~fr.is_active]
R["A_retired"] = {
    "last_age": boot_diff(hr.LAST_SEASON_PLAYER_AGE, frr.LAST_SEASON_PLAYER_AGE),
    "seasons": boot_diff(hr.seasons, frr.seasons),
    "pts": boot_diff(hr.PTS, frr.PTS),
    "allstar": boot_diff(hr.all_star_count, frr.all_star_count),
    "titles": boot_diff(hr.championship_count, frr.championship_count),
}
R["retired_fr_draft_years"] = [int(frr.SEASON.min()), int(frr.SEASON.max())]
R["retired_fr_median_draft_year"] = float(frr.SEASON.median())
R["active_fr_allstars_examples"] = (fr[fr.is_active].sort_values("all_star_count", ascending=False)
    .head(6)[["full_name", "all_star_count"]].values.tolist())
R["active_fr_share_allstar"] = float(fr[fr.is_active].allstar_any.mean())
R["retired_fr_share_allstar"] = float(frr.allstar_any.mean())

# ------------------------------- B. same-era comparison (all careers complete)
fr_pre = fr[(fr.SEASON >= 1993) & (fr.SEASON <= 2005)]
hs_pre = hs[(~hs.is_active) & (hs.SEASON >= 1993) & (hs.SEASON <= 2005)]   # retired, same draft window
R["B_same_era"] = {
    "n_hs": int(len(hs_pre)), "n_fr": int(len(fr_pre)),
    "fr_years": [int(fr_pre.SEASON.min()), int(fr_pre.SEASON.max())],
    "fr_all_retired": bool((~fr_pre.is_active).all()),
    "names_fr": sorted(fr_pre.full_name.tolist()),
    "draft_pick": boot_diff(hs_pre.OVERALL_PICK, fr_pre.OVERALL_PICK),
    "first_age": boot_diff(hs_pre.FIRST_SEASON_PLAYER_AGE, fr_pre.FIRST_SEASON_PLAYER_AGE),
    "seasons": boot_diff(hs_pre.seasons, fr_pre.seasons),
    "last_age": boot_diff(hs_pre.LAST_SEASON_PLAYER_AGE, fr_pre.LAST_SEASON_PLAYER_AGE),
    "gp": boot_diff(hs_pre.GP, fr_pre.GP),
    "mpg": boot_diff(hs_pre.MPG, fr_pre.MPG),
    "ppg": boot_diff(hs_pre.PPG, fr_pre.PPG),
    "pts": boot_diff(hs_pre.PTS, fr_pre.PTS),
    "pts_median": boot_diff(hs_pre.PTS, fr_pre.PTS, stat=np.median),
    "allstar": boot_diff(hs_pre.all_star_count, fr_pre.all_star_count),
    "allstar_any": boot_diff(hs_pre.allstar_any, fr_pre.allstar_any),
    "titles": boot_diff(hs_pre.championship_count, fr_pre.championship_count),
    "title_any": boot_diff(hs_pre.title_any, fr_pre.title_any),
    "sd_pts": boot_sd_ratio(hs_pre.PTS, fr_pre.PTS),
    "sd_allstar": boot_sd_ratio(hs_pre.all_star_count, fr_pre.all_star_count),
}
# leave-top-out sensitivity: drop the three players with most All-Star selections in hs
top3 = hs_pre.sort_values("all_star_count", ascending=False).head(3)
R["B_top3_names"] = top3.full_name.tolist()
hs_trim = hs_pre.drop(top3.index)
R["B_trim"] = {"pts": boot_diff(hs_trim.PTS, fr_pre.PTS),
               "allstar": boot_diff(hs_trim.all_star_count, fr_pre.all_star_count),
               "titles": boot_diff(hs_trim.championship_count, fr_pre.championship_count),
               "seasons": boot_diff(hs_trim.seasons, fr_pre.seasons)}

# --------------------- C. censoring-aware career length (all 139, actives kept)
km = {}
for g, sub in [("hs", hs), ("fr", fr)]:
    k = KaplanMeierFitter().fit(sub.seasons, sub.event, label=LABEL[g])
    km[g] = k
    surv = k.survival_function_at_times([5, 10, 15]).values
    R.setdefault("C_km", {})[g] = {"median": float(k.median_survival_time_),
                                   "S5": float(surv[0]), "S10": float(surv[1]), "S15": float(surv[2])}
lr = logrank_test(hs.seasons, fr.seasons, hs.event, fr.event)
R["C_logrank_p"] = float(lr.p_value)
# KM restricted to the rule era only vs prep-to-pro
fr_rule = fr[fr.SEASON >= 2006]
k_rule = KaplanMeierFitter().fit(fr_rule.seasons, fr_rule.event)
R["C_km_rule_era"] = {"n": int(len(fr_rule)), "median": float(k_rule.median_survival_time_),
                      "S10": float(k_rule.survival_function_at_times([10]).values[0])}
lr2 = logrank_test(hs.seasons, fr_rule.seasons, hs.event, fr_rule.event)
R["C_logrank_rule_p"] = float(lr2.p_value)

# --------- D. five-season window after the draft (reviewer's suggested design)
# Seasons 1-5 after the draft = season ending years D+1 ... D+5, so every player
# drafted in 2021 or earlier has a full window by 2025-26. Active players included.
ps = pd.read_csv(SEASONS_CSV)
win = []
for _, p in d[d.SEASON <= DATA_THROUGH - 5].iterrows():
    w = ps[(ps.player_id == p.id) & (ps.season > p.SEASON) & (ps.season <= p.SEASON + 5)]
    gp, mp, pts = w.g.sum(), w.mp.sum(), w.pts.sum()
    win.append({"id": p.id, "class": p["class"], "SEASON": p.SEASON, "lottery": p.lottery,
                "w_gp": gp, "w_mpg": mp / gp if gp else 0.0, "w_ppg": pts / gp if gp else 0.0,
                "w_pts": pts, "w_seasons": int((w.g > 0).sum()),
                "w_in_year5": int(((w.season == p.SEASON + 5) & (w.g > 0)).any()),
                "w_allstar": int(w.all_star.sum()), "w_allstar_any": int(w.all_star.any()),
                "w_title_any": int(w.champion.any())})
win = pd.DataFrame(win)
def win_block(a, b):
    return {"n_hs": int(len(a)), "n_fr": int(len(b)),
            "gp": boot_diff(a.w_gp, b.w_gp), "mpg": boot_diff(a.w_mpg, b.w_mpg),
            "ppg": boot_diff(a.w_ppg, b.w_ppg), "pts": boot_diff(a.w_pts, b.w_pts),
            "in_year5": boot_diff(a.w_in_year5, b.w_in_year5),
            "allstar_any": boot_diff(a.w_allstar_any, b.w_allstar_any),
            "title_any": boot_diff(a.w_title_any, b.w_title_any)}
wh, wf = win[win["class"] == "hs"], win[win["class"] == "fr"]
R["D_window_all"] = win_block(wh, wf)
R["D_window_all"]["fr_years"] = [int(wf.SEASON.min()), int(wf.SEASON.max())]
R["D_window_all"]["fr_active"] = int(d[d.id.isin(wf.id)].is_active.sum())
wfe = wf[wf.SEASON <= 2005]
R["D_window_era"] = win_block(wh, wfe)
R["D_window_lottery"] = win_block(wh[wh.lottery == 1], wf[wf.lottery == 1])

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: draftees per draft year, both pathways
yrs = np.arange(1993, 2026)
c_h = hs.SEASON.value_counts().reindex(yrs, fill_value=0)
c_f = fr.SEASON.value_counts().reindex(yrs, fill_value=0)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
w = 0.42
ax.bar(yrs - w/2 - 0.01, c_h, w, color=BLUE, label=f"Prep-to-pro (n = {len(hs)})")
ax.bar(yrs + w/2 + 0.01, c_f, w, color=ORANGE, hatch="////", edgecolor="white", linewidth=0,
       label=f"One-and-done (n = {len(fr)})")
ax.axvline(2005.5, color=MUTED, ls="--", lw=1)
ax.text(2005.7, ax.get_ylim()[1]*0.95, "Age rule applies\nfrom 2006 draft", fontsize=8.5, color=MUTED, va="top")
ax.set_xlabel("Draft year"); ax.set_ylabel("Players in sample")
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.set_xticks(np.arange(1995, 2026, 5)); ax.yaxis.grid(True, color="#e5e5e5"); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_draft_years.png", dpi=300); plt.close(fig)

# Fig 2: Kaplan-Meier career length
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for g, col, ls in [("hs", BLUE, "-"), ("fr", ORANGE, "--")]:
    sf = km[g].survival_function_
    ax.step(sf.index, sf.iloc[:, 0], where="post", color=col, lw=2, ls=ls,
            label=f"{LABEL[g]} (n = {len(hs) if g=='hs' else len(fr)})")
    cens = (d[(d['class']==g) & (d.event==0)].seasons)
    ax.plot(cens, km[g].survival_function_at_times(cens).values, "|", color=col, ms=8, mew=1.5)
ax.set_xlabel("NBA seasons spanned"); ax.set_ylabel("Share still in the league")
ax.set_ylim(0, 1.02); ax.set_xlim(0, 24); ax.yaxis.grid(True, color="#e5e5e5"); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower left")
ax.text(24, 0.98, "Tick marks = active players (career still in progress)", fontsize=8, color=MUTED, ha="right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_career_length_km.png", dpi=300); plt.close(fig)

# Fig 3: same-era career points, every player shown
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for i, (sub, col, mk, name) in enumerate([(hs_pre, BLUE, "o", "Prep-to-pro"),
                                         (fr_pre, ORANGE, "s", "One-and-done")]):
    y = np.full(len(sub), i) + RNG.uniform(-0.12, 0.12, len(sub))
    ax.scatter(sub.PTS, y, s=46, color=col, marker=mk, edgecolor="white", linewidth=1, zorder=3)
    med = sub.PTS.median()
    ax.plot([med, med], [i-0.28, i+0.28], color=INK, lw=2, zorder=4)
for _, r in hs_pre.nlargest(2, "PTS").iterrows():
    ax.annotate(r.full_name, (r.PTS, 0), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8, color=INK)
for _, r in fr_pre.nlargest(1, "PTS").iterrows():
    ax.annotate(r.full_name, (r.PTS, 1), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8, color=INK)
ax.set_yticks([0, 1]); ax.set_yticklabels([f"Prep-to-pro\n(n = {len(hs_pre)})", f"One-and-done\n(n = {len(fr_pre)})"])
ax.set_ylim(-0.6, 1.6); ax.set_xlabel("Career regular-season points"); ax.xaxis.grid(True, color="#e5e5e5"); ax.set_axisbelow(True)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{int(v):,}"))
ax.text(ax.get_xlim()[1], -0.55, "Black line = group median", fontsize=8, color=MUTED, ha="right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_same_era_points.png", dpi=300); plt.close(fig)

# Fig 4: milestones in the five-season window, with 95% bootstrap CIs per group
def prop_ci(x):
    x = np.asarray(x, float); idx = RNG.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(axis=1); return x.mean(), *np.percentile(m, [2.5, 97.5])
mil = [("w_in_year5", "Still in the NBA in season 5"), ("w_allstar_any", "1+ All-Star selection"),
       ("w_title_any", "1+ championship")]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for j, (col, name) in enumerate(mil):
    for k, (sub, c, mk, off) in enumerate([(wh, BLUE, "o", -0.13), (wf, ORANGE, "s", 0.13)]):
        m, lo, hi = prop_ci(sub[col])
        ax.plot([lo, hi], [j+off, j+off], color=c, lw=2)
        ax.plot(m, j+off, mk, color=c, ms=8, mec="white", mew=1)
        R.setdefault("fig4", {})[f"{col}_{'hs' if k==0 else 'fr'}"] = [float(m), float(lo), float(hi)]
ax.set_yticks(range(3)); ax.set_yticklabels([n for _, n in mil]); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("Share of players in their first five seasons (95% bootstrap interval)")
ax.xaxis.grid(True, color="#e5e5e5"); ax.set_axisbelow(True)
h1, = ax.plot([], [], "o", color=BLUE, label=f"Prep-to-pro (n = {len(wh)})")
h2, = ax.plot([], [], "s", color=ORANGE, label=f"One-and-done (n = {len(wf)})")
ax.legend(handles=[h1, h2], frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_five_season_window.png", dpi=300); plt.close(fig)

with open(f"{OUT}/results.json", "w") as f:
    json.dump(R, f, indent=1)
print(json.dumps(R, indent=1))
