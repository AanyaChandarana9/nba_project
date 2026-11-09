"""
Build the prep-to-pro / one-and-done dataset from public sources, no API calls.

Inputs (all in this folder or the bball-reference-datasets repo):
  hs_draftees.csv     players drafted directly from U.S. high schools, 1995-2005
                      (Wikipedia, "NBA high school draftees")
  freshman_tags.csv   players drafted after one season of U.S. college basketball,
                      1993-2025 (Wikipedia annual draft pages, "(Fr.)" tags;
                      RealGM draft recaps for 2024 and 2025)
  Basketball-Reference datasets (github.com/sumitrodatta/bball-reference-datasets):
      Draft Pick History.csv, Player Totals.csv, All-Star Selections.csv

Usage: python3 build_dataset_bbref.py <bbref Data folder> <out folder>
Outputs: draft_table.csv (one row per player), player_seasons.csv (one row per
player-season, used for the five-season window), build_log.txt
"""
import sys, os, re, unicodedata
import pandas as pd

BB, OUT = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
THROUGH = 2026                    # last completed season = 2025-26 (season ending year)
FIRST_DRAFT, LAST_DRAFT = 1993, 2025
log = []

# NBA champions by season ending year (Wikipedia, "List of NBA champions")
CHAMPS = {1994: "HOU", 1995: "HOU", 1996: "CHI", 1997: "CHI", 1998: "CHI", 1999: "SAS",
          2000: "LAL", 2001: "LAL", 2002: "LAL", 2003: "SAS", 2004: "DET", 2005: "SAS",
          2006: "MIA", 2007: "SAS", 2008: "BOS", 2009: "LAL", 2010: "LAL", 2011: "DAL",
          2012: "MIA", 2013: "MIA", 2014: "SAS", 2015: "GSW", 2016: "CLE", 2017: "GSW",
          2018: "GSW", 2019: "TOR", 2020: "LAL", 2021: "MIL", 2022: "GSW", 2023: "DEN",
          2024: "BOS", 2025: "OKC", 2026: "NYK"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", re.sub(r"[^a-z ]", "", s))
    return re.sub(r"\s+", " ", s).strip()


draft = pd.read_csv(os.path.join(BB, "Draft Pick History.csv"))
draft = draft[draft.lg == "NBA"].copy()
tot = pd.read_csv(os.path.join(BB, "Player Totals.csv"))
tot = tot[(tot.lg == "NBA")].copy()
allstar = pd.read_csv(os.path.join(BB, "All-Star Selections.csv"))
allstar = allstar[allstar.lg == "NBA"]

# ------------------------------------------------ group membership via (year, pick)
groups = []
ALIASES = {"bill walker": "henry walker", "louis williams": "lou williams",
           "cameron christie": "cam christie"}
draft["nname"] = draft.player.map(norm)
draft["last"] = draft.nname.str.split().str[-1]
for fname, cls in [("hs_draftees.csv", "hs"), ("freshman_tags.csv", "fr")]:
    src = pd.read_csv(os.path.join(HERE, fname))
    for _, r in src.iterrows():
        yr = draft[draft.season == r.season]
        n = ALIASES.get(norm(r.player), norm(r.player))
        hit = yr[yr.nname == n]                                   # 1. same year, same name
        how = "name"
        if len(hit) != 1:
            hit = yr[yr["last"] == n.split()[-1]]                  # 2. same year, same surname
            how = "surname"
        if len(hit) != 1:
            hit = yr[yr.overall_pick == r.pick]                    # 3. same year, same pick
            how = "pick only"
        if len(hit) != 1:
            log.append(f"NO DRAFT MATCH  {cls} {r.season} pick {r.pick} {r.player}")
            continue
        h = hit.iloc[0]
        if how != "name" or int(h.overall_pick) != r.pick:
            log.append(f"matched by {how}: {cls} {r.season} source pick {r.pick} '{r.player}' -> BBRef pick {int(h.overall_pick)} '{h.player}'")
        groups.append({"id": h.player_id, "full_name": h.player, "class": cls,
                       "SEASON": int(r.season), "OVERALL_PICK": int(h.overall_pick),
                       "ORGANIZATION": r.school})
g = pd.DataFrame(groups)
g = g[(g.SEASON >= FIRST_DRAFT) & (g.SEASON <= LAST_DRAFT)]

# a player drafted twice keeps his latest draft; drop him if that draft is not the tagged one
latest = draft.groupby("player_id").season.max()
redrafted = g[g.SEASON < g.id.map(latest)]
for _, r in redrafted.iterrows():
    log.append(f"re-drafted later, removed  {r['class']} {r.SEASON} {r.full_name}")
g = g[g.SEASON >= g.id.map(latest)]
dup = g[g.id.duplicated(keep=False)]
if len(dup):
    log.append(f"in both groups (check): {dup.full_name.tolist()}")

# ------------------------------------------------ season rows, one per player-season
t = tot[tot.player_id.isin(g.id) & (tot.season <= THROUGH)].copy()
t["order"] = t.groupby(["player_id", "season"]).cumcount()
multi = t.team.str.match(r"^\dTM$|^TOT$")
totals = t[multi]
singles = t[~multi]
# season totals: the "2TM/3TM" row if traded, else the only row
one = pd.concat([totals, singles[~singles.set_index(["player_id", "season"]).index.isin(
    totals.set_index(["player_id", "season"]).index)]])
# last team of the season (team rows follow the total row in chronological order)
last_team = singles.sort_values("order").groupby(["player_id", "season"]).team.last()
one = one.set_index(["player_id", "season"])
one["final_team"] = last_team
one = one.reset_index()
one["champion"] = [CHAMPS.get(s) == tm for s, tm in zip(one.season, one.final_team)]
one["all_star"] = one.set_index(["player_id", "season"]).index.isin(
    allstar.set_index(["player_id", "season"]).index)
seasons = one[["player_id", "season", "age", "final_team", "g", "gs", "mp", "pts",
               "champion", "all_star"]].sort_values(["player_id", "season"])
seasons.to_csv(os.path.join(OUT, "player_seasons.csv"), index=False)

# active = appeared in 2025-26 or in the current 2026-27 season
active_ids = set(tot[tot.season >= THROUGH].player_id)

rows = []
for _, p in g.iterrows():
    s = seasons[seasons.player_id == p.id]
    if s.empty or s.g.sum() == 0:
        log.append(f"no NBA games through 2025-26, excluded  {p['class']} {p.SEASON} {p.full_name}")
        continue
    rows.append({**p.to_dict(),
                 "is_active": p.id in active_ids,
                 "all_star_count": int(s.all_star.sum()),
                 "championship_count": int(s.champion.sum()),
                 "FIRST_SEASON_ID": f"{s.season.min() - 1}-{str(s.season.min())[2:]}",
                 "LAST_SEASON_ID": f"{s.season.max() - 1}-{str(s.season.max())[2:]}",
                 "FIRST_SEASON_PLAYER_AGE": s.age.iloc[0],
                 "LAST_SEASON_PLAYER_AGE": s.age.iloc[-1],
                 "seasons_played": len(s),
                 "GP": s.g.sum(), "GS": s.gs.sum(), "MIN": s.mp.sum(), "PTS": s.pts.sum()})
out = pd.DataFrame(rows).sort_values(["class", "SEASON", "OVERALL_PICK"])
out.to_csv(os.path.join(OUT, "draft_table.csv"), index=False)
log.append(f"FINAL: {dict(out['class'].value_counts())}; active {dict(out.groupby('class').is_active.sum())}")
open(os.path.join(OUT, "build_log.txt"), "w").write("\n".join(log) + "\n")
print("\n".join(log))
