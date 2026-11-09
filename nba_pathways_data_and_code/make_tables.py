import json, sys
R = json.load(open(sys.argv[1]))
from decimal import Decimal, ROUND_HALF_UP
def fmt(x, d):
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    s = f"{q:,.{d}f}"
    if s.lstrip("-").strip("0.,") == "": s = s.lstrip("-")
    return s.replace("-", "−")
def row(label, who, r, d, pct=False):
    k = 100 if pct else 1
    a, b, df, lo, hi = (r[x]*k for x in ("a","b","diff","lo","hi"))
    unit = "%" if pct else ""
    star = " *" if r["sig"] else ""
    ci = f"{fmt(lo,d)} to {fmt(hi,d)}"
    cells = [label] + ([who] if who is not None else []) + [fmt(a,d)+unit, fmt(b,d)+unit, fmt(df,d)+((" pt" if abs(round(df)) == 1 else " pts") if pct else "")+star, ci]
    return "| " + " | ".join(cells) + " |"
A, Ar = R["A_full"], R["A_retired"]
n = R["n"]
t1 = ["| Measure | Players compared | Prep-to-pro | One-and-done | Difference | 95% CI |",
      "| --- | --- | --- | --- | --- | --- |",
      row("Draft pick (mean)", f"All ({n['hs']} vs. {n['fr']})", A["draft_pick"], 1),
      row("Age in first NBA season", f"All ({n['hs']} vs. {n['fr']})", A["first_age"], 1),
      row("Minutes per game", f"All ({n['hs']} vs. {n['fr']})", A["mpg"], 1),
      row("Points per game", f"All ({n['hs']} vs. {n['fr']})", A["ppg"], 1),
      row("Share of games started", f"All ({n['hs']} vs. {n['fr']})", A["gs_share"], 0, pct=True),
      row("Age in last NBA season", f"Retired ({n['hs_retired']} vs. {n['fr_retired']})", Ar["last_age"], 1),
      row("Seasons spanned", f"Retired ({n['hs_retired']} vs. {n['fr_retired']})", Ar["seasons"], 1),
      row("Career points", f"Retired ({n['hs_retired']} vs. {n['fr_retired']})", Ar["pts"], 0),
      row("All-Star selections", f"Retired ({n['hs_retired']} vs. {n['fr_retired']})", Ar["allstar"], 2),
      row("Championships", f"Retired ({n['hs_retired']} vs. {n['fr_retired']})", Ar["titles"], 2)]
B = R["B_same_era"]
t2 = ["| Measure | Prep-to-pro (n = %d) | One-and-done (n = %d) | Difference | 95%% CI |" % (B["n_hs"], B["n_fr"]),
      "| --- | --- | --- | --- | --- |",
      row("Draft pick (mean)", None, B["draft_pick"], 1),
      row("Age in first NBA season", None, B["first_age"], 1),
      row("Seasons spanned", None, B["seasons"], 1),
      row("Age in last NBA season", None, B["last_age"], 1),
      row("Games played", None, B["gp"], 0),
      row("Minutes per game", None, B["mpg"], 1),
      row("Points per game", None, B["ppg"], 1),
      row("Career points (mean)", None, B["pts"], 0),
      row("Career points (median)", None, B["pts_median"], 0),
      row("All-Star selections (mean)", None, B["allstar"], 2),
      row("Made 1+ All-Star team", None, B["allstar_any"], 0, pct=True),
      row("Championships (mean)", None, B["titles"], 2),
      row("Won 1+ championship", None, B["title_any"], 0, pct=True)]
W, E, L = R["D_window_all"], R["D_window_era"], R["D_window_lottery"]
def wrows(lbl, X):
    return [row("Games played", lbl, X["gp"], 0),
            row("Minutes per game", lbl, X["mpg"], 1),
            row("Points per game", lbl, X["ppg"], 1),
            row("Points scored", lbl, X["pts"], 0),
            row("Played in season 5", lbl, X["in_year5"], 0, pct=True),
            row("Made 1+ All-Star team", lbl, X["allstar_any"], 0, pct=True),
            row("Won 1+ championship", lbl, X["title_any"], 0, pct=True)]
t3 = ["| Measure (first five seasons) | Players compared | Prep-to-pro | One-and-done | Difference | 95% CI |",
      "| --- | --- | --- | --- | --- | --- |"]
t3 += wrows(f"Drafted 1993-2021 ({W['n_hs']} vs. {W['n_fr']})".replace("-", "\u2013"), W)
t3 += wrows(f"Same era, 1993-2005 ({E['n_hs']} vs. {E['n_fr']})".replace("-", "\u2013"), E)
t3 += [row("Played in season 5", f"Lottery picks ({L['n_hs']} vs. {L['n_fr']})", L["in_year5"], 0, pct=True),
       row("Made 1+ All-Star team", f"Lottery picks ({L['n_hs']} vs. {L['n_fr']})", L["allstar_any"], 0, pct=True),
       row("Points per game", f"Lottery picks ({L['n_hs']} vs. {L['n_fr']})", L["ppg"], 1)]
for name, t in [("T1", t1), ("T2", t2), ("T3", t3)]:
    print(name); print("\n".join(t)); print()
