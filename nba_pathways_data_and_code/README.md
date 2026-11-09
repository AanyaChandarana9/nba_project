# NBA pathways study: data and code

Everything needed to reproduce the numbers in "Pathway vs. Performance." Nothing needs to be run before submitting. These files are here so the GitHub repository matches what the paper describes.

| File | What it is |
| --- | --- |
| draft_table.csv | Final dataset, one row per player (37 prep-to-pro, 277 one-and-done) |
| player_seasons.csv | One row per player-season, used for the five-season window |
| hs_draftees.csv | High school draftees, 1995-2005 (source: Wikipedia, "NBA high school draftees") |
| freshman_tags.csv | Freshman draftees, 1993-2025, with the source page for each year |
| build_dataset_bbref.py | Builds the two data files from Basketball-Reference tables |
| run_analysis.py | Produces every number and figure in the paper (results.json) |
| make_tables.py | Formats Tables 1-3 (tables.md) |

To reproduce:

```
git clone https://github.com/sumitrodatta/bball-reference-datasets
python3 build_dataset_bbref.py bball-reference-datasets/Data out
python3 run_analysis.py out/draft_table.csv out/player_seasons.csv results
python3 make_tables.py results/results.json > results/tables.md
```

Requires Python 3 with pandas, numpy, matplotlib and lifelines.
