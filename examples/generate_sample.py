"""Create the small public synthetic dataset used by the quick-start example."""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(61)
rows = []
for index in range(1200):
    months = random.randint(0, 120)
    debt = round(random.uniform(0, 900_000), 2)
    income = "unknown" if index % 103 == 0 else random.choices(["low", "middle", "high"], weights=[0.35, 0.4, 0.25])[0]
    if index % 97 == 0:
        months = -1
    if index % 89 == 0:
        debt = -999
    missing_debt = index % 107 == 0
    risk = 0.16 + (0.18 if months < 12 else 0) + (0.13 if debt > 450_000 else 0) + {"low": 0.16, "middle": 0.06, "high": -0.08, "unknown": 0.12}[income]
    bad = int(random.random() < max(0.02, min(0.8, risk)))
    day = date(2022, 1, 1) + timedelta(days=index * 365 // 1200)
    # Spread records across the configured two-year development and one-year OOT windows.
    if index >= 800:
        day = date(2024, 1, 1) + timedelta(days=(index - 800) * 365 // 400)
    rows.append({"application_date": day.isoformat(), "defaulted": bad,
        "months_on_book": months, "monthly_debt": None if missing_debt else debt, "income_band": income})

output = Path(__file__).with_name("sample-applications.csv")
with output.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
