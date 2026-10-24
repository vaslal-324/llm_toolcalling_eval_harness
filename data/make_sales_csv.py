"""Regenerates data/sales.csv deterministically (seeded). Not needed for normal runs."""
import csv
import random
from pathlib import Path

random.seed(42)
PRICES = {"Widget": 25.0, "Gadget": 40.0, "Gizmo": 15.5}
out = Path(__file__).with_name("sales.csv")
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["month", "region", "product", "units", "revenue"])
    for m in range(1, 7):
        for region in ["North", "South", "East", "West"]:
            for product, price in PRICES.items():
                units = random.randint(80, 400)
                w.writerow([f"2026-{m:02d}", region, product, units, round(units * price, 2)])
print("wrote", out)
