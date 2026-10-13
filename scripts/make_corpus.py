"""Generate the synthetic Northwind Gear corpus (D3).

Run: uv run python scripts/make_corpus.py
Output is deterministic and committed under data/.
"""

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

PRODUCTS = [
    ("NW-1042", "Ridgeline 2P Tent", "tents", 329.00),
    ("NW-1043", "Ridgeline 3P Tent", "tents", 389.00),
    ("NW-1107", "Alpine Down Sleeping Bag 15F", "sleep", 279.00),
    ("NW-1108", "Alpine Synthetic Sleeping Bag 30F", "sleep", 149.00),
    ("NW-2010", "Summit 45L Backpack", "packs", 219.00),
    ("NW-2015", "Summit 65L Backpack", "packs", 289.00),
    ("NW-3301", "Trailhead Headlamp 400", "lighting", 49.00),
    ("NW-3302", "Trailhead Headlamp 800", "lighting", 79.00),
    ("NW-4020", "Basecamp Stove", "kitchen", 119.00),
    ("NW-4025", "Basecamp Cook Set", "kitchen", 69.00),
    ("NW-5100", "Stormshell Jacket", "apparel", 249.00),
    ("NW-5200", "Cirrus Trekking Poles", "accessories", 129.00),
]

# plan -> (price multiplier, minimum order quantity, lead time in days)
PLANS = {
    "Basic": (0.85, 1, 7),
    "Pro": (0.75, 10, 5),
    "Enterprise": (0.65, 50, 3),
}

HANDBOOK_FACTS = {
    "v1": {
        "free_shipping": "$50",
        "digital_refund_days": "30 days",
        "warranty": "1 year",
        "trade_in": "",
    },
    "v2": {
        "free_shipping": "$75",
        "digital_refund_days": "14 days",
        "warranty": "2 years",
        "trade_in": """
# Trade-in program

## Eligible items

Customers can trade in used Northwind Gear tents and backpacks in working condition. Accepted
trade-ins earn store credit worth up to 30% of the original purchase price. Items with broken
frames or missing poles are recycled instead and earn a flat $10 credit.
""",
    },
}

HANDBOOK = """\
This handbook describes how Northwind Gear handles orders, shipping, refunds, warranty claims and
the Trailpoints loyalty program for retail customers. It applies to purchases made on
northwindgear.example and in Northwind Gear stores.

# Orders

## Payment methods

We accept Visa, Mastercard, American Express, PayPal and Northwind Gift Cards. Business accounts
with approved credit may pay by purchase order with net-30 terms.

## Order changes and cancellation

Orders can be changed or cancelled free of charge within 2 hours of placement. After 2 hours the
order is released to the warehouse and must be returned under the refund policy instead.

# Shipping

## Free shipping threshold

Standard shipping is free on orders over {free_shipping} within the contiguous United States.
Orders below the threshold pay a flat $7.95 standard shipping fee.

## Delivery times

Standard shipping arrives in 3 to 5 business days. Expedited shipping arrives in 2 business days
for $14.95. Overnight shipping costs $29.95 and is available for orders placed before 1 pm Eastern
time.

## International shipping

We ship to Canada, the United Kingdom, the European Union, Japan and Australia. Import duties and
taxes are paid by the customer on delivery. Stoves and fuel canisters cannot be shipped outside
the United States.

# Refunds

## Physical goods

Unused physical goods can be returned within 60 days for a full refund to the original payment
method. Used items can be returned within 30 days for store credit only.

## Digital goods

Digital goods, including trail map downloads and Northwind Navigator app subscriptions, can be
refunded within {digital_refund_days} of purchase as long as fewer than 3 maps were downloaded.

## Refund processing time

Refunds are issued to the original payment method within 5 to 7 business days after the
warehouse receives the return.

# Warranty

## Standard warranty

All Northwind Gear products are covered against manufacturing defects for {warranty} from the
date of purchase. The warranty does not cover normal wear, misuse or accidental damage.

## Lifetime coverage on packs

Summit series backpacks have lifetime coverage on zippers, buckles and seams. Damaged packs are
repaired free of charge or replaced if repair is not possible.

## Claims process

To file a claim, submit the order number and photos of the defect at
northwindgear.example/warranty. Claims are reviewed within 10 business days.

# Loyalty program

## Trailpoints

Members earn 1 Trailpoint for every dollar spent. Every 500 Trailpoints can be redeemed for a $25
reward certificate. Trailpoints expire after 18 months without a purchase.
{trade_in}"""

INTERNAL_NOTES = """\
Internal operations notes for Northwind Gear staff. Do not share with customers or dealers.

# Supplier costs

## Ridgeline tents

The landed cost of one Ridgeline 2P Tent (NW-1042) is $118. Ridgeline tents are manufactured by
Kestrel Outdoor Manufacturing in Da Nang, Vietnam.

# Price changes

## Planned July 2026 increase

List prices on all Summit backpacks rise by 8% on 2026-07-01. Dealers must not be told about the
increase before 2026-06-01.

# Warehouse

## Returns processing

All customer returns are processed at the Reno, Nevada warehouse, location code RNO-2. Returned
stoves are inspected by the safety team before they can be resold.
"""

FLYER_HEADER = "Northwind Gear | Spring 2026 Product Flyer"
FLYER_FOOTER = "northwindgear.example | Page {page} of {pages}"

# Each page: (left column paragraphs, middle column paragraphs, sidebar paragraphs)
FLYER_PAGES = [
    (
        [
            "Ridgeline 2P Tent (NW-1042). Our best-selling backpacking tent weighs 1.8 kg and "
            "pitches in about 4 minutes. The 20D ripstop fly has a 3000 mm hydrostatic head.",
            "Ridgeline 3P Tent (NW-1043). Room for three with two doors and two vestibules. "
            "Packed weight is 2.4 kg and the floor area is 4.1 square meters.",
        ],
        [
            "Alpine Down Sleeping Bag (NW-1107). Filled with 800-fill responsibly sourced down, "
            "it keeps you warm down to 15F and packs to the size of a loaf of bread.",
            "Alpine Synthetic Sleeping Bag (NW-1108). A budget friendly bag rated to 30F that "
            "stays warm even when damp. Machine washable on a gentle cycle.",
        ],
        [
            "Quick facts",
            "68% of the nylon in our tents is recycled.",
            "Every tent is seam taped by hand.",
        ],
    ),
    (
        [
            "Summit 45L Backpack (NW-2010). A light pack for weekend trips with a removable lid "
            "and a ventilated back panel. Weighs 1.3 kg.",
            "Summit 65L Backpack (NW-2015). Built for long trips, with an adjustable torso and "
            "load lifters. Carries loads up to 25 kg comfortably.",
        ],
        [
            "Trailhead Headlamp 400 (NW-3301). 400 lumens, IPX4 splash resistant and runs 40 "
            "hours on low. Recharges over USB-C in 3 hours.",
            "Trailhead Headlamp 800 (NW-3302). 800 lumens with a red night mode. Waterproof to "
            "IPX7 and runs 60 hours on low.",
        ],
        [
            "Dealer tip",
            "Display headlamps near the register.",
            "Bundle packs with poles for spring sales.",
        ],
    ),
    (
        [
            "Stormshell Jacket (NW-5100). A three-layer waterproof shell with pit zips and a "
            "helmet compatible hood. Weighs 410 g.",
        ],
        [
            "Basecamp Stove (NW-4020). A canister stove that boils 1 liter of water in 3.5 "
            "minutes. Includes a piezo igniter and a windscreen.",
        ],
        [
            "Care guide",
            "Dry tents fully before storage.",
        ],
    ),
]

FLYER_TABLE = [
    ["SKU", "Product", "Weight", "Waterproof rating", "List price"],
    ["NW-1042", "Ridgeline 2P Tent", "1.8 kg", "3000 mm", "$329"],
    ["NW-1043", "Ridgeline 3P Tent", "2.4 kg", "3000 mm", "$389"],
    ["NW-3301", "Trailhead Headlamp 400", "78 g", "IPX4", "$49"],
    ["NW-3302", "Trailhead Headlamp 800", "92 g", "IPX7", "$79"],
    ["NW-5100", "Stormshell Jacket", "410 g", "20000 mm", "$249"],
    ["NW-5200", "Cirrus Trekking Poles", "480 g", "n/a", "$129"],
]

PAGE_W, PAGE_H = letter
# x, width, font size, leading, paragraph spacing (the sidebar reads as one block)
COLUMNS = [(50, 200, 10, 13, 14), (270, 200, 10, 13, 14), (490, 80, 8, 10, 0)]
BODY_TOP = 720

MANIFEST = [
    ("data/raw/policy_handbook_v1.md", "policy_handbook", "v1", ["public"], "2026-03-01"),
    ("data/raw/policy_handbook_v2.md", "policy_handbook", "v2", ["public"], "2026-10-01"),
    ("data/raw/pricing_table.csv", "pricing_table", "v1", ["public"], "2026-09-15"),
    ("data/raw/product_flyer.pdf", "product_flyer", "v1", ["public"], "2026-04-01"),
    ("data/raw/internal_ops_notes.md", "internal_ops_notes", "v1", ["role:internal"], "2026-09-30"),
]


def write_handbooks() -> None:
    for version, facts in HANDBOOK_FACTS.items():
        (RAW / f"policy_handbook_{version}.md").write_text(HANDBOOK.format(**facts))
    (RAW / "internal_ops_notes.md").write_text(INTERNAL_NOTES)


def write_pricing_table() -> None:
    with (RAW / "pricing_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["plan", "sku", "product", "category", "unit_price_usd", "min_order_qty", "lead_days"]
        )
        for plan, (mult, min_qty, lead) in PLANS.items():
            for sku, name, category, price in PRODUCTS:
                writer.writerow([plan, sku, name, category, f"{price * mult:.2f}", min_qty, lead])


def _draw_column(c: canvas.Canvas, paragraphs: list[str], x, width, font, leading, spacing, top):
    y = top
    for para in paragraphs:
        for line in simpleSplit(para, "Helvetica", font, width):
            c.setFont("Helvetica", font)
            c.drawString(x, y, line)
            y -= leading
        y -= spacing
    return y


def write_flyer() -> None:
    c = canvas.Canvas(str(RAW / "product_flyer.pdf"), pagesize=letter, invariant=1)
    pages = len(FLYER_PAGES)
    for page_no, columns in enumerate(FLYER_PAGES, start=1):
        c.setFont("Helvetica-Bold", 9)
        c.drawString(50, PAGE_H - 30, FLYER_HEADER)
        c.setFont("Helvetica", 8)
        c.drawString(50, 30, FLYER_FOOTER.format(page=page_no, pages=pages))
        lowest = BODY_TOP
        for paragraphs, layout in zip(columns, COLUMNS, strict=True):
            lowest = min(lowest, _draw_column(c, paragraphs, *layout, BODY_TOP))
        if page_no == pages:
            table = Table(FLYER_TABLE, colWidths=[70, 160, 70, 110, 70])
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                        ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
                    ]
                )
            )
            _, h = table.wrapOn(c, PAGE_W, PAGE_H)
            table.drawOn(c, 50, lowest - 30 - h)
        c.showPage()
    c.save()


def write_manifest() -> None:
    entries = [
        {"path": p, "doc_id": d, "version": v, "visibility": vis, "updated_at": u}
        for p, d, v, vis, u in MANIFEST
    ]
    (ROOT / "data" / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    write_handbooks()
    write_pricing_table()
    write_flyer()
    write_manifest()
    print(f"Wrote corpus to {RAW}")


if __name__ == "__main__":
    main()
