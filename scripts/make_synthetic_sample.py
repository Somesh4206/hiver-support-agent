"""Generate a clearly-labelled SYNTHETIC twcs-format sample.

Purpose: allow the Phase 1 pipeline (and CI-style smoke tests) to run
end-to-end on machines without Kaggle access. The output mimics the twcs.csv
schema (same columns, same Twitter timestamp format, alternating
customer/support threads, deliberate edge cases) but is **not** real data:
every metric computed from it is a pipeline-shape check, never a dataset
statistic. The file name carries ``synthetic`` so downstream run reports and
the dashboard can flag it as such.

Usage:
    python scripts/make_synthetic_sample.py [--rows N] [--output PATH]
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import RAW_DIR, get_settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("make_synthetic_sample")

SUPPORT_ACCOUNT = "AppleSupport"
OTHER_BRANDS = ["AmazonHelp", "SpotifyCares", "Uber_Support", "Tesco"]

# topic -> (customer phrasings incl. typos/short variants, support replies)
APPLE_TOPICS: dict[str, tuple[list[str], list[str]]] = {
    "battery": (
        [
            "my iphone battery drains so fast after the update",
            "battery health dropped 12% in a month, is that normal?",
            "phone wont charge unless i hold the cable at an angle",
            "charging is super slow since ios update",
            "battery dies at 30% help",
        ],
        [
            "We'd like to look into this with you. Can you send us a DM with your "
            "device model and iOS version? We'll take it from there.",
            "Thanks for reaching out — let's start with the basics: force restart "
            "the device and check Settings > Battery for the biggest consumers. "
            "DM us if that doesn't help.",
        ],
    ),
    "icloud": (
        [
            "icloud says storage full but i deleted everything",
            "cant backup to icloud keeps failing",
            "icloud photos not syncing to my new phone",
            "my icloud is full and i dont know what to delete",
        ],
        [
            "Happy to help. Check Settings > [your name] > iCloud > Manage Storage "
            "to see what's using the space — backups of old devices are often the "
            "culprit. We're here if you have questions.",
            "Let's troubleshoot: sign out and back into iCloud, then try the "
            "backup again over Wi-Fi. Send us a DM if it still fails.",
        ],
    ),
    "apple_id": (
        [
            "my apple id is locked for security reasons what do i do",
            "forgot my apple id password and recovery email is old",
            "verification code never arrives on my phone",
            "cant sign into app store keeps saying password wrong but its right",
            "apple id disabled??",
        ],
        [
            "We can help with that. Head to iforgot.apple.com to start account "
            "recovery, or send us a DM and we'll walk you through the steps.",
            "Sorry for the trouble — DM us the device type and we'll check what "
            "options are available for your account.",
        ],
    ),
    "screen": (
        [
            "screen cracked after a small drop, how much to fix?",
            "my screen went black but the phone is on i can hear notifications",
            "there is a green line on my display",
            "screen not responding to touch on the left side",
        ],
        [
            "We're sorry to see that. Screen service options and pricing depend "
            "on your model and coverage — check the 'Get a repair' section or DM "
            "us your device model and we'll point you to the right page.",
            "Let's isolate the issue first: try a force restart. If the display "
            "still misbehaves, hardware service may be needed — DM us and we'll "
            "help arrange it.",
        ],
    ),
    "software": (
        [
            "ios update stuck on apple logo for an hour",
            "phone froze after the update had to force restart twice",
            "apps keep crashing after update",
            "update failed now phone asks to connect to itunes",
            "keyboard lag since update esp in messages",
        ],
        [
            "Thanks for letting us know. If the update stalled, connecting to a "
            "computer and using the Finder/iTunes recovery steps usually gets "
            "things moving. DM us if you get stuck.",
            "Let's try the basics first: close the app, restart the device, and "
            "check for a follow-up update in Settings > General > Software "
            "Update. Let us know how it goes.",
        ],
    ),
    "wifi": (
        [
            "wifi keeps disconnecting at home other devices fine",
            "no service after update cant even call",
            "bluetooth wont pair with my car anymore",
            "airdrop not working between my phone and mac",
        ],
        [
            "Let's troubleshoot together: forget the Wi-Fi network, restart the "
            "device, and rejoin. If it keeps dropping, DM us your iOS version.",
            "We can help with the Bluetooth pairing — remove the car kit from "
            "the Bluetooth list, restart both devices, then pair again. We're "
            "here if it still won't connect.",
        ],
    ),
    "billing": (
        [
            "charged twice for apple music this month",
            "i want a refund for an app i bought by accident",
            "subscription charged even after i cancelled",
            "unknown charge on my card from itunes",
        ],
        [
            "We'd be happy to check this with you. Please report the charge via "
            "reportaproblem.apple.com — refunds there are reviewed case by case. "
            "DM us if you need help finding the right option.",
            "Thanks for reaching out. Charges can take a few days to settle — "
            "check your purchase history in Settings, and use 'Report a Problem' "
            "for anything unexpected.",
        ],
    ),
    "appstore": (
        [
            "app wont download stuck on waiting",
            "in app purchase never showed up but money left my account",
            "app store says update all but nothing updates",
            "cant find an app i purchased before on my new phone",
        ],
        [
            "Let's get that moving: pause and resume the download, then restart "
            "the device if it stays stuck. DM us your Apple ID region if it "
            "still won't budge.",
            "Purchases can be re-downloaded from the App Store's purchased tab "
            "with the same Apple ID. For missing in-app items, 'Report a "
            "Problem' is the fastest path.",
        ],
    ),
    "faceid": (
        [
            "face id not working after update it says move iphone lower",
            "face id keeps failing have to type passcode every time",
            "touch id fails on first try always",
        ],
        [
            "We'll help with that. Go to Settings > Face ID & Passcode > Reset "
            "Face ID and set it up again in good lighting. DM us if it keeps "
            "failing.",
            "Try re-enrolling your fingerprint/face — sometimes an update needs "
            "a fresh scan. We're here if that doesn't do it.",
        ],
    ),
    "order": (
        [
            "where is my order its been 2 weeks",
            "order stuck on processing for 10 days",
            "wrong item shipped what do i do",
            "delivery date changed 3 times",
        ],
        [
            "Sorry about the wait. DM us your order number and we'll check the "
            "status — shipping delays do happen and we can look into options.",
            "We can help with that. Head to your Order Listing for the current "
            "status, or DM us the order number and we'll take a look.",
        ],
    ),
    "repair": (
        [
            "is my phone still under warranty bought it last year",
            "how much is screen repair out of warranty",
            "genius bar appointment available near me this weekend?",
            "my macbook keyboard keys are sticky",
        ],
        [
            "You can check your coverage at checkcoverage.apple.com with the "
            "serial number. For repair pricing by model, the 'Get a repair' "
            "page has the full list — DM us if you can't find your model.",
            "Happy to help arrange that — Genius Bar availability shows in the "
            "support app or website once you pick a store. DM us your city if "
            "you'd like a hand.",
        ],
    ),
    "security": (
        [
            "got an email saying my apple id will be closed is it real",
            "someone logged into my apple id from another country",
            "im getting fake text messages saying my account is locked",
            "i think my account got hacked what do i do",
        ],
        [
            "Good instinct to double-check. Phishing is common — never share "
            "your password or verification codes. Change your password now and "
            "enable two-factor authentication; then DM us and we'll review the "
            "account with you.",
            "Thanks for flagging this. Please change your Apple ID password "
            "immediately and review trusted devices in Settings > [your name]. "
            "DM us afterwards so we can take a closer look.",
        ],
    ),
    "misc": (
        [
            "phone is so slow what can i do",
            "storage full and phone is lagging",
            "how do i transfer data to new iphone",
            "airpods one side not working",
            "watch not syncing steps",
            "pls help",
        ],
        [
            "We can help. A restart often helps with slowdowns; also check "
            "Settings > General > iPhone Storage for big items. DM us if it "
            "stays slow.",
            "Happy to help with the setup — the migration walkthrough in "
            "Settings covers moving data to a new iPhone. We're here if "
            "anything looks off.",
        ],
    ),
}

OTHER_BRAND_TOPICS: dict[str, tuple[list[str], list[str]]] = {
    "delivery": (
        ["where is my package", "delivery late again", "package says delivered but nothing here"],
        ["We're sorry about that — DM us your order number and we'll check it right away."],
    ),
    "account": (
        ["cant log into my account", "password reset email not coming", "account locked help"],
        ["Let's fix that — try the password reset again and DM us your account email if it still fails."],
    ),
    "billing": (
        ["charged twice this month", "refund please", "why was i charged"],
        ["We'd be happy to review the charge — please DM your account details so we can take a look."],
    ),
}

_EDGE_ARTIFACTS = [
    "check this screenshot http://example.com/proof.png",
    "RT @someone: everyone having this problem too?",
    " &amp; now the app wont open at all ",
    "   my   phone   is   slow   ",
]


def _twitter_ts(dt: datetime) -> str:
    return dt.strftime("%a %b %d %H:%M:%S +0000 %Y")


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", type=int, default=420,
                        help="Number of conversations to generate (default 420)")
    parser.add_argument("--output", default=None,
                        help="Output path (default data/raw/twcs_synthetic_sample.csv)")
    args = parser.parse_args()

    rng = random.Random(settings.random_seed)
    output = Path(args.output) if args.output else RAW_DIR / "twcs_synthetic_sample.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    next_tweet_id = 902_000_000_000_000_000
    rows: list[dict] = []

    def emit(author: str, inbound: bool, created: datetime, text: str,
             parent_id: str) -> str:
        nonlocal next_tweet_id
        tid = str(next_tweet_id)
        next_tweet_id += rng.randint(1, 4)
        rows.append({
            "tweet_id": tid,
            "author_id": author,
            "inbound": "True" if inbound else "False",
            "created_at": _twitter_ts(created),
            "text": text,
            "response_tweet_id": "",
            "in_response_to_tweet_id": parent_id,
        })
        return tid

    def link(parent: str, child: str) -> None:
        row = next(r for r in rows if r["tweet_id"] == parent)
        row["response_tweet_id"] = (
            f"{row['response_tweet_id']},{child}" if row["response_tweet_id"] else child
        )

    base = datetime(2017, 11, 1, 8, 0, 0, tzinfo=timezone.utc)
    for conv_no in range(args.conversations):
        is_apple = rng.random() < 0.62
        brand = SUPPORT_ACCOUNT if is_apple else rng.choice(OTHER_BRANDS)
        topic_pool = APPLE_TOPICS if is_apple else OTHER_BRAND_TOPICS
        topic = rng.choice(list(topic_pool))
        customer_texts, support_texts = topic_pool[topic]

        customer_id = str(rng.randint(100_000, 999_999))
        start = base + timedelta(
            days=rng.randint(0, 120), minutes=rng.randint(0, 1400)
        )
        turns = rng.choice([2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8])

        parent = ""
        for turn in range(turns):
            now = start + timedelta(minutes=7 * turn + rng.randint(0, 5))
            customer_text = rng.choice(customer_texts)
            if turn == 0:
                customer_text = f"@{brand} {customer_text}"
            if rng.random() < 0.06:
                customer_text = f"{customer_text} {rng.choice(_EDGE_ARTIFACTS)}"

            # Occasionally the customer double-texts before support replies.
            double_text = None
            if 0 < turn < turns - 1 and rng.random() < 0.12:
                double_text = f"@{brand} {rng.choice(customer_texts)}"

            support_text = f"@{customer_id} {rng.choice(support_texts)}"

            c1 = emit(customer_id, True, now, customer_text, parent)
            prev = c1
            if double_text is not None:
                d = emit(customer_id, True, now + timedelta(minutes=1),
                         double_text, c1)
                link(c1, d)
                prev = d
            s = emit(brand, False, now + timedelta(minutes=4), support_text, prev)
            link(prev, s)
            parent = s
        _ = conv_no

    # --- deliberate edge cases (exercising cleaning + reconstruction) -------
    edge_base = next_tweet_id + 1000
    customer_id = "424242"
    c = emit(customer_id, True, base, "@AppleSupport ", "")
    s = emit(SUPPORT_ACCOUNT, False, base + timedelta(minutes=3),
             f"@{customer_id} Hi there — how can we help today?", c)
    link(c, s)
    rows.append({  # exact duplicate record
        **rows[0], "tweet_id": str(edge_base + 99),
    })
    rows.append({  # empty text
        "tweet_id": str(edge_base + 100), "author_id": "777888",
        "inbound": "True", "created_at": _twitter_ts(base),
        "text": "", "response_tweet_id": "", "in_response_to_tweet_id": "",
    })
    rows.append({  # whitespace-only text
        "tweet_id": str(edge_base + 101), "author_id": "777889",
        "inbound": "True", "created_at": _twitter_ts(base),
        "text": "   ", "response_tweet_id": "", "in_response_to_tweet_id": "",
    })
    rows.append({  # missing author id
        "tweet_id": str(edge_base + 102), "author_id": "",
        "inbound": "True", "created_at": _twitter_ts(base),
        "text": "hello?", "response_tweet_id": "", "in_response_to_tweet_id": "",
    })
    rows.append({  # orphan support reply (parent id absent)
        "tweet_id": str(edge_base + 103), "author_id": SUPPORT_ACCOUNT,
        "inbound": "False", "created_at": _twitter_ts(base),
        "text": "@555001 Happy to help — DM us anytime.",
        "response_tweet_id": "", "in_response_to_tweet_id": str(edge_base + 555),
    })
    # a 2-cycle with no root (back-edge guard)
    a, b = str(edge_base + 110), str(edge_base + 111)
    rows.append({"tweet_id": a, "author_id": "900001", "inbound": "True",
                 "created_at": _twitter_ts(base), "text": "@AppleSupport test",
                 "response_tweet_id": b, "in_response_to_tweet_id": b})
    rows.append({"tweet_id": b, "author_id": "900002", "inbound": "True",
                 "created_at": _twitter_ts(base), "text": "@AppleSupport test 2",
                 "response_tweet_id": a, "in_response_to_tweet_id": a})

    frame = pd.DataFrame(rows, columns=[
        "tweet_id", "author_id", "inbound", "created_at", "text",
        "response_tweet_id", "in_response_to_tweet_id",
    ])
    frame = frame.sample(frac=1.0, random_state=settings.random_seed).reset_index(drop=True)
    frame.to_csv(output, index=False)

    apple_out = int((frame["author_id"] == SUPPORT_ACCOUNT).sum())
    logger.info("Wrote SYNTHETIC sample: %s (%d rows, %d AppleSupport replies, "
                "%d conversations requested)", output, len(frame), apple_out,
                args.conversations)
    logger.info("This file is for pipeline smoke tests only — never for "
                "reported dataset metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
