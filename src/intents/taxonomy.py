"""Phase 2 intent taxonomy for AppleSupport customer messages.

The taxonomy is *informed by Phase 1 EDA* (``data/processed/eda_summary.json``
provisional topic shares + top terms/bigrams) and by a manual reading of
randomly sampled conversation openers. Design rules:

1. **Every EDA topic above ~2% share earns a class**, because EDA measured
   enough occurrences for the class to matter at golden-set scale
   (200 examples ≈ 0.25% of the 80,247 customer-initiated openers).
2. **EDA topics below ~2.5% that share a routing/escalation profile are
   merged** so that the merged class can reach usable support in a
   200-example golden set:
   - ``account_icloud``   <- Account/Apple ID (2.2%) + iCloud/Backup (1.7%)
   - ``billing_purchases``<- Billing (2.4%) + Refunds (1.5%) + Order (1.0%)
                            + Subscriptions (0.4%)
3. **The large EDA "unmatched" pool (messages matching no keyword topic)
   is not one class.** Reading real openers shows it contains at least
   three distinct behaviours the agent must route differently: pure
   venting with no actionable issue (``general_complaint``), meta
   complaints about *reaching* support (``support_process`` — the natural
   escalation trigger), and everything else (``other``).
4. ``software_bug`` deliberately stays a single class even though EDA's
   "Software / Apps" (29.0%) mixes OS-update regressions with app-specific
   faults: splitting a 29% class into ~24%/5% would leave the app slice
   with ~10 golden examples. The split is revisited when the golden set
   grows.

This module is the single source of truth: the golden-set builder validates
labels against it, the baseline trainer reads its label set, the report and
dashboard render its metadata, and Phase 3+ (embedding classifier) will
reuse it unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IntentClass:
    """One class of the customer-intent taxonomy."""

    label: str
    name: str
    description: str
    eda_anchor: str
    signals: tuple[str, ...]
    escalation_hint: str
    typically_escalated: bool = False
    examples: tuple[str, ...] = field(default_factory=tuple)


INTENT_TAXONOMY: tuple[IntentClass, ...] = (
    IntentClass(
        label="software_bug",
        name="Software / OS bug",
        description=(
            "The message reports a software malfunction: an OS-update "
            "regression (slowdown, battery drain after update), a system or "
            "app bug, crash, freeze, glitch, or a feature that stopped "
            "working as expected."
        ),
        eda_anchor="Software / Apps (29.0% EDA share)",
        signals=(
            "update", "ios", "macos", "bug", "glitch", "crash", "freeze",
            "frozen", "stuck", "lag", "slow", "app", "restart", "reboot",
        ),
        escalation_hint=(
            "Usually auto-handlable with troubleshooting evidence; escalate "
            "when the user reports data loss or a bricked device."
        ),
        examples=(
            "since new update my phone is running slow and battery drains",
            "why when I type the letter I this box keeps popping up",
        ),
    ),
    IntentClass(
        label="device_hardware",
        name="Device / hardware fault",
        description=(
            "The message reports a physical-device problem: battery, screen, "
            "buttons, camera, speaker, charging, water damage, cracks, or a "
            "device that will not turn on / start."
        ),
        eda_anchor="Device / Hardware (14.2% EDA share)",
        signals=(
            "screen", "battery", "broke", "broken", "button", "water",
            "camera", "speaker", "charger", "charging", "drop", "dead",
            "wont turn on", "black screen", "crack",
        ),
        escalation_hint=(
            "Auto-handle with troubleshooting evidence first; hardware "
            "damage usually ends in repair guidance (see repair_warranty)."
        ),
        examples=(
            "My MacBook Air won't start! Was fine last night",
            "anyway you can run a remote test on my battery health?",
        ),
    ),
    IntentClass(
        label="account_icloud",
        name="Account / Apple ID / iCloud",
        description=(
            "Sign-in and account-access problems (Apple ID, password, "
            "verification codes, two-factor) and iCloud data problems "
            "(storage, backup, sync, restore, contacts/photos in the cloud)."
        ),
        eda_anchor="Account / Apple ID (2.2%) + iCloud / Backup (1.7%)",
        signals=(
            "apple id", "password", "sign in", "login", "verification code",
            "2fa", "icloud", "backup", "sync", "storage full", "restore",
        ),
        escalation_hint=(
            "Account recovery steps are well covered by historical "
            "evidence; escalate on suspected takeover (see security_privacy)."
        ),
        examples=(
            "why won't one of my email accounts sign in?",
            "why doesn't it let me turn on iCloud, I click it and nothing happens",
        ),
    ),
    IntentClass(
        label="connectivity",
        name="Connectivity",
        description=(
            "Wireless/network problems: Wi-Fi, Bluetooth pairing, cellular "
            "signal, hotspot, AirDrop, or general connection failures."
        ),
        eda_anchor="Connectivity (2.8% EDA share)",
        signals=(
            "wifi", "wi-fi", "bluetooth", "signal", "cellular", "network",
            "internet", "hotspot", "airdrop", "connect", "no sim",
        ),
        escalation_hint="Auto-handle with standard troubleshooting evidence.",
        examples=(
            "There might be a bug in iPhone X and Bluetooth, keeps "
            "disconnecting from my car",
            'started getting a "No Sim Card Installed" message',
        ),
    ),
    IntentClass(
        label="billing_purchases",
        name="Billing / purchases / orders",
        description=(
            "Anything commercial: charges and invoices, refunds and "
            "purchase problems, subscriptions and trials, and online-store "
            "orders/delivery/shipping."
        ),
        eda_anchor=(
            "Billing (2.4%) + Refunds (1.5%) + Order/Delivery (1.0%) "
            "+ Subscriptions (0.4%) — merged, see module docstring rule 2"
        ),
        signals=(
            "charge", "charged", "bill", "invoice", "refund", "money back",
            "purchase", "bought", "subscription", "trial", "order",
            "delivery", "shipping", "tracking",
        ),
        escalation_hint=(
            "Refunds and account-bound purchases typically need a human "
            "agent with account access — default to escalation."
        ),
        typically_escalated=True,
        examples=(
            "just wasted $30 on a movie I can't even watch now",
            "when is my order going to arrive",
        ),
    ),
    IntentClass(
        label="repair_warranty",
        name="Repair / warranty",
        description=(
            "Repair status and logistics, replacement requests, warranty "
            "coverage questions, Genius Bar / authorized service."
        ),
        eda_anchor="Repair / Warranty (1.9% EDA share)",
        signals=(
            "warranty", "repair", "replace", "replacement", "genius bar",
            "out of warranty", "service",
        ),
        escalation_hint=(
            "Needs appointment/order lookups a historical-evidence agent "
            "cannot perform — escalate unless purely informational."
        ),
        typically_escalated=True,
        examples=("I think I either scratched or cracked my Apple Watch",),
    ),
    IntentClass(
        label="security_privacy",
        name="Security / privacy",
        description=(
            "Phishing and scam messages impersonating Apple, hacked or "
            "compromised accounts, stolen devices, privacy concerns."
        ),
        eda_anchor="Security / Privacy (0.6% EDA share)",
        signals=(
            "phishing", "scam", "hack", "hacked", "stolen", "compromised",
            "suspicious", "fake email", "fake text", "spam", "privacy",
        ),
        escalation_hint=(
            "Safety-critical — always escalate to a human; never "
            "auto-generate account-security advice."
        ),
        typically_escalated=True,
        examples=(
            "This is spam right, I clicked verify your account and it took "
            "me to a webpage of apple ID",
        ),
    ),
    IntentClass(
        label="general_complaint",
        name="General complaint / venting",
        description=(
            "Dissatisfaction with Apple or a product/update WITHOUT any "
            "specific actionable issue, malfunction or request stated. If "
            "the message names a concrete problem, classify by that problem "
            "instead (e.g. software_bug, device_hardware)."
        ),
        eda_anchor="Part of the EDA unmatched pool (no keyword topic)",
        signals=(
            "angry venting", "sarcasm", "brand criticism", "switching to "
            "competitor", "no concrete symptom mentioned",
        ),
        escalation_hint=(
            "Empathetic acknowledgement from evidence is fine; offer "
            "troubleshooting only if the user later states a concrete issue."
        ),
        examples=("FIX IT.", "IOS 11 has killed all the perks of having an iPhone"),
    ),
    IntentClass(
        label="support_process",
        name="Support-process problem",
        description=(
            "Meta-complaints about reaching support itself: cannot reach a "
            "human, useless automated assistant, DM/chat channel problems, "
            "support page or phone-line issues."
        ),
        eda_anchor="Part of the EDA unmatched pool (no keyword topic)",
        signals=(
            "speak to a person", "actual person", "call me", "human",
            "automated answer", "support page", "no one answers",
        ),
        escalation_hint=(
            "The user is explicitly asking for a human — the clearest "
            "escalation trigger in the taxonomy."
        ),
        typically_escalated=True,
        examples=(
            "I'm trying to get into contact with someone from customer "
            "support, I need to speak to an actual person",
        ),
    ),
    IntentClass(
        label="other",
        name="Other / out of scope",
        description=(
            "Everything that fits no other class: pre-sales product "
            "questions (discounts, availability), ambiguous one-word "
            "messages, non-English messages that cannot be confidently "
            "classified, and off-topic content."
        ),
        eda_anchor="Part of the EDA unmatched pool (no keyword topic)",
        signals=("pre-sales question", "discount", "availability", "ambiguous", "off-topic"),
        escalation_hint=(
            "Heterogeneous by construction — default to escalation."
        ),
        typically_escalated=True,
        examples=("Are there any student discounts on laptops in store?"),
    ),
)

TAXONOMY_LABELS: tuple[str, ...] = tuple(c.label for c in INTENT_TAXONOMY)

TAXONOMY_BY_LABEL: dict[str, IntentClass] = {c.label: c for c in INTENT_TAXONOMY}


def validate_label(label: str) -> bool:
    """Return True iff ``label`` is a valid taxonomy label."""
    return label in TAXONOMY_BY_LABEL
