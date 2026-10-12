"""Adduct and polarity routing module for CASMI26 (Phase 2).

Disentangles positive mode ([M+H]+, [M+Na]+, etc.) from negative mode ([M-H]-)
and provides principled, evidence-backed simulator / retrieval weights instead of
blanket heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class IonMode(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class AdductClass(Enum):
    PROTONATED = "[M+H]+"
    DEPROTONATED = "[M-H]-"
    SODIATED = "[M+Na]+"
    POTASSIATED = "[M+K]+"
    AMMONIATED = "[M+NH4]+"
    FORMATED = "[M+FA-H]-"
    CHLORINATED = "[M+Cl]-"
    ACETATED = "[M+AcO-H]-"
    OTHER = "other"


POSITIVE_ADDUCTS: Set[str] = {
    "[M+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M+NH4]+",
    "[M+H-H2O]+",
    "[2M+H]+",
    "[M+2H]2+",
}

NEGATIVE_ADDUCTS: Set[str] = {
    "[M-H]-",
    "[M+FA-H]-",
    "[M+Cl]-",
    "[M+AcO-H]-",
    "[M-H-H2O]-",
    "[2M-H]-",
    "[M-2H]2-",
}


@dataclass(frozen=True)
class AdductPolicy:
    adduct: str
    ion_mode: IonMode
    adduct_class: AdductClass
    glacier_eligible: bool
    iceberg_eligible: bool
    popularity_prior_weight: float
    engine_blend_alpha: float
    description: str


DEFAULT_POLICIES: Dict[str, AdductPolicy] = {
    "[M+H]+": AdductPolicy(
        adduct="[M+H]+",
        ion_mode=IonMode.POSITIVE,
        adduct_class=AdductClass.PROTONATED,
        glacier_eligible=True,
        iceberg_eligible=False,  # GLACIER takes precedence; ICE only on fallback
        popularity_prior_weight=0.25,
        engine_blend_alpha=0.60,
        description="Standard positive protonated adduct; primary GLACIER target.",
    ),
    "[M-H]-": AdductPolicy(
        adduct="[M-H]-",
        ion_mode=IonMode.NEGATIVE,
        adduct_class=AdductClass.DEPROTONATED,
        glacier_eligible=False,  # GLACIER does not support negative mode
        iceberg_eligible=True,
        popularity_prior_weight=0.25,
        engine_blend_alpha=0.60,
        description="Standard negative deprotonated adduct; dedicated ICE/retrieval branch.",
    ),
    "[M+Na]+": AdductPolicy(
        adduct="[M+Na]+",
        ion_mode=IonMode.POSITIVE,
        adduct_class=AdductClass.SODIATED,
        glacier_eligible=False,  # Sodium adduct fragmentation is poorly simulated
        iceberg_eligible=True,
        popularity_prior_weight=0.35,  # Upweight popularity due to sparse informative fragments
        engine_blend_alpha=0.55,
        description="Sodiated adduct; elevated molecular prior to counter sparse fragmentation.",
    ),
    "[M+NH4]+": AdductPolicy(
        adduct="[M+NH4]+",
        ion_mode=IonMode.POSITIVE,
        adduct_class=AdductClass.AMMONIATED,
        glacier_eligible=False,
        iceberg_eligible=True,
        popularity_prior_weight=0.30,
        engine_blend_alpha=0.58,
        description="Ammoniated adduct; ICE fallback eligible.",
    ),
}


def classify_adduct(adduct_str: Optional[str], mode_val: Optional[Any] = None) -> AdductPolicy:
    """Classify an adduct string and return its corresponding execution policy.

    Args:
        adduct_str: Raw adduct string from spectrum metadata (e.g. '[M+H]+').
        mode_val: Ionization mode indicator (1, -1, 'positive', 'negative', etc.).
    """
    cleaned = (adduct_str or "").strip()
    if cleaned in DEFAULT_POLICIES:
        return DEFAULT_POLICIES[cleaned]

    # Infer mode
    mode = IonMode.UNKNOWN
    if mode_val in (1, 1.0, "1", "positive", "pos", "+"):
        mode = IonMode.POSITIVE
    elif mode_val in (-1, -1.0, "-1", "negative", "neg", "-"):
        mode = IonMode.NEGATIVE
    elif cleaned.endswith("+"):
        mode = IonMode.POSITIVE
    elif cleaned.endswith("-"):
        mode = IonMode.NEGATIVE

    # Identify class
    if cleaned in POSITIVE_ADDUCTS:
        adduct_class = AdductClass.PROTONATED if cleaned == "[M+H]+" else AdductClass.OTHER
        mode = IonMode.POSITIVE
    elif cleaned in NEGATIVE_ADDUCTS:
        adduct_class = AdductClass.DEPROTONATED if cleaned == "[M-H]-" else AdductClass.OTHER
        mode = IonMode.NEGATIVE
    else:
        adduct_class = AdductClass.OTHER

    # Determine fallback policy
    is_gl = (cleaned == "[M+H]+")
    is_ice = not is_gl
    pop_w = 0.30 if adduct_class in (AdductClass.SODIATED, AdductClass.POTASSIATED) else 0.25

    return AdductPolicy(
        adduct=cleaned or "unknown",
        ion_mode=mode,
        adduct_class=adduct_class,
        glacier_eligible=is_gl,
        iceberg_eligible=is_ice,
        popularity_prior_weight=pop_w,
        engine_blend_alpha=0.60,
        description=f"Inferred policy for {cleaned} ({mode.value})",
    )


def summarize_molecule_adducts(spectra: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Inspect all spectra of a molecule and produce a consolidated adduct profile."""
    adducts = [s.get("adduct") for s in spectra if s.get("adduct")]
    modes = [s.get("mode") for s in spectra if s.get("mode") is not None]

    primary_adduct = adducts[0] if adducts else None
    primary_mode = modes[0] if modes else None

    # Check for [M+H]+ presence
    has_protonated = any(a == "[M+H]+" for a in adducts)
    has_deprotonated = any(a == "[M-H]-" for a in adducts)

    policy = classify_adduct(primary_adduct, primary_mode)

    return {
        "adduct_count": len(adducts),
        "primary_adduct": primary_adduct,
        "primary_mode": policy.ion_mode.value,
        "has_m_plus_h": has_protonated,
        "has_m_minus_h": has_deprotonated,
        "glacier_eligible": has_protonated,
        "policy": policy,
    }
