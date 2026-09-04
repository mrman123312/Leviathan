"""Leviathan research scaffold."""

from .controller import BaselineMetaController, UtilityWeights, should_continue_deliberating, utility
from .deepseek_v4 import (
    CANONICAL_MODEL_ID,
    CANONICAL_REPO_ID,
    DeepSeekV4Fingerprint,
    DeepSeekV4Manifest,
    MixtureOfParametersPlan,
    build_manifest,
    verify_full_checkpoint_files,
)
from .trust import PromotionThresholds, candidate_trust, may_promote, verification_trust
from .types import (
    Belief,
    CognitiveMode,
    LearningCandidate,
    MetaState,
    ModeEstimate,
    Provenance,
    ProvenanceKind,
    UncertaintyKind,
    Verification,
)

__all__ = [
    "BaselineMetaController",
    "Belief",
    "CANONICAL_MODEL_ID",
    "CANONICAL_REPO_ID",
    "CognitiveMode",
    "DeepSeekV4Fingerprint",
    "DeepSeekV4Manifest",
    "LearningCandidate",
    "MetaState",
    "MixtureOfParametersPlan",
    "ModeEstimate",
    "PromotionThresholds",
    "Provenance",
    "ProvenanceKind",
    "UncertaintyKind",
    "UtilityWeights",
    "Verification",
    "build_manifest",
    "candidate_trust",
    "may_promote",
    "should_continue_deliberating",
    "utility",
    "verification_trust",
    "verify_full_checkpoint_files",
]
