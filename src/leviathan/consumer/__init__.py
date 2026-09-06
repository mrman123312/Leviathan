"""Consumer Leviathan. Neural modules are imported explicitly, not by core CI."""
from .profiles import PROFILES, ModelProfile, get_profile

__all__ = ["PROFILES", "ModelProfile", "get_profile"]
