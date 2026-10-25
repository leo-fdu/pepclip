"""Lazy registry: cached training does not import ESM, RDKit or GROVER."""
REGISTRY = {}


def register_encoder(modality, name, factory):
    key = (modality, name)
    if key in REGISTRY:
        raise ValueError(f"Encoder already registered: {key}")
    REGISTRY[key] = factory


def build_encoder(modality, name, **kwargs):
    if (modality, name) in REGISTRY:
        return REGISTRY[(modality, name)](**kwargs)
    if (modality, name) == ("sequence", "esm2"):
        from .esm2 import ESM2Encoder
        return ESM2Encoder(**kwargs)
    if (modality, name) == ("graph", "grover"):
        from .grover import GroverEncoder
        return GroverEncoder(**kwargs)
    raise ValueError(f"Unknown {modality} encoder: {name}")
