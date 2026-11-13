from .attnsleep import AttnSleep
from .transformer import Transformer

NETWORKS = {"Transformer": Transformer, "AttnSleep": AttnSleep}

__all__ = ["NETWORKS", "AttnSleep", "Transformer"]
