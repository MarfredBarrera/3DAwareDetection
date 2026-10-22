"""SAM3 -> SAM3D -> view selection (VQF or top-2 gap) -> SAM3, with an optional
ensemble filter: verify object detections from a better viewpoint.

See ``perception.client.Verifier``; ``python -m perception`` runs it on one frame.
"""
from perception.client import HOUSEHOLD, Verdict, Verifier, View

__all__ = ["HOUSEHOLD", "Verdict", "Verifier", "View"]
