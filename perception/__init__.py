"""SAM3 -> SAM3D -> view selection (VQF or top-2 gap) -> SAM3, with an optional
ensemble filter: verify object detections from a better viewpoint.

See ``perception.client.Verifier``; ``python -m perception`` runs it on one frame.
"""
__all__ = ["HOUSEHOLD", "Verdict", "Verifier", "View"]


def __getattr__(name):
    if name in __all__:
        from perception import client
        return getattr(client, name)
    raise AttributeError(name)
