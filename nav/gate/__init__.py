"""Perception gate: a target cluster must pass the Verifier (``perception.server`` over HTTP) before the planner may stop
at it; when the first view is insufficient the gate drives a closed-loop approach (``approach.LiveApproach``). See
``gate.VerificationGate`` and README.md."""
