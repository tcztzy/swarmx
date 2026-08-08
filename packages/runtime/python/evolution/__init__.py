"""Skill evolution sidecar: DSPy/GEPA candidate generation.

This package is imported only by the `swarmx.evolve_skill` worker operation for
the `dspy.gepa.v1` optimizer. It lives in the locked `evolution` dependency
group and never touches active Skill pointers, promotions, Sessions, provider
credentials, or the Skill install directory.
"""
