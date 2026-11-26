"""Canonical v1.6 AdaptiveDeltaXAI implementation."""

from .swing import multipath_integrated_gradients, swing_attribution, linear_completeness_check

__all__ = ["multipath_integrated_gradients", "swing_attribution", "linear_completeness_check",
           "AdaptiveSwingController", "StepRecord", "summarize"]
