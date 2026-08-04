"""Smoke test - run by the harness against the live ephemeral app.

Unit tests of the pure helpers run free (no GPU, no deploy); integration
tests take the harness `app_client` fixture and skip unless HARNESS_APP_URL
is set (harness/check.py sets it after the ephemeral deploy).
"""
