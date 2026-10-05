"""NVPA: whether each required non-verbal vocalization is present, of the right type, at the intended position.

Method (development approximation, NOT the organizers' implementation):
  1. ASR gives recognised words with token timestamps (shared with WER).
  2. Reference words are aligned to recognised words, so every GAP between two reference
     words maps to an audio window [end of previous word, start of next word].
  3. A window-level detector estimates which NV types are present in each window.
  4. Gold events (type, gap) are matched one-to-one to detections within a position tolerance.
"""
