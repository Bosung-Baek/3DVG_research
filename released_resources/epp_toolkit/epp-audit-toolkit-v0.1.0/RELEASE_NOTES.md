# EPP Audit Toolkit v0.1.0

This release contains the dependency-free EPP evaluator, canonical trace
contract, native adapters, tests, wheel, and repository reproduction driver.

Clean-run status:

- SeqVLM ScanRefer: exact reproduction
- SeqVLM NR3D: exact reproduction
- CSVG-compatible ScanRefer: sensitivity-only exact reproduction
- SeeGround: source aggregate hash verified; third-party proposal archive is
  not redistributed, so the full row-level rerun is not included
- M3DRef-CLIP: source aggregate hash verified; no standalone row-level adapter
  artifact is included

No ScanNet/ScanRefer/ReferIt3D data, model weights, rendered images, or API
credentials are included. Obtain third-party resources under their own terms.
