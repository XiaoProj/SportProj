"""Feature-extraction entry point placeholder.

This stage intentionally does not process raw data. Future implementation should
write derived feature tables only under outputs/processed/ or outputs/results/.
"""

from __future__ import annotations


def main() -> int:
    print("Feature extraction framework is scaffolded. No raw data processing was run.")
    print("Next stage: implement adapters that convert confirmed pose schemas into feature tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
