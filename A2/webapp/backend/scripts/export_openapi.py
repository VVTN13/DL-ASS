"""Write the API contract to ``backend/openapi.json``; the frontend types are generated from it.

    python scripts/export_openapi.py        # rewrite the file
    python scripts/export_openapi.py --check  # fail if it is out of date (CI)
"""

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.main import app  # noqa: E402

TARGET = BACKEND / "openapi.json"


def main() -> int:
    content = json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"
    if "--check" in sys.argv:
        if not TARGET.is_file() or TARGET.read_text() != content:
            print("openapi.json is out of date: run scripts/export_openapi.py", file=sys.stderr)
            return 1
        return 0
    TARGET.write_text(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
