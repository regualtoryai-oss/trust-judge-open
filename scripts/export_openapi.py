"""Write the implemented API contract without starting servers or model calls."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import create_app

if __name__ == '__main__':
    target = Path(__file__).resolve().parents[1] / 'contracts' / 'runtime-openapi.json'
    target.write_text(json.dumps(create_app().openapi(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(target)
