"""Print the key structure of a JSON file (top levels, truncated leaves). Usage: python scripts/inspect_json.py FILE [depth]"""
import json
import sys


def walk(o, depth, ind=""):
    if isinstance(o, dict):
        for k, v in list(o.items())[:25]:
            if isinstance(v, (dict, list)) and depth > 0:
                print(f"{ind}{k}: {type(v).__name__}[{len(v)}]")
                walk(v, depth - 1, ind + "  ")
            else:
                print(f"{ind}{k}: {str(v)[:100]}")
    elif isinstance(o, list) and o:
        print(f"{ind}[0] of {len(o)}:")
        walk(o[0], depth - 1, ind + "  ")


walk(json.load(open(sys.argv[1])), int(sys.argv[2]) if len(sys.argv) > 2 else 3)
