#!/usr/bin/env python3
"""
Simple VRM/GLB validation diagnostics.
Saves a short report and tries both binary and JSON parsing using pygltflib.

Usage:
  python tools/validate_vrm.py assets/models/AvatarSample_D.vrm
  python tools/validate_vrm.py assets/models/*.vrm

This script is intentionally minimal and has no external dependencies besides pygltflib.
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime
import glob
import shutil
import subprocess
# json will be imported where needed for report writing

try:
    from pygltflib import GLTF2
except Exception:
    print("pygltflib is required. Install with: pip install pygltflib")
    raise

REPORT_DIR = Path("logs")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_file(path: Path):
    report = {
        'file': str(path),
        'size': path.stat().st_size,
        'is_glb_magic': False,
        'binary_load': None,
        'json_load': None,
        'extensions': None,
        'has_external_resources': False,
        'gltf_transform_inspect': None,
        'embed_attempt': None,
        'error': None,
    }

    raw = path.read_bytes()
    # Check GLB magic ("glTF")
    if len(raw) >= 4 and raw[:4] == b'glTF':
        report['is_glb_magic'] = True

    # Try binary load
    try:
        glb = GLTF2.load_binary(str(path))
        report['binary_load'] = 'ok'
        exts = getattr(glb, 'extensions', None)
        report['extensions'] = {} if not exts else exts
    except Exception as e:
        report['binary_load'] = f'fail: {e}'

    # Try JSON load
    try:
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            text = raw.decode('utf-8', errors='replace')

        obj = GLTF2.from_json(text)
        report['json_load'] = 'ok'
        exts = getattr(obj, 'extensions', None)
        report['extensions'] = report['extensions'] or ({} if not exts else exts)
        # Detect external resources (images/buffers with uri not data:)
        try:
            imgs = getattr(obj, 'images', []) or []
            for im in imgs:
                uri = getattr(im, 'uri', None)
                if uri and not uri.startswith('data:'):
                    report['has_external_resources'] = True
            bufs = getattr(obj, 'buffers', []) or []
            for b in bufs:
                uri = getattr(b, 'uri', None)
                if uri and not uri.startswith('data:'):
                    report['has_external_resources'] = True
        except Exception:
            # ignore detection errors
            pass
    except Exception as e:
        report['json_load'] = f'fail: {e}'

    # If both failed, capture a short traceback snippet
    if (not report['binary_load'] or report['binary_load'].startswith('fail')) and \
       (not report['json_load'] or report['json_load'].startswith('fail')):
        report['error'] = 'Both binary and JSON parsing failed'

    return report


def run_gltf_transform_inspect(path: Path):
    """Run `gltf-transform inspect` if available and return output."""
    cmd = shutil.which('gltf-transform')
    if not cmd:
        return {'available': False, 'output': None}

    try:
        proc = subprocess.run([cmd, 'inspect', str(path)], capture_output=True, text=True, timeout=60)
        return {'available': True, 'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
    except Exception as e:
        return {'available': True, 'error': str(e)}


def attempt_embed_with_gltf_transform(path: Path):
    """If gltf-transform is available, run `gltf-transform copy --embed` to produce an embedded GLB and return path."""
    cmd = shutil.which('gltf-transform')
    if not cmd:
        return {'available': False, 'error': 'gltf-transform not found'}

    out_dir = Path('assets/models/embedded')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (path.stem + '.embedded.glb')

    try:
        proc = subprocess.run([cmd, 'copy', str(path), str(out_path), '--embed'], capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            return {'available': True, 'success': False, 'stdout': proc.stdout, 'stderr': proc.stderr}

        return {'available': True, 'success': True, 'out_path': str(out_path), 'stdout': proc.stdout}
    except Exception as e:
        return {'available': True, 'success': False, 'error': str(e)}


def main(paths):
    results = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"Skipping missing: {p}")
            continue
        try:
            r = inspect_file(path)
            results.append(r)
            print(f"Inspected {p}: size={r['size']} glb_magic={r['is_glb_magic']} binary={r['binary_load']} json={r['json_load']}")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error inspecting {p}: {e}\n{tb}")
            results.append({'file': str(p), 'error': str(e)})

    # Save report
    # Use a filesystem-safe timestamp for Windows (no colons)
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = REPORT_DIR / f"vrm_report_{now}.json"
    import json
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({'generated': now, 'results': results}, fh, indent=2, ensure_ascii=False)

    print(f"Report written to {out}")


if __name__ == '__main__':
    raw_args = sys.argv[1:]
    # If no args provided, default to the sample VRM
    if not raw_args:
        patterns = ['assets/models/AvatarSample_D.vrm']
    else:
        patterns = raw_args

    # Expand glob patterns (works on Windows powerShell where shell may not expand)
    paths = []
    for p in patterns:
        expanded = glob.glob(p)
        if expanded:
            paths.extend(expanded)
        else:
            # Keep literal path if nothing matched (so main will skip missing)
            paths.append(p)

    main(paths)
