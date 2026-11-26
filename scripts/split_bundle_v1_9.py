"""Split and verify a frozen ZIP without loading it into RAM."""
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def split_and_verify(bundle, size=15 * 1024 * 1024):
    bundle = Path(bundle)
    if size <= 0:
        raise ValueError('Part size must be positive.')
    parts = []
    with bundle.open('rb') as src:
        index = 1
        while True:
            block = src.read(size)
            if not block:
                break
            path = bundle.with_name(bundle.name + f'.part{index:03d}')
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite existing part: {path}')
            with path.open('xb') as out:
                out.write(block)
            parts.append({'name': path.name, 'size_bytes': len(block), 'sha256': sha(path)})
            index += 1
    if not parts:
        raise ValueError('Refusing to split an empty file.')
    combined = hashlib.sha256()
    for part in parts:
        path = bundle.parent / part['name']
        if sha(path) != part['sha256']:
            raise RuntimeError('Part checksum mismatch: ' + part['name'])
        with path.open('rb') as src:
            for block in iter(lambda: src.read(1024 * 1024), b''):
                combined.update(block)
    original_hash = sha(bundle)
    if combined.hexdigest() != original_hash:
        raise RuntimeError('Reassembled SHA-256 does not match original ZIP.')
    manifest = {'original_name': bundle.name, 'original_sha256': original_hash,
                'part_size_bytes': size, 'parts': parts, 'reassembled_sha256_verified': True}
    manifest_path = bundle.with_name(bundle.name + '.parts_manifest.json')
    if manifest_path.exists():
        raise FileExistsError(f'Refusing to overwrite manifest: {manifest_path}')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
