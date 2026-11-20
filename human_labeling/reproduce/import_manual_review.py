"""Validate an actual human export; drafts and doubts never become golden labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from pipeline import bbox_answer

P = Path(__file__).resolve().parent


def validate(data: dict, packet_path: Path | None = None, source: str = 'manual-review36-v1', frames_path: Path | None = None) -> dict:
    packet = json.loads((packet_path or P / 'manual-review-packet36.json').read_text())
    expected = {r['photo_id']: r for r in packet}
    if data.get('schema') != 'final844-human-adjudication-v1':
        raise ValueError('Unexpected human export schema')
    if data.get('packet_photo_ids') != [r['photo_id'] for r in packet]:
        raise ValueError('Export belongs to a different packet')
    confirmed, uncertain, seen = [], [], set()
    for answer in data['answers']:
        pid = answer['photo_id']
        if pid not in expected or pid in seen:
            raise ValueError('Unknown or duplicate photo ID')
        seen.add(pid)
        record = expected[pid]
        for key in ('candidate_id', 'issued_sha256', 'width', 'height', 'original_status', 'original_boxes'):
            if answer.get(key) != record.get(key):
                raise ValueError(f'{pid}: changed source identity: {key}')
        actual_hash = hashlib.sha256(((frames_path or P / 'issued_frames') / f'{pid}.jpg').read_bytes()).hexdigest()
        if actual_hash != record['issued_sha256']:
            raise ValueError(f'{pid}: local issued frame changed')
        if answer.get('actor') != 'Human team adjudicator (user)' or answer.get('source') != source:
            raise ValueError(f'{pid}: unexpected adjudication provenance')
        datetime.fromisoformat(answer['reviewed_at'].replace('Z', '+00:00'))
        if answer['resolution'] not in ('CONFIRMED', 'UNCERTAIN'):
            raise ValueError(f'{pid}: unknown resolution')
        status, boxes = bbox_answer(answer)
        if status not in ('PRESENT', 'ABSENT'):
            raise ValueError(f'{pid}: unresolved annotation status')
        if answer['resolution'] == 'CONFIRMED':
            try:
                original_status, original_boxes = bbox_answer({'status': record['original_status'], 'boxes': record['original_boxes']})
                changed = status != original_status or boxes != original_boxes
            except ValueError:
                changed = True
            if changed and not answer.get('note', '').strip():
                raise ValueError(f'{pid}: correction has no explanation')
            confirmed.append(dict(answer, boxes=[dict(b, shape='rectangle') for b in boxes]))
        else:
            uncertain.append(answer)
    if data.get('confirmed_count') != len(confirmed):
        raise ValueError('Confirmed count mismatch')
    return {'source_schema': data['schema'], 'exported_at': data['exported_at'], 'review_source': source,
            'confirmed': confirmed, 'uncertain': uncertain,
            'unreviewed_photo_ids': sorted(set(expected) - seen),
            'complete': len(confirmed) == len(expected),
            'drafts_are_final_labels': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', type=Path)
    parser.add_argument('--output', type=Path, default=P / 'manual-adjudication-human.json')
    parser.add_argument('--packet', type=Path, default=P / 'manual-review-packet36.json')
    parser.add_argument('--source', default='manual-review36-v1')
    parser.add_argument('--frames', type=Path, default=P / 'issued_frames')
    args = parser.parse_args()
    result = validate(json.loads(args.export.read_text()), args.packet, args.source, args.frames)
    if args.output.exists():
        raise SystemExit('Output exists; use a new --output to preserve history')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'confirmed': len(result['confirmed']), 'uncertain': len(result['uncertain']),
                      'unreviewed': len(result['unreviewed_photo_ids']), 'complete': result['complete']}))
