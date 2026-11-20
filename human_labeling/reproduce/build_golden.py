"""Build the 400-photo team golden set from frozen answers and adjudications.

No network access, uploads, payment, or course submission. Preserve all raw inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from PIL import Image
from pipeline import bbox_answer

P = Path(__file__).resolve().parent
WEAK_REVIEWER = 'd7210ff3934608d1832e94043b3e4977'


def read(path: Path) -> object:
    return json.loads(path.read_text())


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def version(row: dict) -> int:
    match = re.search(r'-v(\d+)$', row['candidate_id'])
    return int(match.group(1)) if match else 1


def latest_versions(rows: list[dict]) -> dict[int, dict]:
    result = {}
    seen = set()
    for row in rows:
        key = (row['photo_id'], version(row))
        if key in seen:
            raise ValueError(f'Duplicate candidate version {key}')
        seen.add(key)
        previous = result.get(row['photo_id'])
        if previous is None or version(row) > version(previous):
            result[row['photo_id']] = row
    return result


def build(root: Path, frames: Path) -> tuple[list[dict], list[dict], list[dict], dict]:
    model = read_lines(root / 'model_manifest.jsonl')
    metadata = {r['photo_id']: r for r in read_lines(root / 'main_manifest.jsonl')}
    expected = {r['photo_id'] for r in model}
    if len(model) != 400 or len(expected) != 400 or set(metadata) != expected:
        raise ValueError('Frozen 400-photo scope changed')
    main_rows = read(root / 'main-review-final-aggregation.json')['items']
    pilot_rows = read(root / 'pilot-review-final-aggregation.json')['items']
    latest = latest_versions(main_rows)
    if {k: v['candidate_id'] for k, v in latest.items()} != {
        k: v['candidate_id'] for k, v in latest_versions(list(reversed(main_rows))).items()
    }:
        raise ValueError('Candidate selection depends on serialization order')
    history = defaultdict(list)
    for r in pilot_rows + main_rows:
        if len({v['worker_id'] for v in r['votes']}) != len(r['votes']):
            raise ValueError('Repeated reviewer')
        if any(v['worker_id'] == r['author_id'] for v in r['votes']):
            raise ValueError('Self review')
        history[r['photo_id']].append(r)
    original_reviewed = {r['photo_id']: r for r in pilot_rows + main_rows if version(r) == 1}
    if len(original_reviewed) != 396 or any(len(r['votes']) != 2 for r in original_reviewed.values()):
        raise ValueError('Expected two original independent reviews for 396 photos')
    final = {pid: {'status': r['status'], 'boxes': r['boxes']} for pid, r in latest.items()}
    routes = {}
    for pid, r in latest.items():
        if len(r['votes']) == 1 and r['votes'][0]['verdict'] == 'OK':
            routes[pid] = 'model_correction_then_one_independent_OK_and_visual_adjudication'
        elif len(r['votes']) >= 2 and all(v['verdict'] == 'OK' for v in r['votes']):
            routes[pid] = 'two_independent_OK'
        elif pid == 760828121:
            routes[pid] = 'visual_adjudication_seven_butterflies_two_OK_one_BAD'
        else:
            routes[pid] = 'unresolved'
    partial = read_lines(root / 'golden-reviewed-partial39.jsonl')
    partial_provenance = {r['photo_id']: r for r in read(root / 'golden-reviewed-partial39-provenance.json')}
    for r in partial:
        pid = r['photo_id']
        provenance = partial_provenance[pid]
        if pid in latest and latest[pid]['candidate_id'] != provenance['candidate_id']:
            raise ValueError('Prior golden version is obsolete')
        final[pid] = {'status': 'PRESENT' if r['boxes'] else 'ABSENT', 'boxes': r['boxes']}
        routes[pid] = 'previously_visually_adjudicated_pilot_or_first20'
    manual36 = read(root / 'manual-adjudication-human-20261117.json')
    manual17 = read(root / 'manual-adjudication-human-corrections17-20261117.json')
    human = {}
    for source, count in ((manual36, 36), (manual17, 17)):
        if not source['complete'] or len(source['confirmed']) != count or source['uncertain'] or source['unreviewed_photo_ids']:
            raise ValueError('Human adjudication incomplete')
        if len({r['photo_id'] for r in source['confirmed']}) != count:
            raise ValueError('Duplicate human photo')
        for r in source['confirmed']:
            pid = r['photo_id']
            if pid not in expected or r['resolution'] != 'CONFIRMED':
                raise ValueError('Unexpected human decision')
            if source is manual17 and (pid not in human or r['original_status'] != human[pid][-1]['status'] or r['original_boxes'] != human[pid][-1]['boxes']):
                raise ValueError('Human correction ancestry mismatch')
            human.setdefault(pid, []).append(r)
            final[pid] = {'status': r['status'], 'boxes': r['boxes']}
            routes[pid] = 'human_team_confirmed_model_proposal' if source is manual17 else 'human_team_adjudication'
    if set(final) != expected or any(v == 'unresolved' for v in routes.values()):
        raise ValueError('Missing, extra, or unresolved photos')
    audit = read(root / 'final-audit-completed68.json')
    audited = {r['photo_id'] for r in audit['items'] if r['audit_status'] == 'passed_after_adjudication'}
    if len(audited) != 68:
        raise ValueError('Visual audit incomplete')
    raw = read(root / 'platform-final-20261117.json')
    originals = {}
    for pool in raw['pools']:
        if pool['pool']['id'] not in ('6868003', '6868546'):
            continue
        for a in pool['assignments']:
            if a['status'] != 'ACCEPTED':
                continue
            if len(a['tasks']) != len(a['solutions']) or len(a['tasks']) != 1:
                raise ValueError('Unexpected bbox page')
            pid = int(a['tasks'][0]['input_values']['photo_id'])
            if pid in originals:
                raise ValueError('Duplicate original bbox answer')
            originals[pid] = {'assignment_id': a['id'], 'author_id': a['user_id'],
                              'output': a['solutions'][0]['output_values']}
    if set(originals) != expected:
        raise ValueError('Paid original answers do not cover frozen 400')
    weak_ids = {r['photo_id'] for r in read(root / 'control-failure-audit12.json')}
    proposals = {r['photo_id']: r for r in read(root / 'final-source-corrections17-proposals.json')}
    golden, provenance, manifest = [], [], []
    hashes = set()
    for m in sorted(model, key=lambda r: r['photo_id']):
        pid = m['photo_id']
        data = (frames / f'{pid}.jpg').read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != m['issued_sha256'] or digest in hashes:
            raise ValueError(f'{pid}: changed or duplicate issued image')
        hashes.add(digest)
        with Image.open(frames / f'{pid}.jpg') as im:
            im.verify()
        with Image.open(frames / f'{pid}.jpg') as im:
            if im.size != (m['width'], m['height']):
                raise ValueError(f'{pid}: changed dimensions')
        if metadata[pid]['license_code'] != 'cc0':
            raise ValueError(f'{pid}: missing CC0 evidence')
        status, boxes = bbox_answer(final[pid])
        if status not in ('PRESENT', 'ABSENT'):
            raise ValueError('Unresolved final status')
        if len({json.dumps(b, sort_keys=True) for b in boxes}) != len(boxes):
            raise ValueError('Duplicate final rectangles')
        for h in human.get(pid, []):
            if h['issued_sha256'] != digest or (h['width'], h['height']) != (m['width'], m['height']):
                raise ValueError('Human source changed')
        golden.append({'photo_id': pid, 'boxes': boxes, 'error': None})
        exclusions = [dict(v, candidate_id=r['candidate_id'], exclusion='Two incorrect controls out of three in main review; replaced by explicit team adjudication')
                      for r in history[pid] for v in r['votes']
                      if pid in weak_ids and v['worker_id'] == WEAK_REVIEWER]
        if exclusions and pid not in human:
            raise ValueError('Weak review has not been adjudicated')
        adjudicator = ('human team' if pid in human else
                       'two qualified platform reviewers; sampled visual audit recorded separately'
                       if routes[pid] == 'two_independent_OK' else
                       'documented Codex-assisted visual adjudication with retained human reviews')
        provenance.append({'photo_id': pid, 'issued_sha256': digest, 'final_status': status,
                           'final_boxes': boxes, 'decision_route': routes[pid],
                           'original_bbox': originals[pid], 'candidate_history': history[pid],
                           'prior_adjudication': partial_provenance.get(pid),
                           'human_team_adjudications': human.get(pid, []),
                           'final_model_proposal': proposals.get(pid),
                           'quarantined_votes_retained_for_audit': exclusions,
                           'source_visual_audit': pid in audited,
                           'final_adjudicator': adjudicator})
        manifest.append(dict(metadata[pid], **{k: m[k] for k in ('issued_sha256', 'width', 'height', 'source_sha256', 'frame_transform')},
                             source_local_path=metadata[pid]['local_path'],
                             source_task_image_path=metadata[pid]['task_image_path'],
                             image=f'images/{pid}.jpg', local_path=f'images/{pid}.jpg',
                             task_image_path=f'images/{pid}.jpg', task_image_url=m['task_image_url'],
                             sha256=digest, annotation_status='final',
                             annotation_status_label=status, object_count=len(boxes)))
    stats = {'complete': True, 'photos': len(golden), 'unique_image_hashes': len(hashes),
             'positive_photos': sum(bool(r['boxes']) for r in golden),
             'negative_photos': sum(not r['boxes'] for r in golden),
             'boxes': sum(len(r['boxes']) for r in golden), 'unresolved': 0,
             'original_photos_with_two_independent_platform_reviews': len(original_reviewed),
             'special_original_answers_resolved_by_team': 4,
             'human_team_unique_photos': len(human), 'human_team_decisions': 53,
             'source_visual_audit_photos': len(audited),
             'decision_routes': dict(Counter(routes.values())),
             'photos_by_group': dict(Counter(r['group'] for r in manifest)),
             'quarantined_main_votes': sum(len(r['quarantined_votes_retained_for_audit']) for r in provenance),
             'latest_candidate_selection': 'highest numeric version, independent of input order',
             'final_course_submission': False}
    return golden, provenance, manifest, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=P)
    parser.add_argument('--frames', type=Path, default=P / 'issued_frames')
    parser.add_argument('--output', type=Path, default=P / 'delivery')
    args = parser.parse_args()
    golden, provenance, manifest, stats = build(args.root, args.frames)
    args.output.mkdir(exist_ok=False, parents=True)
    for name, value in [('golden.json', golden), ('provenance.json', provenance), ('validation.json', stats)]:
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    for name, rows in [('golden.jsonl', golden), ('manifest.jsonl', manifest)]:
        (args.output / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
