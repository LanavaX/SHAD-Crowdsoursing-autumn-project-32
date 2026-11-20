"""Audit reviewer independence and aggregate actual votes without changing labels."""
from pathlib import Path
from collections import Counter
import argparse
import json
from pipeline import normalized_boxes, review_decision

P = Path(__file__).resolve().parent


def aggregate(assignments: dict, lineage: list[dict], qualified: set[str]) -> dict:
    candidates = {r['candidate_id']: r for r in lineage}
    if len(candidates) != len(lineage):
        raise ValueError('Duplicate candidate')
    gold = {r['candidate_id']: r for r in json.loads((P / 'review-gold.json').read_text())}
    votes = {key: [] for key in candidates}
    controls, excluded = [], []
    for assignment in assignments['items']:
        if assignment['status'] not in ('ACCEPTED', 'SUBMITTED'):
            continue
        tasks, solutions = assignment['tasks'], assignment['solutions']
        if len(tasks) != len(solutions):
            raise ValueError('Missing review solutions')
        worker = assignment['user_id']
        for task, solution in zip(tasks, solutions):
            inp, out = task['input_values'], solution['output_values']
            key = inp['candidate_id']
            vote = {'worker_id': worker, 'assignment_id': assignment['id'], 'verdict': out['verdict'], 'reason': out.get('reason')}
            if vote['verdict'] not in ('OK', 'BAD', 'UNCERTAIN', 'IMAGE_ERROR'):
                raise ValueError('Unknown review verdict')
            if key in gold:
                control = gold[key]
                if int(inp['photo_id']) != control['photo_id'] or normalized_boxes(inp['boxes']) != normalized_boxes(control['boxes']):
                    raise ValueError('Control input mismatch')
                controls.append(dict(vote, candidate_id=key, correct=out['verdict'] == control['expected_verdict']))
                continue
            if key not in candidates:
                raise ValueError('Unknown candidate')
            original = candidates[key]
            if int(inp['photo_id']) != original['photo_id'] or normalized_boxes(inp['boxes']) != normalized_boxes(original['boxes']):
                raise ValueError('Candidate input mismatch')
            if worker == original['author_id']:
                raise ValueError('Self review')
            if worker not in qualified:
                excluded.append(dict(vote, candidate_id=key, exclusion='Exam not passed'))
                continue
            votes[key].append(vote)
    result = []
    for key, candidate in candidates.items():
        current = votes[key]
        # A separately issued, author-excluded third vote may resolve a dispute.
        # Keep disagreements for manual adjudication even with a majority.
        if len(current) > 3:
            raise ValueError('Unexpected excess votes; inspect reissues')
        decision = review_decision(candidate['author_id'], current)
        result.append(dict(candidate, votes=current, decision=decision))
    control_summary = []
    for worker in sorted({r['worker_id'] for r in controls}):
        current = [r for r in controls if r['worker_id'] == worker]
        wrong = sum(not r['correct'] for r in current)
        control_summary.append({'worker_id': worker, 'controls': len(current), 'wrong': wrong, 'requires_vote_audit': wrong > 0})
    return {'items': result, 'counts': dict(Counter(r['decision'] for r in result)), 'controls': controls, 'control_summary': control_summary, 'excluded': excluded}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('assignments', type=Path)
    parser.add_argument('--lineage', type=Path, default=P / 'review-pilot-lineage.json')
    parser.add_argument('--output', type=Path, default=P / 'review-pilot-aggregation.json')
    args = parser.parse_args()
    passed = {r['worker_id'] for r in json.loads((P / 'review-exam-assessment.json').read_text()) if r['qualified']}
    report = aggregate(json.loads(args.assignments.read_text()), json.loads(args.lineage.read_text()), passed)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'counts': report['counts'], 'controls': report['control_summary'], 'excluded': len(report['excluded'])}, ensure_ascii=False))
