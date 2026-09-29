"""Conservative periodic-overlap tracks; split/merge creates new censored identities."""
import numpy as np


class StructureTracker:
    def __init__(self):
        self.previous = {}
        self.tracks = {}
        self.next_id = 1
        self.rows = []
        self.last_time = None

    def update(self, time, maps):
        for kind in ('hole', 'peak'):
            labels = maps[kind + '_labels']
            old, ids = self.previous.get(kind, (np.zeros_like(labels), {}))
            current = [int(x) for x in np.unique(labels) if x]
            pairs = np.unique(np.stack([old.ravel(), labels.ravel()], axis=1), axis=0)
            parents = {c: set(int(a) for a,b in pairs if b == c and a) for c in current}
            children = {p: set(c for c in current if p in parents[c]) for p in ids}
            new_ids = {}
            for label in current:
                pp = parents[label]
                one = len(pp) == 1 and len(children[next(iter(pp))]) == 1
                if one:
                    track = ids[next(iter(pp))]; event = 'continue'
                else:
                    track = self.next_id; self.next_id += 1
                    event = 'birth' if not pp else ('merge' if len(pp)>1 else 'split')
                    self.tracks[track] = {'track_id': track, 'kind': kind, 'start': float(time),
                                          'left_censored': self.last_time is None or bool(pp)}
                self.tracks[track]['last_seen'] = float(time)
                self.tracks[track]['right_censored'] = True
                new_ids[label] = track
                self.rows.append({'time': float(time), 'kind': kind, 'label': label, 'track_id': track,
                                  'event': event, 'parent_tracks': ';'.join(str(ids[x]) for x in sorted(pp)),
                                  'cells': int(np.count_nonzero(labels == label))})
            continued = set(new_ids.values())
            for parent, track in ids.items():
                if track not in continued:
                    self.tracks[track]['right_censored'] = bool(children[parent])
                    self.tracks[track]['end_reason'] = 'split_or_merge' if children[parent] else 'no_overlap_or_disappearance'
            self.previous[kind] = (labels.copy(), new_ids)
        self.last_time = time

    def summary(self):
        return [{**row, 'observed_duration': row['last_seen']-row['start'],
                 'association': 'pixel overlap only; fast advection can break a track'} for row in self.tracks.values()]
