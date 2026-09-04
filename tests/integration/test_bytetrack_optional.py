import numpy as np
import pytest


def test_real_bytetrack_preserves_ids_on_observed_boxes():
    pytest.importorskip("ultralytics")
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    tracker = BYTETracker(IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml"))))
    identities = []
    for frame in range(10):
        x = 30 + frame * 4
        boxes = Boxes(np.array([[x, 20, x + 60, 60, 0.9, 0]], dtype=np.float32), (200, 200))
        tracked = tracker.update(boxes)
        assert len(tracked) == 1
        identities.append(int(tracked[0, 4]))
    assert len(set(identities)) == 1
