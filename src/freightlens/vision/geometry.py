"""Pixel geometry independent from detector/tracker implementations."""


def side(point, line) -> float:
    (x1, y1), (x2, y2) = line
    return float((point[0] - x1) * (y2 - y1) - (point[1] - y1) * (x2 - x1))


def crosses_segment(previous, current, line, direction: int = 0) -> bool:
    before, after = side(previous, line), side(current, line)
    if before * after >= 0:
        return False
    if direction and (1 if after > before else -1) != direction:
        return False
    fraction = -before / (after - before)
    intersection = (
        previous[0] + fraction * (current[0] - previous[0]),
        previous[1] + fraction * (current[1] - previous[1]),
    )
    (x1, y1), (x2, y2) = line
    return min(x1, x2) - 1e-6 <= intersection[0] <= max(x1, x2) + 1e-6 and (
        min(y1, y2) - 1e-6 <= intersection[1] <= max(y1, y2) + 1e-6
    )


def point_in_polygon(point, polygon) -> bool:
    if len(polygon) < 3:
        raise ValueError("Zone needs at least three points")
    x, y = point
    inside = False
    for i in range(len(polygon)):
        x1, y1 = polygon[i - 1]
        x2, y2 = polygon[i]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside
