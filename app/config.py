config = {
    'left_foot_flip_x': 1.0,
    'left_foot_flip_y': 1.0,
    'left_foot_flip_z': 1.0,
    'right_foot_flip_x': 1.0,
    'right_foot_flip_y': 1.0,
    'right_foot_flip_z': 1.0,
}


def set_value(d, path, value):
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
