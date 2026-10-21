"""The 16-colour palette of grid games: cell value (0-15) -> RGB, the official ARC-AGI-3 one
(mirrors ``arc_agi.rendering.COLOR_MAP``)."""

ARC_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 255, 255),  # 0  white
    (204, 204, 204),  # 1  off-white
    (153, 153, 153),  # 2  neutral light
    (102, 102, 102),  # 3  neutral
    (51, 51, 51),  # 4  off black
    (0, 0, 0),  # 5  black
    (229, 58, 163),  # 6  magenta
    (255, 123, 204),  # 7  magenta light
    (249, 60, 49),  # 8  red
    (30, 147, 255),  # 9  blue
    (136, 216, 241),  # 10 blue light
    (255, 220, 0),  # 11 yellow
    (255, 133, 27),  # 12 orange
    (146, 18, 49),  # 13 maroon
    (79, 204, 48),  # 14 green
    (163, 86, 214),  # 15 purple
)
