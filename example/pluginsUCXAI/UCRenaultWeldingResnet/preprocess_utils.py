def make_square_bbox_with_margin(xyxy, img_w, img_h, margin=100):
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1

    # Desired square side with margin
    s = max(w, h) + 2 * margin
    half_s = s / 2

    # Initial center of bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    # Shift center to keep full square inside image if possible
    cx = min(max(cx, half_s), img_w - half_s)
    cy = min(max(cy, half_s), img_h - half_s)

    # Check if full square fits after center adjustment
    fits_width = (cx - half_s) >= 0 and (cx + half_s) <= img_w
    fits_height = (cy - half_s) >= 0 and (cy + half_s) <= img_h

    if fits_width and fits_height:
        # Full square fits, return it
        cx, cy = int(round(cx)), int(round(cy))
        r = int(half_s)
        return cx - r, cy - r, cx + r, cy + r

    # Otherwise, adjust width and height independently (loose square)
    max_half_w = min(cx, img_w - cx)
    max_half_h = min(cy, img_h - cy)

    half_w = min(half_s, max_half_w)
    half_h = min(half_s, max_half_h)

    cx, cy = int(round(cx)), int(round(cy))
    x1_sq, x2_sq = int(cx - half_w), int(cx + half_w)
    y1_sq, y2_sq = int(cy - half_h), int(cy + half_h)

    return x1_sq, y1_sq, x2_sq, y2_sq