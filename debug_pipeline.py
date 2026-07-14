import os
import cv2
import numpy as np

os.makedirs("output/pipeline_debug", exist_ok=True)

debug_counter = 0

def save_pipeline(raw_cell, digit, processed, pred, conf):

    global debug_counter

    raw = raw_cell.copy()

    if len(raw.shape) == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    if digit is None:
        digit = np.ones((32,32), np.uint8) * 255

    digit = cv2.resize(digit, (96,96))

    processed = cv2.resize(processed, (96,96))

    if len(digit.shape) == 2:
        digit = cv2.cvtColor(digit, cv2.COLOR_GRAY2BGR)

    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    raw = cv2.resize(raw, (96,96))

    panel = np.hstack([
        raw,
        digit,
        processed
    ])

    footer = np.ones((30, panel.shape[1], 3), dtype=np.uint8) * 255

    cv2.putText(
        footer,
        f"Prediction: {pred}    Confidence: {conf:.3f}",
        (5,20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0,0,255),
        1,
        cv2.LINE_AA
    )

    panel = np.vstack([panel, footer])

    cv2.imwrite(
        f"output/pipeline_debug/{debug_counter:04d}.png",
        panel
    )

    debug_counter += 1