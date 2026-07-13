from inference.predictor import Predictor

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor("weights/best_model.pth")
    return _predictor


def predict_digit(cell):
    digit, _, _ = _get_predictor().predict(cell)
    return digit
