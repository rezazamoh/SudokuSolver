import cv2

from inference.predictor import Predictor

predictor = Predictor(
    "weights/best_model.pth"
)

image = cv2.imread(
    "test_images/test_predict.png",
    cv2.IMREAD_GRAYSCALE
)

digit, confidence, probabilities = predictor.predict(image)

print("Prediction :", digit)
print(f"Confidence : {confidence*100:.2f}%")
print(probabilities)