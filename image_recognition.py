from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import os


subscription_key = "b92b5b90c94046eead521f1787224a7a"
endpoint = "https://computer-vision-image-recognition.cognitiveservices.azure.com/"


client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))


def analyze_image(image_path):
    with open(image_path, "rb") as image_file:
        results = client.analyze_image_in_stream(
            image_file, visual_features=[VisualFeatureTypes.objects]
        )

    return results.objects


if __name__ == "__main__":
    image_path = (
        "/Users/jamiezzz/Desktop/cloud_services/image-recognition/img/dandelion.png"
    )

    objects = analyze_image(image_path)

    for obj in objects:
        print(f"Object: {obj.object_property} (Confidence: {obj.confidence:.2f})")
