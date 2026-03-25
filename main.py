import cv2
import json
from PIL import Image

from CLIP_model import CLIPInference
from YOLO_model import YOLOInference
from database import SceneStorage
from chatbot import ChatBot
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")


def run_pipeline():
    clip_infer = CLIPInference()
    # replace with your YOLOv26 weights
    yolo_infer = YOLOInference("yolo26n.pt")
    storage = SceneStorage()

    cap = cv2.VideoCapture(0)
    # NEW
    # video_path = r"CamAI\videos\Normal_Videos012_x264.mp4"
    # cap = cv2.VideoCapture(video_path)
    texts = [
        # People and objects
        "a person", "a group of people", "a child", "a man", "a woman",
        "a bag", "a backpack", "a chair", "a sofa", "a table", "a laptop",
        "a dog", "a cat", "a bird", "a car", "a bicycle",

        # Indoor scenes
        "a living room", "a kitchen", "a bedroom", "an office", "a classroom",
        "a conference room", "a hallway", "a dining hall", "a library",
        "a shopping mall", "a supermarket", "a hospital ward", "a waiting room",
        "a ceremony indoors", "a wedding hall", "a party scene", "a crowded elevator",

        # Outdoor scenes
        "a street", "a park", "a playground", "a beach", "a mountain",
        "a forest", "a garden", "a marketplace", "a stadium", "a bus stop",
        "a train station", "an airport terminal", "a festival crowd",
        "a concert", "a parade", "a protest", "a ceremony outdoors",
        "a sports event", "a crowded street", "a picnic scene",

        # Activities
        "people walking", "people sitting", "people dancing", "people eating",
        "people shopping", "people celebrating", "people praying",
        "people studying", "people working at computers", "people watching TV",
        "people exercising", "people playing music", "people playing sports"
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        clip_results = clip_infer.infer(image, texts, top_k=3)
        yolo_results = yolo_infer.infer(frame)

        storage.insert(clip_results, yolo_results)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def chatbot_demo():
    storage = SceneStorage()
    chatbot = ChatBot(api_key=api_key)

    logs = storage.query()
    answer = chatbot.ask("Who entered the room today?", logs)
    print(answer)


if __name__ == "__main__":
    run_pipeline()
    chatbot_demo()
