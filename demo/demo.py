from dotenv import load_dotenv
from PIL import Image
import sys
import cv2
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CLIP_model import CLIPInference
from YOLO_model import YOLOInference
from database import SceneStorage
from chatbot import ChatBot

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")


def run_pipeline(video_source):
    clip_infer = CLIPInference()
    yolo_infer = YOLOInference("yolo26n.pt")
    storage = SceneStorage()

    cap = cv2.VideoCapture(video_source)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps)
    frame_count = 0

    texts = [
        # People and objects
        "a person", "a group of people", "a child", "a man", "a woman",
        "a bag", "a backpack", "a chair", "a sofa", "a table", "a laptop",
        "a dog", "a cat", "a bird", "a car", "a bicycle", "a weapon", "a suspicious package",
        "a knife", "a gun", "a mask", "a hoodie", "sunglasses",

        # Indoor scenes
        "a living room", "a kitchen", "a bedroom", "an office", "a classroom",
        "a conference room", "a hallway", "a dining hall", "a library",
        "a shopping mall", "a supermarket", "a hospital ward", "a waiting room",
        "a ceremony indoors", "a wedding hall", "a party scene", "a crowded elevator", "robbery in progress",

        # Outdoor scenes
        "a street", "a park", "a playground", "a beach", "a mountain",
        "a forest", "a garden", "a marketplace", "a stadium", "a bus stop",
        "a train station", "an airport terminal", "a festival crowd",
        "a concert", "a parade", "a protest", "a ceremony outdoors",
        "a sports event", "a crowded street", "a picnic scene", "fighting in public",
        "vandalism in progress", "looting in progress", "an accident scene", "a fire scene",
        "a medical emergency", "a suspicious package", "a person running away", "a person hiding", "a person carrying a weapon",
        "a person carrying a bag", "a person wearing a mask", "a person wearing a hoodie", "a person wearing sunglasses", "people shopping", "people celebrating", "people praying",
        "people studying", "people working at computers", "people watching TV", "people exercising", "people playing music", "people playing sports",


        # Activities
        "people walking", "people sitting", "people dancing", "people eating",
        "people shopping", "people celebrating", "people praying",
        "people studying", "people working at computers", "people watching TV",
        "people exercising", "people playing music", "people playing sports"
        "people fighting", "people looting", "people vandalizing", "people running away",
        "people hiding", "people carrying weapons", "people carrying bags",
        "people wearing masks", "people wearing hoodies", "people wearing sunglasses"
    ]
    print(f"Starting analysis on: {video_source}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Get the current timestamp in the video (seconds)
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Run models
            clip_results = clip_infer.infer(image, texts, top_k=3)
            yolo_results = yolo_infer.infer(frame)

            # Modified insert to include timestamp if your SceneStorage supports it
            storage.insert(clip_results, yolo_results)

        # Optional visual feedback
        cv2.imshow("Surveillance Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Analysis Complete.")


if __name__ == "__main__":
    # Point this to your surveillance file
    SURVEILLANCE_FILE = "Fighting011_x264.mp4"

    if os.path.exists(SURVEILLANCE_FILE):
        run_pipeline(SURVEILLANCE_FILE)

        storage = SceneStorage()
        chatbot = ChatBot(api_key=api_key)
        
        # add any query related to video logs here
        logs = storage.query()
        print(chatbot.ask("At what time did a person carrying a bag appear?", logs))
    else:
        print("Video file not found. Check the path.")
