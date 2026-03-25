import sqlite3
import json
from datetime import datetime


class SceneStorage:
    """
    Handles local SQLite storage of scene logs.
    """

    def __init__(self, db_path="scene_log.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS scene_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            clip_top3 TEXT,
            yolo_detections TEXT
        )""")
        self.conn.commit()

    def insert(self, clip_results: dict, yolo_results: list[str]):
        """
        Insert a new scene entry into the database.
        """
        self.cursor.execute(
            "INSERT INTO scene_log (timestamp, clip_top3, yolo_detections) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), json.dumps(
                clip_results["results"]), json.dumps(yolo_results))
        )
        self.conn.commit()

    def query(self, start_time=None, end_time=None):
        """
        Query logs between start_time and end_time.
        """
        q = "SELECT * FROM scene_log"
        params = []
        if start_time and end_time:
            q += " WHERE timestamp BETWEEN ? AND ?"
            params = [start_time, end_time]
        self.cursor.execute(q, params)
        return self.cursor.fetchall()
