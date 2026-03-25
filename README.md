# 🛡️ CamAI: Intelligent Surveillance & Scene Analysis

**CamAI** is a smart surveillance pipeline that blends real-time object detection, deep scene understanding, and natural language interaction. It transforms raw security footage into a searchable, structured database, allowing users to query events as if they were talking to a human guard.

## 🚀 Key Features
- **Dual-Model Inference**: Combines **YOLO** (object detection) and **CLIP** (scene/contextual understanding).
- **Persistent Storage**: Automatically logs detections and scene descriptions into an SQLite database with ISO timestamps.
- **AI Chatbot Interface**: A built-in GPT-powered assistant that parses logs to answer questions like *"When did someone carrying a bag enter?"*
- **Interactive Dashboard**: A Streamlit-based web UI to monitor recent logs and generate comprehensive markdown reports.
- **Optimized for Surveillance**: Intelligent frame skipping (1 FPS processing) to handle long-duration footage without data redundancy.

---

## 📂 Project Structure
```text
CamAI/
├── demo/
│   ├── demo.py            # Main entry point for surveillance video processing
├── app.py                 # Streamlit Dashboard UI
├── main.py                # entry point to run project only in CLI (should be run before app.py)
├── CLIP_model.py          # CLIP inference logic for scene recognition
├── YOLO_model.py          # YOLO inference logic for object detection
├── database.py            # SQLite SceneStorage handler
├── chatbot.py             # GPT-4o-mini integration for natural language querying
├── .env                   # Configuration for API keys
└── report.md              # Auto-generated incident reports
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sanjeevspuranik/CamAI.git
   cd CamAI
   ```

2. **Install dependencies**:
   ```bash
   pip install opencv-python pillow python-dotenv streamlit ultralytics transformers torch
   ```

3. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

---

## 🏃 Usage

### 1. Run Analysis Pipeline
Process a surveillance video file and log data to the database:
```bash
python demo/demo.py
```

Or

Process a surveillance live footage file and log data to the database:
```bash
python CamAI/main.py
```

### 2. Launch Dashboard
View logs and chat with your surveillance data:
```bash
streamlit run CamAI/app.py
```
---

---

## 🎥 Demo & Samples

For evaluation purposes, pre-processed samples and reference footages are available in the `/demo` folder:

* **Sample Video**: Includes `Fighting011_x264.mp4`, a representative surveillance clip used for testing detection accuracy.
* **Generated Reports**: View `report.md` in the root or `/demo` directory to see an example of the AI-generated incident summary.

> [!TIP]
> You can quickly test the Dashboard functionality without running the full pipeline by using the existing `scene_log.db` file provided in the repository.

---

## 📝 Configuration Note: BoT-SORT Re-ID
To ensure the highest accuracy in tracking individuals across occlusions (e.g., when a person walks behind a pillar and reappears), this project uses **BoT-SORT** tracking.

> [!IMPORTANT]
> To enable appearance-based Re-Identification (ReID), modify your tracker configuration (usually `botsort.yaml`) to set:
> ```yaml
> with_reid: True
> ```
> This allows the YOLO model to use visual embeddings to maintain the same ID for an object even if it is temporarily lost or moves significantly.

---

## 📄 Automated Reporting
Upon completion of a video analysis, you can generate a `report.md` via the Streamlit dashboard. This report uses the LLM to summarize all detected events, suspicious activities, and timelines into a professional executive summary.

---
*Developed for intelligent monitoring and security automation.*
