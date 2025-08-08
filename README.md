# 🎬 TED Talk Clip Extractor

AI-powered video analysis tool that extracts the best speaker-focused moments from TED talks and educational videos. Features a beautiful modern web interface with real-time processing and smart filtering.

## ✨ Features

- **🎥 Smart Video Processing**: Automatically identifies and extracts 30-second clips where only the speaker is present
- **🧠 AI-Powered Analysis**: Uses advanced computer vision (HOG, YOLOv8) to detect people and filter out slides
- **🎨 Modern Web Interface**: Beautiful, responsive design inspired by modern web applications
- **📱 Multi-Input Support**: Upload local videos or paste YouTube links
- **⚡ Real-time Progress**: Live progress tracking with Server-Sent Events
- **🔍 Smart Filtering**: Avoids presentation slides and audience shots
- **📦 Easy Download**: Direct download and preview options for generated clips

## 🚀 Live Demo

Deploy to Railway: [![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/khan1267s/tedtalk_new_version)

## 🛠️ Local Development

### Prerequisites

- Python 3.11+
- FFmpeg (for video processing)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/khan1267s/tedtalk_new_version.git
   cd tedtalk_new_version
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python web_app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:7864`

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t ted-clip-extractor .

# Run the container
docker run -p 8080:8080 ted-clip-extractor
```

## 🚂 Railway Deployment

1. **Fork this repository**
2. **Connect to Railway**
   - Visit [Railway](https://railway.app)
   - Connect your GitHub account
   - Select this repository
3. **Deploy automatically**
   - Railway will detect the `Dockerfile` and `railway.json`
   - Deployment starts automatically

## 📖 How It Works

### Video Processing Pipeline

1. **Download/Upload**: Accepts YouTube URLs or local video files
2. **Audio Analysis**: Extracts audio and performs Voice Activity Detection (VAD)
3. **Visual Analysis**: 
   - Detects people using HOG or YOLOv8
   - Identifies presentation slides using image analysis
   - Filters out audience shots
4. **Clip Generation**: Selects the best 30-second segments with high speech activity
5. **Output**: Generates downloadable MP4 clips

### Technology Stack

- **Backend**: Flask (Python)
- **Video Processing**: FFmpeg, OpenCV
- **AI/ML**: YOLOv8 (Ultralytics), HOG Detection
- **Audio**: WebRTC VAD, NumPy
- **Frontend**: HTML5, CSS3, JavaScript
- **Deployment**: Docker, Railway

## 🎯 Usage

### Web Interface

1. **Input**: Paste a YouTube URL or upload a video file
2. **Configure**: Set number of clips (3-10) and duration (20-45 seconds)
3. **Options**: 
   - ✅ Strict person-only filtering
   - ✅ Avoid presentation slides
4. **Process**: Click "🚀 Create Clips" and watch real-time progress
5. **Download**: Preview and download generated clips

### Command Line

```bash
python tool.py "https://www.youtube.com/watch?v=VIDEO_ID" \\
  --output-dir ./output \\
  --num-clips 5 \\
  --clip-duration 30 \\
  --strict-person-only \\
  --avoid-slides
```

## 🔧 Configuration

### Environment Variables

- `PORT`: Server port (default: 7864, Railway sets automatically)
- `PYTHONUNBUFFERED`: Set to 1 for proper logging

### Processing Options

- **Number of clips**: 3, 5, or 10 clips
- **Clip duration**: 20, 30, or 45 seconds
- **Detection method**: HOG (fast) or YOLOv8 (accurate)
- **Strict filtering**: Ensures only single-person scenes
- **Slide detection**: Filters out presentation slides

## 📁 Project Structure

```
├── web_app.py              # Flask web application
├── tool.py                 # Core video processing logic
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── railway.json           # Railway deployment config
├── README.md              # This file
└── out/                   # Generated clips output
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube video downloading
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [OpenCV](https://opencv.org/) for computer vision
- [Railway](https://railway.app/) for easy deployment

## 🐛 Issues & Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/khan1267s/tedtalk_new_version/issues) page
2. Create a new issue with detailed information
3. Include error logs and steps to reproduce

---

**Made with ❤️ for better video content creation**