# Face Recognition Attendance System

This project implements a **real-time student attendance system** using **FaceNet** for face recognition. The system can detect multiple faces from a webcam feed and recognize students based on a pre-built database of their images. The FaceNet model is **pre-trained on VGGFace2** and fine-tuned on your student dataset for higher accuracy.  

## Requirements

- Python 3.10+
- PyTorch
- facenet-pytorch
- MTCNN
- OpenCV
- NumPy

All dependencies are listed in `requirements.txt`.

---

## Folder Structure for Student Images

```

students/
├── <Name_or_Roll#>/
│   ├── 01.jpg
│   ├── 02.jpg
│   └── 03.jpg
├── <Name_or_Roll#>/
│   └── ...

````

> **Tip:** Provide at least **3–5 clear frontal images per student** to get good recognition results. More images improve accuracy.

---

## Setup Instructions

1. **Create a virtual environment**
```bash
py -3.10 -m venv venv
````

2. **Activate the environment**

```bash
# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

3. **Install dependencies**

```bash
python -m pip install -r requirements.txt
```

4. **Create the student database**

```bash
python facenet_students.py --create_db
```

This will process all student images in `students/`, extract embeddings using FaceNet, and save them to `students_db.npz`.

5. **Run real-time recognition**

```bash
python facenet_students.py --recognize --threshold 0.7
```

* The `--threshold` parameter sets the cosine similarity threshold for recognition.
* Detected faces above this threshold will be recognized; otherwise, they appear as "Unknown".