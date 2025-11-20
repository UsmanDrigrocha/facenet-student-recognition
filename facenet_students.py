# facenet_students.py
# pip install mtcnn facenet-pytorch opencv-python numpy torch
# create DB : python facenet_students.py --create_db
# run : python facenet_students.py --recognize --threshold 0.7

import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 160  # FaceNet input size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STUDENT_FOLDER = "students"
DB_FILE = "students_db.npz"

# -----------------------------
# 1. Load models
# -----------------------------
mtcnn = MTCNN(image_size=IMG_SIZE, margin=10, device=DEVICE)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# -----------------------------
# 2. Extract embeddings
# -----------------------------
def create_student_db(folder):
    embeddings_dict = {}
    for student_id in os.listdir(folder):
        pfolder = os.path.join(folder, student_id)
        if os.path.isdir(pfolder):
            embeddings_dict[student_id] = []
            for img_file in os.listdir(pfolder):
                img_path = os.path.join(pfolder, img_file)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face = mtcnn(img_rgb)
                if face is not None:
                    face = face.unsqueeze(0).to(DEVICE)
                    emb = facenet(face)
                    emb = emb.detach().cpu().numpy()
                    embeddings_dict[student_id].append(emb[0])
            if embeddings_dict[student_id]:
                embeddings_dict[student_id] = np.array(embeddings_dict[student_id])
    # Save DB
    np.savez(DB_FILE, **embeddings_dict)
    print(f"Saved student DB to {DB_FILE}")

# -----------------------------
# 3. Recognize from webcam
# -----------------------------
# -----------------------------

def recognize_webcam(threshold=0.7):
    db = np.load(DB_FILE, allow_pickle=True)
    names = list(db.keys())
    embeddings = [db[n] for n in names]
    centroids = [np.mean(e / np.linalg.norm(e, axis=1, keepdims=True), axis=0) for e in embeddings]
    centroids = np.array([c / np.linalg.norm(c) for c in centroids])

    cap = cv2.VideoCapture(0)
    print("Starting webcam. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using mtcnn.detect()
        boxes, _ = mtcnn.detect(img_rgb)

        faces_tensors = []
        valid_boxes = []

        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                face_img = img_rgb[y1:y2, x1:x2]

                # Skip tiny or invalid crops
                if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                    continue

                # Get face tensor
                face_tensor = mtcnn(face_img)
                if face_tensor is not None:
                    if face_tensor.ndim == 3:  # single face
                        face_tensor = face_tensor.unsqueeze(0)
                    faces_tensors.append(face_tensor)
                    valid_boxes.append((x1, y1, x2, y2))

        if faces_tensors:
            faces_batch = torch.cat(faces_tensors).to(DEVICE)
            embeddings_batch = facenet(faces_batch)
            embeddings_batch = embeddings_batch.detach().cpu().numpy()

            for i, emb in enumerate(embeddings_batch):
                emb = emb / np.linalg.norm(emb)
                sims = centroids @ emb
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]
                if best_score > threshold:
                    name = names[best_idx]
                    color = (0, 255, 0)
                else:
                    name = "Unknown"
                    color = (0, 0, 255)

                # Draw rectangle using valid box
                x1, y1, x2, y2 = valid_boxes[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {best_score:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# 4. CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--create_db', action='store_true')
    parser.add_argument('--recognize', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.7)
    args = parser.parse_args()

    if args.create_db:
        create_student_db(STUDENT_FOLDER)
    if args.recognize:
        recognize_webcam(threshold=args.threshold)
