import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import AutoModel, AutoFeatureExtractor
from PIL import Image
from scipy.spatial.distance import cosine
import os

# === 1. Modelni yuklash ===
def load_model():
    model_name = "google/vit-base-patch16-224-in21k"
    print("Model yuklanmoqda...")
    model = AutoModel.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    print("Model muvaffaqiyatli yuklandi.")
    return model, feature_extractor

# === 2. Yuz embeddinglarini olish ===
def get_face_embedding(model, feature_extractor, image):
    try:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = feature_extractor(images=pil_image, return_tensors="pt")
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        return embeddings
    except Exception as e:
        print(f"Xatolik yuz berdi: {e}")
        return None

# === 3. O'xshashlikni hisoblash ===
def calculate_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# === 4. Yuzlar ma'lumotini saqlash va terminalda chiqarish ===
def save_detected_faces_to_excel_and_print(name, similarity, gender, timestamp):
    file_name = "detected_faces.xlsx"
    new_data = {
        "시간": [timestamp],
        "이름": [name],
        "성별": [gender],
        "Similarity": [similarity]
    }
    df_new = pd.DataFrame(new_data)

    # Terminalda ma'lumotlarni chiqarish
    print(df_new)

    try:
        if os.path.exists(file_name):
            with pd.ExcelWriter(file_name, mode="a", if_sheet_exists="overlay", engine="openpyxl") as writer:
                df_new.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
        else:
            df_new.to_excel(file_name, index=False)
    except PermissionError:
        print(f"Xatolik: '{file_name}' fayli boshqa dastur tomonidan foydalanilmoqda. Uni yoping va qayta urinib ko'ring.")

# === 5. Genderni aniqlash ===
def predict_gender(name):
    gender_mapping = {
        "Kim Songmin": "male",
        "Kim Yeongho": "male",
        "Lee Jeak": "male",
        "Umida": "female"
    }
    return gender_mapping.get(name, "Unknown")

# === 6. Kameradan real vaqt rejimida yuzlarni tanib olish ===
def detect_and_compare_faces(model, feature_extractor, reference_embeddings, reference_names):
    cap = cv2.VideoCapture(0)
    print("Kameradan yuzlarni qidirishni boshlash. ESC tugmasini bosing chiqish uchun.")

    logged_names = set()  # Yuzlarni bir marta log qilish uchun to'plam

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera bilan muammo yuz berdi.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            embedding = get_face_embedding(model, feature_extractor, face)
            if embedding is not None:
                similarities = [calculate_similarity(embedding, ref_emb) for ref_emb in reference_embeddings]
                max_similarity = max(similarities)
                best_match_index = similarities.index(max_similarity)
                name = reference_names[best_match_index] if max_similarity > 0.70 else "Unknown"

                # Genderni aniqlash
                gender = predict_gender(name)

                # Yuz ramkasi va ism
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{name}: {max_similarity:.2f}, {gender}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Ismni bir marta log qilish
                if name not in logged_names and name != "Unknown":
                    print(f"Detected: {name} ({gender})")
                    logged_names.add(name)

                    # Yuz ma'lumotlarini Excel faylga saqlash va terminalda chiqarish
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_detected_faces_to_excel_and_print(name, f"{max_similarity:.2f}", gender, timestamp)

        # Tasvirni ko'rsatish
        cv2.imshow("Real-Time Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC tugmasi bilan chiqish
            break

    cap.release()
    cv2.destroyAllWindows()

# === 7. Asosiy kodni ishga tushirish ===
if __name__ == "__main__":
    model, feature_extractor = load_model()

    # Referens tasvirlar va ismlar
    reference_images = [
        r"C:\Users\13\Desktop\face_recog\Kim Songmin.jpg",
        r"C:\Users\13\Desktop\face_recog\Kim Yeongho.jpg",
        r"C:\Users\13\Desktop\face_recog\Lee Jeak.jpg",
        r"C:\Users\13\Desktop\face_recog\Umida.jpg"
    ]
    reference_names = ["Kim Songmin", "Kim Yeongho", "Lee Jeak", "Umida"]

    # Referens embeddinglarini olish
    reference_embeddings = []
    for image_path, name in zip(reference_images, reference_names):
        image = cv2.imread(image_path)
        if image is not None:
            embedding = get_face_embedding(model, feature_extractor, image)
            if embedding is not None:
                reference_embeddings.append(embedding)
            else:
                print(f"{name} uchun embeddinglarni olishda muammo yuz berdi.")
        else:
            print(f"Tasvir ochib bo'lmadi: {image_path}")

    if reference_embeddings:
        detect_and_compare_faces(model, feature_extractor, reference_embeddings, reference_names)
    else:
        print("Referens embeddinglar topilmadi.")
