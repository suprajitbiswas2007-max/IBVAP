from deepface import DeepFace
import mysql.connector
import json

conn = mysql.connector.connect(
    host="localhost", user="root", password="idkanymore", database="ibvap"
)
cursor = conn.cursor()

people = {
    "Suprajit": [
        r"C:\Programming\SIH\webcam_reg_1.jpg",
        r"C:\Programming\SIH\webcam_reg_2.jpg",
        r"C:\Programming\SIH\webcam_reg_3.jpg",
        r"C:\Programming\SIH\webcam_reg_4.jpg",
        r"C:\Programming\SIH\webcam_reg_5.jpg",
        r"C:\Programming\SIH\webcam_reg_6.jpg",
        r"C:\Programming\SIH\webcam_reg_7.jpg"

    ]
}

for name, paths in people.items():     
    for path in paths:                    
        try:
            embedding = DeepFace.represent(
                img_path=path, model_name="ArcFace", detector_backend="opencv"
            )[0]["embedding"]
            encoding_str = json.dumps(embedding)

            cursor.execute(
                "INSERT INTO faces (name, encoding) VALUES (%s, %s)",
                (name, encoding_str)
            )
            print(f"Registered {name} from {path}")
        except Exception as e:
            print(f"FAILED for {name} ({path}): {e}")

conn.commit()
cursor.close()
conn.close()
print("Done")