from deepface import DeepFace
import mysql.connector
import json

conn = mysql.connector.connect(
    host="", user="", password="", database="" #ADD YOUR OWN DATABASE
)
cursor = conn.cursor()

people = {                  #ADD THE IMAGE PATHS OF IMAGES TAKEN FROM camertest.py
    "NAME": [
        r"path1.mp4",
        r"path2.mp4
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
