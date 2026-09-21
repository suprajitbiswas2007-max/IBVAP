import mysql.connector
conn = mysql.connector.connect(
    host="", user="", password="", database=""   #ADD YOUR OWN DATABASE
)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS platess (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plate_number VARCHAR(50) UNIQUE,
    plate_owner VARCHAR(100),
    vehicle_type VARCHAR(50)
)
""")

plates = {                           #ADD YOUR OWN PLATES
    "NAME": "WB06B1428",
    "NAME1": "RJ27BE7084
}

for owner_name, plate_number in plates.items():
    clean_number = "".join(c for c in plate_number if c.isalnum()).upper()
    try:
        cursor.execute(
            """
            INSERT INTO platess (plate_number, plate_owner, vehicle_type)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE plate_owner = VALUES(plate_owner), vehicle_type = VALUES(vehicle_type)
            """,
            (clean_number, owner_name, "Civil")
        )
        print(f"Registered plate '{clean_number}' for {owner_name}")

    except Exception as e:
        print(f"FAILED for {owner_name} ({clean_number}): {e}")

conn.commit()
cursor.close()
conn.close()
print("Done")
