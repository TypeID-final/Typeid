import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), "instance", "biometric_app.db")
conn = sqlite3.connect(db_path)
conn.execute("UPDATE login_session SET login_method = 'biometrics' WHERE login_method = 'ml_typing_app'")
conn.commit()
conn.close()
print("Updated successfully")
