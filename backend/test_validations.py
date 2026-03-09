import requests

BASE_URL = "http://127.0.0.1:5001"

def test_register():
    print("Testing /api/register...")
    payloads = [
        {"desc": "Valid", "data": {"name": "testuser1", "email": "test@test.com", "password": "Password1!", "keystroke_features": [{"time": 100}]}, "expect": 201},
        {"desc": "Invalid Username", "data": {"name": "a", "email": "test@test.com", "password": "Password1!", "keystroke_features": [{"time": 100}]}, "expect": 400},
        {"desc": "Invalid Email", "data": {"name": "testuser", "email": "test.com", "password": "Password1!", "keystroke_features": [{"time": 100}]}, "expect": 400},
        {"desc": "Weak Password", "data": {"name": "testuser", "email": "test@test.com", "password": "weak", "keystroke_features": [{"time": 100}]}, "expect": 400},
    ]

    for p in payloads:
        try:
            res = requests.post(f"{BASE_URL}/api/register", json=p["data"])
            status = res.status_code
            print(f"[{'PASS' if status == p['expect'] else 'FAIL'}] {p['desc']} - Expected: {p['expect']}, Got: {status} (Msg: {res.json().get('message')})")
        except Exception as e:
            print(f"[ERROR] {p['desc']} failed to connect: {e}")

if __name__ == "__main__":
    test_register()

