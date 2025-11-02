import requests
import math
import os

# === Получение погоды ===
def get_weather(lat, lon):
    try:
        api_key = os.getenv("OPENWEATHER_KEY", "dc825ffd002731568ec7766eafb54bc9")
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=10)
        data = r.json()
        wind = data.get("wind", {})
        return wind.get("speed", 0), wind.get("gust", 0), data.get("main", {}).get("temp", 0)
    except Exception as e:
        print(f"⚠️ Ошибка погоды: {e}")
        return 0, 0, 0

# === Получение данных о почве ===
def get_soil(lat, lon):
    try:
        url = f"https://rest.soilgrids.org/query?lon={lon}&lat={lat}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            print(f"⚠️ Ошибка Soil API: {resp.status_code}")
            return 0, 0, 0, 0, 0
        try:
            data = resp.json()
        except Exception:
            print("⚠️ Soil API вернул не-JSON ответ.")
            return 0, 0, 0, 0, 0

        props = data.get("properties", {}).get("layers", {})
        clay = props.get("clay", {}).get("mean", [0])[0]
        sand = props.get("sand", {}).get("mean", [0])[0]
        silt = props.get("silt", {}).get("mean", [0])[0]
        bd = props.get("bdod", {}).get("mean", [0])[0]
        oc = props.get("ocd", {}).get("mean", [0])[0]
        return clay, sand, silt, bd, oc
    except Exception as e:
        print(f"❌ Ошибка get_soil: {e}")
        return 0, 0, 0, 0, 0

# === Факторы почвы ===
def soil_factor(clay, sand):
    return round(max(0.5, 1 - abs(clay - sand) / 200), 2)

# === Расчёт риска ===
def compute_risk(species, height, dbh, crown, wind, gust, k_soil):
    try:
        index = (height * (wind + gust) * (1 / max(k_soil, 0.1))) / (dbh + 0.1)
        score = min(100, max(0, index))
        level = "Низкий" if score < 40 else "Средний" if score < 70 else "Высокий"
        return score, level
    except Exception as e:
        print(f"⚠️ Ошибка compute_risk: {e}")
        return 0, "Неопределён"
