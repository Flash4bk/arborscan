import requests
import random


def get_weather(lat, lon):
    """Имитация API погоды (в Railway может быть ограничен доступ)."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        r = requests.get(url, timeout=5)
        data = r.json()
        cw = data.get("current_weather", {})
        return {
            "wind_speed": cw.get("windspeed", 0),
            "wind_gusts": cw.get("windgusts", 0),
            "temperature": cw.get("temperature", 0)
        }
    except Exception as e:
        print(f"⚠️ Ошибка get_weather: {e}")
        return None


def get_soil(lat, lon):
    """Имитация данных о почве (soilgrids.org часто недоступен на Railway)."""
    try:
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}"
        r = requests.get(url, timeout=8)
        data = r.json()
        if "properties" not in data:
            raise ValueError("Неверный ответ SoilGrids.")
        return {
            "clay": random.uniform(0, 40),
            "sand": random.uniform(0, 70),
            "density": random.uniform(1.0, 1.6),
            "organic_carbon": random.uniform(0.5, 3.0)
        }
    except Exception as e:
        print(f"⚠️ Ошибка get_soil: {e}")
        return None


def compute_risk(tree_data, weather, soil):
    """Простейшая логика расчёта риска."""
    try:
        height = tree_data.get("height", 0)
        diameter = tree_data.get("diameter", 0)
        wind = weather["wind_speed"] if weather else 0
        clay = soil["clay"] if soil else 0

        score = (height * 2 + wind * 5 - clay) / 10.0
        score = max(0, min(100, score))

        if score < 35:
            level = "Низкий"
        elif score < 65:
            level = "Средний"
        else:
            level = "Высокий"

        return level, round(score, 1)
    except Exception as e:
        print(f"⚠️ Ошибка compute_risk: {e}")
        return "Не рассчитано", 0.0
