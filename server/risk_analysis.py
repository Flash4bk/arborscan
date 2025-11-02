import requests
import math

# === Твои ключи ===
OPENWEATHER_KEY = "dc825ffd002731568ec7766eafb54bc9"

# === Получение данных погоды ===
def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
    data = requests.get(url).json()
    wind_speed = data["wind"]["speed"]
    gust = data["wind"].get("gust", wind_speed)
    temp = data["main"]["temp"]
    print(f"🌬️ Ветер: {wind_speed} м/с, порывы: {gust} м/с, температура: {temp}°C")
    return wind_speed, gust, temp

# === Получение данных почвы ===
def get_soil(lat, lon):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&depth=0-30cm&value=mean"
    data = requests.get(url).json()
    
    # универсальный парсер
    if "properties" in data:  # старый формат (props)
        props = data["properties"]
        getv = lambda key: props.get(key, {"mean": [0]})["mean"][0]
        clay = getv("clay")
        sand = getv("sand")
        silt = getv("silt")
        bulk_density = getv("bdod") / 100  # г/см³
        organic_carbon = getv("ocd") / 10  # %
    elif "layers" in data:  # новый формат (через layers)
        layers = {l["name"]: l for l in data["layers"]}
        def safe(layer, prop="mean"):
            try:
                return layers[layer]["depths"][0]["values"][prop]
            except Exception:
                return 0
        clay = safe("clay")
        sand = safe("sand")
        silt = safe("silt")
        bulk_density = safe("bdod") / 100
        organic_carbon = safe("ocd") / 10
    else:
        raise ValueError("Неожиданный формат ответа SoilGrids")

    print(f"🌱 Почва: глина={clay:.1f}%, песок={sand:.1f}%, плотность={bulk_density:.2f} г/см³, орг.углерод={organic_carbon:.2f}%")
    return clay, sand, silt, bulk_density, organic_carbon

# === Перевод почвы в коэффициент устойчивости ===
def soil_factor(clay, sand):
    # Из PDF «Влияние почвы на ветроустойчивость деревьев…»
    if clay > 40:
        return 1.05  # тяжёлая почва — высокая устойчивость
    elif sand > 60:
        return 0.85  # рыхлая песчаная — низкая устойчивость
    else:
        return 0.95  # средняя устойчивость

# === Основная функция расчёта риска ===
def compute_risk(species, H, DBH, CL, wind_speed, gust, k_soil):
    # Простейшая модель риска (позже заменим на физическую формулу из PDF)
    S = H / max(DBH, 0.01)  # стройность
    crown_ratio = CL / H
    base = 0.4 * S + 0.3 * crown_ratio + 0.2 * gust + 0.1 * (1/k_soil)
    risk = min(100, base * 5)
    # Корректировка по виду дерева
    species_factor = {
        "Берёза": 1.1, "Дуб": 0.8, "Ель": 1.2, "Сосна": 0.9, "Тополь": 1.3
    }.get(species, 1.0)
    risk *= species_factor
    # Категория
    if risk < 35:
        level = "Низкий"
    elif risk < 70:
        level = "Средний"
    else:
        level = "Высокий"
    return risk, level
