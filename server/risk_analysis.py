def compute_risk(height_m, diameter_cm, wind_speed, soil=None):
    """Вычисляет риск падения дерева"""

    if height_m <= 0 or diameter_cm <= 0:
        return 0.0, "Низкий"

    # Базовая чувствительность
    ratio = height_m / (diameter_cm / 100)
    risk = ratio * (wind_speed / 10) * 20

    # Коррекция по типу почвы
    if soil:
        if soil.get("clay", 0) > 40:
            risk *= 1.2
        if soil.get("sand", 0) > 60:
            risk *= 1.1

    risk = min(max(risk, 0), 100)

    if risk < 33:
        level = "Низкий"
    elif risk < 66:
        level = "Средний"
    else:
        level = "Высокий"

    return round(risk, 1), level
