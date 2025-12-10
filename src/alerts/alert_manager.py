import datetime

ALERT_LOG = "alerts/aqi_alerts.log"

def check_aqi_alert(aqi_value: float):
    if aqi_value >= 200:
        return "HAZARDOUS"
    elif aqi_value >= 150:
        return "UNHEALTHY"
    elif aqi_value >= 100:
        return "MODERATE"
    else:
        return "GOOD"


def log_alert(aqi_value: float, level: str):
    timestamp = datetime.datetime.now().isoformat()

    message = f"{timestamp} | AQI={aqi_value:.2f} | LEVEL={level}\n"

    with open(ALERT_LOG, "a") as f:
        f.write(message)
