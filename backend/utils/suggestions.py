# Provide health advice based on severity score

def get_suggestions(score):
    if score < 5:
        return "Mild condition. Rest and hydration."
    elif score < 10:
        return "Moderate condition. Consult a doctor."
    elif score < 15:
        return "High severity. Medical advice recommended."
    else:
        return "Critical condition. Seek immediate care."
