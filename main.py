import pandas as pd
import matplotlib.pyplot as plt
crop_info = {
    "rice": "Requires high water and warm climate.",
    "maize": "Requires moderate rainfall and fertile soil.",
    "cotton": "Requires warm temperature and black soil.",
    "coconut": "Requires high humidity and rainfall.",
    "papaya": "Requires warm climate and well-drained soil.",
    "banana": "Requires rich soil and regular watering.",
    "mango": "Requires tropical climate and moderate rainfall.",
    "coffee": "Requires cool climate and sufficient rainfall.",
    "apple": "Requires cool temperatures and fertile soil.",
    "orange": "Requires moderate climate and irrigation.",
    "pigeonpeas": "Requires warm climate and moderate rainfall."
}
fertilizer_info = {
    "rice": "Urea",
    "maize": "NPK",
    "coffee": "Organic Compost",
    "mango": "DAP",
    "coconut": "Potassium Fertilizer",
    "papaya": "Organic Fertilizer",
    "pigeonpeas": "DAP",
    "chickpea": "NPK",
    "banana": "Compost",
    "orange": "NPK",
    "blackgram": "DAP"
}
water_info = {
    "banana": "High",
    "rice": "High",
    "coffee": "Medium",
    "mango": "Medium",
    "papaya": "Medium",
    "coconut": "High",
    "orange": "Medium",
    "pigeonpeas": "Low",
    "blackgram": "Low",
    "chickpea": "Low"
}
advice = {
    "rice": "Maintain proper water levels in the field.",
    "banana": "Apply organic compost regularly.",
    "coffee": "Avoid waterlogging and provide shade.",
    "papaya": "Ensure proper drainage and regular watering.",
    "mango": "Prune trees regularly for better growth.",
    "coconut": "Provide sufficient irrigation during summer.",
    "orange": "Use balanced fertilizers and proper drainage.",
    "pigeonpeas": "Avoid excess irrigation.",
    "chickpea": "Grow in well-drained soil.",
    "blackgram": "Maintain moderate soil moisture."
}
# Load dataset
df = pd.read_csv(r"C:\Users\megha\Downloads\Crop_recommendation.csv")
# Input and Output
X = df.drop("label", axis=1)
y = df["label"]
# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")
N = float(input("Enter Nitrogen: "))
P = float(input("Enter Phosphorus: "))
K = float(input("Enter Potassium: "))
temp = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
ph = float(input("Enter pH: "))
rainfall = float(input("Enter Rainfall: "))
input_data = [[N, P, K, temp, humidity, ph, rainfall]]
result = model.predict(input_data)
probabilities = model.predict_proba(input_data)[0]
top3 = probabilities.argsort()[-3:][::-1]
crop = result[0]
confidence = max(probabilities) * 100
print("Recommended Crop:", crop)
print("Prediction Confidence:", round(confidence, 2), "%")
if ph < 6:
    print("Soil Status: Acidic")
elif ph > 7.5:
    print("Soil Status: Alkaline")
else:
    print("Soil Status: Neutral")
if crop in fertilizer_info:
    print("Suggested Fertilizer:", fertilizer_info[crop])
if crop in water_info:
    print("Water Requirement:", water_info[crop])
if rainfall < 100:
    print("Rainfall Status: Low")
elif rainfall < 200:
    print("Rainfall Status: Medium")
else:
    print("Rainfall Status: High")
if confidence >= 80:
    print("Suitability Rating: Excellent")
elif confidence >= 50:
    print("Suitability Rating: Good")
else:
    print("Suitability Rating: Average")
print("\nTop 3 Recommended Crops:")
rank = 1
for i in top3:
    print(f"{rank}. {model.classes_[i]}")
    rank += 1
if crop in crop_info:
    print("Crop Information:", crop_info[crop])
if crop in advice:
    print("Farmer Advice:", advice[crop])
with open("history.txt", "a") as f:
    f.write(f"Crop: {crop}\n")
df["label"].value_counts().plot(kind="bar")
plt.title("Crop Distribution")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.show()
