python
import pandas as pd
import folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_excel('Traffic_Accident-v3.0_AbuDhabi_2025.xlsx')

# Preprocess the dataset
data['Accident_Date'] = pd.to_datetime(data['Accident_Date'])
data['Month'] = data['Accident_Date'].dt.month
data['Hour'] = data['Accident_Date'].dt.hour

# Feature selection
features = ['Month', 'Hour', 'Weather_Condition', 'Surface_Condition', 'Accident_Type']
X = data[features]
y = data['Accident_Severity']

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['Weather_Condition', 'Surface_Condition', 'Accident_Type'], drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Visualize accident hotspots
map_accidents = folium.Map(location=[24.4539, 54.3773], zoom_start=12)
for _, row in data.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color='red',
        fill=True,
        fill_opacity=0.6
    ).add_to(map_accidents)

map_accidents.save('accident_hotspots_map.html')
