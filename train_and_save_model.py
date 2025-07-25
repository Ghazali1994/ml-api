import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Load and train
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

model = Ridge()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load model back
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Predict to confirm it works
print(loaded_model.predict(X_test[:1]))
