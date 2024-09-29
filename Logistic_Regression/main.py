from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# Load the Wine dataset
X, y = load_wine(return_X_y=True)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Initialize Logistic Regression model with 'saga' solver and increased max_iter
clf = LogisticRegression(solver='saga', max_iter=1000, random_state=0)

# Train the model using the training data
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model's accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Logistic Regression model accuracy: {acc * 100:.2f}%")