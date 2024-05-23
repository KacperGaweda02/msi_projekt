import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# walidacja
def get_input(prompt, type_func, condition_func):
    while True:
        try:
            value = type_func(input(prompt))
            if condition_func(value):
                return value
            else:
                print("Wartość nie spełnia warunków. Spróbuj ponownie.")
        except ValueError:
            print("Nieprawidłowy typ danych. Spróbuj ponownie.")


# uczenie i podanie accuracy
data = pd.read_excel('Houses.xlsx')
X = data[['rooms', 'sq', 'year', 'price', 'floor']].values
y = data['city'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=44)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_classifier.fit(X_train_scaled, y_train)
y_prediction = knn_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_prediction)
print("Accuracy:", accuracy)

# "front"
rooms = get_input("Podaj ilość pokoi: ", int, lambda x: x > 0)
sq = get_input("Podaj powierzchnię [m^2]: ", float, lambda x: x > 0)
year = get_input("Podaj rok budowy: ", int, lambda x: 2024 >= x >= 1500)
price = get_input("Podaj cenę [zł]: ", float, lambda x: x > 0)
floor = get_input("Podaj piętro [0 - parter]: ", int, lambda x: x >= 0)
user_data = np.array([[rooms, sq, year, price, floor]])
user_data_scaled = scaler.transform(user_data)
prediction = knn_classifier.predict(user_data_scaled)
print("Przewidywane miasto:", prediction[0])
