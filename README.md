# msi_projekt
Projekt na metody sztucznej inteligencji

Program napisany w pythonie korzysta z knn jako modelu, pobiera dane z pliku Houses.xlsx (2000 rekordów), dane pochodzą z poniższej strony:
https://www.kaggle.com/datasets/dawidcegielski/house-prices-in-poland?resource=download
optymalne accuracy na poziomie nieco ponad 75% znalazłem dla poniższych parametrów:
- test_size=0.21,
- random_state=44,
- n_neighbors=1.

Program na podstawie podanych parametrów domu (piętro, powierzchnia, rok budowy, cena i ilość pokoi) odpowiada czy posiadłość znajduje się w Krakowie, Warszawie czy Poznaniu.

Program powstał jedynie w celach naukowych.
