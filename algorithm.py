from itertools import chain

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class Advantage:
    def __init__(self, year: int, sales: int = 0, games: int = 0):
        self.sales = sales
        self.games = games
        self.year = year

class GenreRecord:
    def __init__(self, name: str):
        self.name = name
        self.history = []
        self.sales_model = None
        self.games_model = None
        self.predicted = []

    def get_advantage(self, year: int) -> Advantage:
        if not self.history:
            advantage = Advantage(year)
            self.history.append(advantage)
            return advantage

        first_year: int = self.history[0].year
        index: int = year - first_year

        while index < 0:
            first_year -= 1
            self.history.insert(0, Advantage(first_year))
            index += 1

        while index >= len(self.history):
            next_year = first_year + len(self.history)
            self.history.append(Advantage(next_year))

        return self.history[int(index)]

    def make_model(self):
        years_data = [[x.year] for x in self.history]
        sales_data = [x.sales for x in self.history]
        games_data = [x.games for x in self.history]

        poly_features = PolynomialFeatures(degree=len(self.history) + 1)
        poly_years = poly_features.fit_transform(years_data)

        self.sales_model = LinearRegression()
        self.sales_model.fit(poly_years, sales_data)

        self.games_model = LinearRegression()
        self.games_model.fit(poly_years, games_data)

    def predict(self, up_to: int):
        if self.games_model is None or self.sales_model is None:
            self.make_model()

        self.predicted.clear()
        next_year = self.history[-1].year
        while next_year <= up_to:
            next_year += 1
            prediction = Advantage(next_year)
            prediction.sales = self.sales_model.predict([[next_year]])
            prediction.games = self.games_model.predict([[next_year]])
            self.predicted.append(prediction)

    def predict_advantage(self, year: int) -> Advantage:
        if year > self.predicted[-1].year:
            if self.games_model is None or self.sales_model is None:
                self.make_model()
            result = Advantage(year)
            result.sales = self.sales_model.predict([[year]])
            result.games = self.games_model.predict([[year]])
            return result
        else:
            prediction_index = year - self.history[-1].year
            return self.predicted[prediction_index]

    def plot_data(self):
        years_data = [x.year for x in chain(self.history, self.predicted)]
        sales_data = [x.sales for x in chain(self.history, self.predicted)]
        games_data = [x.games for x in chain(self.history, self.predicted)]
        color = ["r" if x.year > self.history[-1] else "g" for x in chain(self.history, self.predicted)]

        plt.bar([0, 1], [2, 3])
        # plt.bar(years_data, sales_data)
        # plt.bar(years_data, games_data, color=color)
        plt.show()


class MarketData:
    def __init__(self):
        self.genre_records = {}
        self.skipped_entries: [InvalidGameData] = []
        self.total_entries: int = 0

    def add_record(self, data, entry_id):
        try:
            genre = data["Genre"]
            record = self.genre_records.get(genre)
            if record is None:
                record = GenreRecord(genre)
                self.genre_records[genre] = record

            advantage = record.get_advantage(data["Year"])
            advantage.games += 1
            advantage.sales += data["Global_Sales"]
            self.total_entries += 1
        except Exception as err:
            self.skipped_entries.append(InvalidGameData(entry_id, err))

    def create_models(self):
        for genre in self.genre_records.values():
            genre.make_model()

class InvalidGameData(Exception):
    def __init__(self, entry_id, inner: Exception):
        super().__init__()
        self.entry_id = entry_id
        self.inner = inner

    def __str__(self):
        return f"Entry {self.entry_id} caused {self.inner}"

def market_data_from_csv(path: str) -> MarketData:
    data = MarketData()

    csv = pd.read_csv(path)
    for line in csv.iterrows():
        data.add_record(line[1], f"row: {line[0]}")

    tmp = data.genre_records["Sports"]
    tmp.predict(2020)
    tmp.plot_data()
    return data
