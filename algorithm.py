from itertools import chain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")

class Advantage:
    def __init__(self, year: int, sales: int = 0, games: float = 0):
        self.sales = sales
        self.games = games
        self.year = year

    def ratio(self):
        return self.sales / self.games

class GenreRecord:
    def __init__(self, name: str):
        self.name = name
        self.history = []
        self.sales_model = None
        self.games_model = None
        self.predicted = []
        self.degree_divisor = 15

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
        years_data = [x.year for x in self.history]
        sales_data = [x.sales for x in self.history]
        games_data = [x.games for x in self.history]

        self.sales_model = np.poly1d(np.polyfit(years_data, sales_data, int(len(years_data) / self.degree_divisor)))
        self.games_model = np.poly1d(np.polyfit(years_data, games_data, int(len(years_data) / self.degree_divisor)))

    def direct_predict(self, year: int):
        if self.games_model is None or self.sales_model is None:
            self.make_model()
        prediction = Advantage(year)
        prediction.sales = max(self.sales_model(year), 0)
        prediction.games = max(self.games_model(year), 0)
        return prediction

    def predict(self, up_to: int):
        self.predicted.clear()
        next_year = self.history[-1].year
        while next_year <= up_to:
            next_year += 1
            self.predicted.append(self.direct_predict(next_year))

    def predict_advantage(self, year: int) -> Advantage:
        if year > self.predicted[-1].year:
            return self.direct_predict(year)
        else:
            prediction_index = year - self.history[-1].year
            return self.predicted[prediction_index]

    def plot_sales(self):
        years_total = [x.year for x in chain(self.history, self.predicted)]
        sales_total = [x.sales for x in chain(self.history, self.predicted)]
        years_history = [x.year for x in self.history]
        sales_history = [x.sales for x in self.history]

        plt.plot(years_total, sales_total, color="y")
        plt.scatter(years_history, sales_history, color="b")
        plt.title(f"{self.name}'s Sales")
        plt.show()

    def plot_games(self):
        years_total = [x.year for x in chain(self.history, self.predicted)]
        games_total = [x.games for x in chain(self.history, self.predicted)]
        years_history = [x.year for x in self.history]
        games_history = [x.games for x in self.history]

        plt.plot(years_total, games_total, color="y")
        plt.scatter(years_history, games_history, color="b")
        plt.title(f"{self.name}'s Games")
        plt.show()

    def plot_advantage(self):
        years_total = [x.year for x in chain(self.history, self.predicted) if x.games > 0]
        advantage_total = [x.ratio() for x in chain(self.history, self.predicted) if x.games > 0]
        years_history = [x.year for x in self.history if x.games > 0]
        advantage_history = [x.ratio() for x in self.history if x.games > 0]

        plt.plot(years_total, advantage_total, color="y")
        plt.scatter(years_history, advantage_history, color="b")
        plt.title(f"{self.name}'s Advantage")
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

    def predict(self, up_to: int, custom_degree_divisor: str | None = None):
        if custom_degree_divisor:
            custom_degree_divisor = int(custom_degree_divisor)
            for genre in self.genre_records.values():
                genre.degree_divisor = custom_degree_divisor

        for genre in self.genre_records.values():
            genre.predict(up_to)

    def plot_advantages(self):
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(self.genre_records))))

        earliest_prediction_threshold = 30_000
        for (genre, record) in self.genre_records.items():
            years_total = [x.year for x in chain(record.history, record.predicted) if x.games > 0]
            advantage_total = [x.ratio() for x in chain(record.history, record.predicted) if x.games > 0]
            years_history = [x.year for x in record.history if x.games > 0]
            advantage_history = [x.ratio() for x in record.history if x.games > 0]

            color = next(colors)
            earliest_prediction_threshold = min(earliest_prediction_threshold, record.history[-1].year)
            plt.plot(years_total, advantage_total, color=color, label=genre)
            plt.scatter(years_history, advantage_history, color=color)

        plt.axvline(x=earliest_prediction_threshold, color="w", label="Prediction Threshold")
        plt.legend()
        plt.title("Average Sales Per Genre")
        plt.show()

class InvalidGameData(Exception):
    def __init__(self, entry_id, inner: Exception):
        super().__init__()
        self.entry_id = entry_id
        self.inner = inner

    def __str__(self):
        return f"Entry {self.entry_id} caused {self.inner}"

def market_data_from_csv(path: str, from_year: int | None, to_year: int | None) -> MarketData:
    data = MarketData()

    csv = pd.read_csv(path)
    for line in csv.iterrows():
        year = line[1]["Year"]
        if from_year is not None and year < from_year:
            continue
        if to_year is not None and year > to_year:
            continue
        data.add_record(line[1], f"row: {line[0]}")

    return data
