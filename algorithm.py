from itertools import chain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# provides better contrast for graphs
plt.style.use("dark_background")

# Represents the average sales of an average game in a particular genre.
# Practically speaking, this defines how advantageous it is to make a game in this genre, hence the name.
class Advantage:
    def __init__(self, year: int, sales: int = 0, games: float = 0):
        self.sales = sales
        self.games = games
        self.year = year

    # The actual number for the average sales of an average game in a particular genre.
    def ratio(self):
        return self.sales / self.games

# Provides data regarding a genre
class GenreRecord:
    def __init__(self, name: str):
        self.name = name
        self.history = [] # the data from the input
        self.sales_model = None # AI models if generated
        self.games_model = None # AI models if generated
        self.predicted = [] # predicted future data points
        self.degree_divisor = 15 # the number used to generate the AI models.

    # Retrieves the advantage for that year, creating it if it did not exist.
    def get_advantage(self, year: int) -> Advantage:
        # Handle empty case
        if not self.history:
            advantage = Advantage(year)
            self.history.append(advantage)
            return advantage

        # compute the relative index
        first_year: int = self.history[0].year
        index: int = year - first_year

        # handle walking the list back to keep storage dense.
        while index < 0:
            first_year -= 1
            self.history.insert(0, Advantage(first_year))
            index += 1

        # handle walking the list forward to keep storage dense
        while index >= len(self.history):
            next_year = first_year + len(self.history)
            self.history.append(Advantage(next_year))

        # retrieve at the index since we know it exists now
        return self.history[int(index)]

    # uses AI to generate the AI model.
    def make_model(self):
        years_data = [x.year for x in self.history]
        sales_data = [x.sales for x in self.history]
        games_data = [x.games for x in self.history]

        # Use the polyfit algorithm for regression analysis.
        self.sales_model = np.poly1d(np.polyfit(years_data, sales_data, int(len(years_data) / self.degree_divisor)))
        self.games_model = np.poly1d(np.polyfit(years_data, games_data, int(len(years_data) / self.degree_divisor)))

    # generates a new Advantage based only on the AI model
    def direct_predict(self, year: int) -> Advantage:
        if self.games_model is None or self.sales_model is None:
            self.make_model()
        prediction = Advantage(year)
        prediction.sales = max(self.sales_model(year), 0)
        prediction.games = max(self.games_model(year), 0)
        return prediction

    # Uses AI to predict data up to this year.
    def predict(self, up_to: int):
        self.predicted.clear()
        next_year = self.history[-1].year
        while next_year <= up_to:
            next_year += 1
            self.predicted.append(self.direct_predict(next_year))

    # Same as direct_predict, but uses the cache where possible
    def predict_advantage(self, year: int) -> Advantage:
        if year > self.predicted[-1].year:
            return self.direct_predict(year)
        else:
            prediction_index = year - self.history[-1].year
            return self.predicted[prediction_index]

    # uses PyPlot to graph the sales.
    def plot_sales(self):
        years_total = [x.year for x in chain(self.history, self.predicted)]
        sales_total = [x.sales for x in chain(self.history, self.predicted)]
        years_history = [x.year for x in self.history]
        sales_history = [x.sales for x in self.history]

        plt.plot(years_total, sales_total, color="y")
        plt.scatter(years_history, sales_history, color="b")
        plt.title(f"{self.name}'s Sales")
        plt.show()

    # uses PyPlot to graph the games.
    def plot_games(self):
        years_total = [x.year for x in chain(self.history, self.predicted)]
        games_total = [x.games for x in chain(self.history, self.predicted)]
        years_history = [x.year for x in self.history]
        games_history = [x.games for x in self.history]

        plt.plot(years_total, games_total, color="y")
        plt.scatter(years_history, games_history, color="b")
        plt.title(f"{self.name}'s Games")
        plt.show()

    # uses PyPlot to graph the advantages.
    def plot_advantage(self):
        # Check zeroes to make sure we never divide by zero
        years_total = [x.year for x in chain(self.history, self.predicted) if x.games > 0]
        advantage_total = [x.ratio() for x in chain(self.history, self.predicted) if x.games > 0]
        years_history = [x.year for x in self.history if x.games > 0]
        advantage_history = [x.ratio() for x in self.history if x.games > 0]

        plt.plot(years_total, advantage_total, color="y")
        plt.scatter(years_history, advantage_history, color="b")
        plt.title(f"{self.name}'s Advantage")
        plt.show()

# Represents the overall market data
class MarketData:
    def __init__(self):
        self.genre_records = {}
        self.skipped_entries: [InvalidGameData] = []
        self.total_entries: int = 0

    # Creates a record for the particular genre, creating it if it does not exist.
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

    # Predicts the future for all known genres using the custom_degree_divisor to do so if applicable
    def predict(self, up_to: int, custom_degree_divisor: str | None = None):
        # Set the custom_degree_divisor
        if custom_degree_divisor:
            custom_degree_divisor = int(custom_degree_divisor)
            for genre in self.genre_records.values():
                genre.degree_divisor = custom_degree_divisor

        # predict the future
        for genre in self.genre_records.values():
            genre.predict(up_to)

    # uses PyPlot to graph the advantages for every included genre.
    def plot_advantages(self):
        # ensures a unique color
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(self.genre_records))))

        # plot data for each genre
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

        # clean hup the graph
        plt.axvline(x=earliest_prediction_threshold, color="w", label="Prediction Threshold")
        plt.legend()
        plt.title("Average Sales Per Genre")
        plt.show()

# represents an invalid part of the data. These can be ignored, but they still need to be reported to the users.
class InvalidGameData(Exception):
    def __init__(self, entry_id, inner: Exception):
        super().__init__()
        self.entry_id = entry_id
        self.inner = inner

    def __str__(self):
        return f"Entry {self.entry_id} caused {self.inner}"

# Uses Pandas to parse a CSV file into MarketData
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
