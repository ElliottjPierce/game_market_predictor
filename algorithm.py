import pandas as pd

class Advantage:
    def __init__(self, year: int, sales: int = 0, games: int = 0):
        self.sales = sales
        self.games = games
        self.year = year

class GenreRecord:
    def __init__(self, name: str):
        self.name = name
        self.advantages = []

    def get_advantage(self, year: int) -> Advantage:
        if not self.advantages:
            advantage = Advantage(year)
            self.advantages.append(advantage)
            return advantage

        first_year: int = self.advantages[0].year
        index: int = year - first_year

        while index < 0:
            first_year -= 1
            self.advantages.insert(0, Advantage(first_year))
            index += 1

        while index >= len(self.advantages):
            next_year = first_year + len(self.advantages)
            self.advantages.append(Advantage(next_year))

        return self.advantages[int(index)]

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

    return data
