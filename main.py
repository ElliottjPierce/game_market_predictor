import tkinter as tk
import algorithm

class TextEntry(tk.Frame):
    def __init__(self, master, text, initial: str | None = None, **kwargs):
        super().__init__(master, **kwargs)

        self.value = tk.StringVar(master=self, value=initial)
        self.label = tk.Label(self, text=text)
        self.entry = tk.Entry(self, textvariable=self.value)

        self.label.pack(side="left")
        self.entry.pack(side="right")

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Game Market Predictor")
        self.data = None

        self.instructions = tk.Label(self, text="Instructions: \n Complete the following prompts and click 'Generate' to produce the results.", font=('Arial', 18))
        self.instructions.pack(padx=20, pady=20)

        self.input_path = TextEntry(self, text="Path to game data:", initial="~/desktop/vgsales.csv")
        self.input_path.pack()

        self.start_date = TextEntry(self, text="Earliest year if limiting:")
        self.start_date.pack()

        self.end_date = TextEntry(self, text="Latest year if limiting:")
        self.end_date.pack()

        self.predict_date = TextEntry(self, text="Predict year:")
        self.predict_date.pack()

        self.logs = tk.StringVar()
        self.log_display = tk.Label(self, textvariable=self.logs)
        self.log_display.pack()

        self.generator = tk.Button(self, text="Generate", command=self.generate)
        self.generator.pack(pady=10)

        self.genres = tk.Variable(master=self, value=[])
        self.genre_selector = tk.Listbox(self, listvariable=self.genres, state="disabled", height=1)
        self.genre_selector.pack(pady=10)

        self.genre_sales_btn = tk.Button(self, text="Show Genre Sales", command=self.show_genre_sales, state="disabled")
        self.genre_sales_btn.pack()

        self.genre_games_btn = tk.Button(self, text="Show Genre Games", command=self.show_genre_games, state="disabled")
        self.genre_games_btn.pack()

        self.genre_average_sales_btn = tk.Button(self, text="Show Genre Average Sales", command=self.show_genre_average_sales, state="disabled")
        self.genre_average_sales_btn.pack()

        self.advantages_btn = tk.Button(self, text="Show Averages Sales Per Genre", command=self.plot_advantages, state="disabled")
        self.advantages_btn.pack(pady=10)

    def show_genre_sales(self):
        genres = self.genres.get()
        for index in range(0, len(genres)):
            if self.genre_selector.selection_includes(index):
                genre = genres[index]
                self.data.genre_records[genre].plot_sales()

    def show_genre_games(self):
        genres = self.genres.get()
        for index in range(0, len(genres)):
            if self.genre_selector.selection_includes(index):
                genre = genres[index]
                self.data.genre_records[genre].plot_games()

    def show_genre_average_sales(self):
        genres = self.genres.get()
        for index in range(0, len(genres)):
            if self.genre_selector.selection_includes(index):
                genre = genres[index]
                self.data.genre_records[genre].plot_advantage()

    def plot_advantages(self):
        self.data.plot_advantages()

    def generate(self):
        try:
            self.log("Parsing...")
            start_date = self.start_date.value.get()
            end_date = self.end_date.value.get()
            self.data = algorithm.market_data_from_csv(self.input_path.value.get(), None if not start_date else int(start_date), None if not end_date else int(end_date))
            if len(self.data.genre_records) == 0:
                self.log("Operation Successful but no genres were found.")
                self.data = None
                return
            predict_to = self.predict_date.value.get()
            if not predict_to:
                self.log("A date needs to be supplied to predict to.")
                self.data = None
                return

            self.log("Parsed. Predicting...")
            self.data.predict(int(predict_to))
            self.genres.set([genre for genre in self.data.genre_records.keys()])
            self.genre_selector["state"] = "normal"
            self.genre_selector["height"] = 10
            self.genre_sales_btn["state"] = "normal"
            self.genre_games_btn["state"] = "normal"
            self.genre_average_sales_btn["state"] = "normal"
            self.advantages_btn["state"] = "normal"

            if self.data.skipped_entries:
                self.log(F"Operation Successful but skipped {len(self.data.skipped_entries)} invalid entries ({self.data.total_entries} included).")
            else:
                self.log("Operation Successful!")
        except Exception as e:
            self.log(e.__str__())

    def log(self, msg: str):
        self.logs.set(msg)

if __name__ == '__main__':
    app = MainApp()
    app.mainloop()
