import tkinter as tk
import algorithm


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Game Market Predictor")
        self.data = None

        self.instructions = tk.Label(self, text="Instructions: \n Complete the following prompts and click 'Generate' to produce the results.", font=('Arial', 18))
        self.instructions.pack(padx=20, pady=20)

        # self.input_path = tk.StringVar(value="path/to/game/data.csv")
        self.input_path = tk.StringVar(value="~/desktop/vgsales.csv")
        self.input_path_entry = tk.Entry(self, textvariable=self.input_path)
        self.input_path_entry.pack()

        self.start_date = tk.StringVar(value="")
        self.start_date_entry = tk.Entry(self, textvariable=self.start_date)
        self.start_date_entry.pack()

        self.end_date = tk.StringVar(value="")
        self.end_date_entry = tk.Entry(self, textvariable=self.end_date)
        self.end_date_entry.pack()

        self.predict_date = tk.StringVar(value="")
        self.predict_date_entry = tk.Entry(self, textvariable=self.predict_date)
        self.predict_date_entry.pack()

        self.logs = tk.StringVar()
        self.log_display = tk.Label(self, textvariable=self.logs)
        self.log_display.pack()

        self.generator = tk.Button(self, text="Generate", command=self.generate)
        self.generator.pack()

    def generate(self):
        try:
            self.log("Parsing...")
            start_date = self.start_date.get()
            end_date = self.end_date.get()
            self.data = algorithm.market_data_from_csv(self.input_path.get(), 0 if not start_date else int(start_date), 0 if not end_date else int(end_date))
            if len(self.data.genre_records) == 0:
                self.log("Operation Successful but no genres were found.")
                self.data = None
                return
            predict_to = self.predict_date.get()
            if not predict_to:
                self.log("A date needs to be supplied to predict to.")
                self.data = None
                return

            self.log("Parsed. Predicting...")
            self.data.predict(int(predict_to))

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
