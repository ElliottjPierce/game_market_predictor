import tkinter as tk
import algorithm


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Game Market Predictor")

        self.instructions = tk.Label(self, text="Instructions: \n Complete the following prompts and click 'Generate' to produce the results.", font=('Arial', 18))
        self.instructions.pack(padx=20, pady=20)

        # self.input_path = tk.StringVar(value="path/to/game/data.csv")
        self.input_path = tk.StringVar(value="~/desktop/vgsales.csv")
        self.input_path_entry = tk.Entry(self, textvariable=self.input_path)
        self.input_path_entry.pack()

        self.logs = tk.StringVar()
        self.log_display = tk.Label(self, textvariable=self.logs)
        self.log_display.pack()

        self.generator = tk.Button(self, text="Generate", command=self.generate)
        self.generator.pack()

    def generate(self):
        try:
            data = algorithm.market_data_from_csv(self.input_path.get())
            data.create_models()

            if data.skipped_entries:
                self.log(F"Operation Successful but skipped {len(data.skipped_entries)} invalid entries ({data.total_entries} included).")
            elif len(data.genre_records) == 0:
                self.log("Operation Successful but no genres were found.")
            else:
                self.log("Operation Successful!")
        except Exception as e:
            self.log(e.__str__())

    def log(self, msg: str):
        self.logs.set(msg)

if __name__ == '__main__':
    app = MainApp()
    app.mainloop()
