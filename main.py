import tkinter as tk

import algorithm


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Game Market Predictor")

        self.instructions = tk.Label(self, text="Instructions: \n Complete the following prompts and click 'Generate' to produce the results.", font=('Arial', 18))
        self.instructions.pack(padx=20, pady=20)

        self.input_path = tk.StringVar(value="path/to/game/data.csv")
        self.input_path_entry = tk.Entry(self, textvariable=self.input_path)
        self.input_path_entry.pack()

        self.errors = tk.StringVar()
        self.errors_display = tk.Label(self, textvariable=self.errors)
        self.errors_display.pack()

        self.generator = tk.Button(self, text="Generate", command=self.generate)
        self.generator.pack()

    def generate(self):
        try:
            algorithm.predict_market(self.input_path.get())
        except Exception as e:
            self.error(e.__str__())


    def error(self, msg: str):
        self.errors.set(msg)

if __name__ == '__main__':
    app = MainApp()
    app.mainloop()
