import tkinter as tk

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Game Market Predictor")

        self.instructions = tk.Label(self, text="Instructions: ", font=('Arial', 18))

        self.instructions.pack(padx=20, pady=20)


if __name__ == '__main__':
    app = MainApp()
    app.mainloop()
