import tkinter as tk

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Game Market Predictor")
        self.mainloop()


if __name__ == '__main__':
    app = MainApp()
