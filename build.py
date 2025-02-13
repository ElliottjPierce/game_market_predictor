import PyInstaller.__main__

if __name__ == '__main__':
    # Uses Pyinstaller to create executables
    PyInstaller.__main__.run([
        'main.py',
        '--collect-all=matplotlib', # used to capture some hidden dependencies
        '--optimize=2',
        '--onefile',
        '--windowed',
        '--noconsole',
        '--noconfirm',
        '-p ./.venv/', # use the project's packages instead of root
        '-n Game Market Predictor'
    ])