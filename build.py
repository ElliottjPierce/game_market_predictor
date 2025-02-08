import PyInstaller.__main__

if __name__ == '__main__':
    PyInstaller.__main__.run([
        'main.py',
        '--onefile',
        '--windowed',
        '--noconsole',
        '--noconfirm',
        '--collect-all=tk_tools',
        '-n Game Market Predictor'
    ])