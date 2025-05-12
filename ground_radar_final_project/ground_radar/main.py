import sys
import os
# If this script is run directly, its directory is usually added to sys.path.
# Imports like 'from ui.visualization...' assume 'ui' is a top-level module
# or 'ground_radar' (the directory containing main.py, ui, hardware etc.) is in sys.path.
# Running as 'python -m ground_radar.main' from the parent directory of 'ground_radar'
# or 'python main.py' from within the 'ground_radar' directory should work.
# The original sys.path.append might be needed depending on execution context, but often isn't.

from PyQt5.QtWidgets import QApplication
# Assuming main_window is in the ui subdirectory relative to this main.py
# If main.py is at ground_radar/main.py, and main_window.py is at ground_radar/ui/main_window.py
from ui.main_window import MainWindow

def main():
    """
    Main entry point for the Ground Penetrating Radar (BD-GPR) application.
    Initializes and runs the PyQt5 user interface.
    """
    app = QApplication(sys.argv)

    # Create and show the main window
    # MainWindow will load configurations and initialize necessary components.
    main_window = MainWindow()
    main_window.show()

    # Start the Qt event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    # This structure ensures that main() is called only when the script is executed directly.
    main()

