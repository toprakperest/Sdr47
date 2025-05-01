#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced SDR System Launcher

Version: 2.1
Features:
- Thread-safe GUI updates
- Comprehensive error handling
- System resource monitoring
- Graceful process termination
- Real-time output capture
"""

import os
import sys
import time
import signal
import threading
import subprocess
import psutil
import tkinter as tk
from tkinter import ttk, messagebox
from queue import Queue, Empty
import logging
from logging.handlers import RotatingFileHandler

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            "system_launcher.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=3
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SystemLauncher")



class SystemLauncherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SDR System Launcher v2.1")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)

        # Process management
        self.process = None
        self.output_queue = Queue()
        self.running = True

        # UI setup
        self._setup_ui()
        self._setup_status_bar()

        # Start output processor
        self._start_output_processor()

        # Handle shutdown
        self.root.protocol("WM_DELETE_WINDOW", self._safe_shutdown)
        logger.info("Application initialized")

    def _setup_ui(self):
        """Initialize main UI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Configuration panels
        self._setup_mode_panel(main_frame)
        self._setup_test_panel(main_frame)
        self._setup_advanced_panel(main_frame)

        # Output console
        self._setup_output_console(main_frame)

        # Control buttons
        self._setup_control_buttons(main_frame)

    def _setup_mode_panel(self, parent):
        """Mode selection panel"""
        frame = ttk.LabelFrame(parent, text="Operation Mode", padding="10")
        frame.pack(fill=tk.X, pady=5)

        self.mode_var = tk.StringVar(value="normal")

        modes = [
            ("Normal Operation", "normal"),
            ("Test Mode", "test"),
            ("Calibration", "calibration")
        ]

        for text, mode in modes:
            ttk.Radiobutton(
                frame, text=text, variable=self.mode_var,
                value=mode, command=self._update_ui_state
            ).pack(anchor=tk.W, pady=2)

    def _setup_test_panel(self, parent):
        """Test configuration panel"""
        self.test_frame = ttk.LabelFrame(parent, text="Test Configuration", padding="10")
        self.test_frame.pack(fill=tk.X, pady=5)

        # Test type
        self.test_type_var = tk.StringVar(value="synthetic")
        ttk.Radiobutton(
            self.test_frame, text="Synthetic Data",
            variable=self.test_type_var, value="synthetic"
        ).pack(anchor=tk.W)

        ttk.Radiobutton(
            self.test_frame, text="Replay Data",
            variable=self.test_type_var, value="replay"
        ).pack(anchor=tk.W)

        # Test scenario
        ttk.Label(self.test_frame, text="Scenario:").pack(anchor=tk.W)

        self.scenario_var = tk.StringVar(value="mixed")
        scenarios = [
            "Mixed Objects", "Metal Only", "Void Only",
            "Mineral Only", "Noisy", "Deep Objects"
        ]

        scenario_menu = ttk.OptionMenu(
            self.test_frame, self.scenario_var, "mixed", *[
                (s.lower().replace(" ", "_"), s) for s in scenarios
            ]
        )
        scenario_menu.pack(fill=tk.X, pady=5)

        # Duration
        duration_frame = ttk.Frame(self.test_frame)
        duration_frame.pack(fill=tk.X, pady=5)

        ttk.Label(duration_frame, text="Duration (sec):").pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value="300")
        ttk.Entry(duration_frame, textvariable=self.duration_var, width=10).pack(side=tk.LEFT, padx=5)

        # Initially disabled
        self.test_frame.pack_forget()

    def _setup_advanced_panel(self, parent):
        """Advanced options panel"""
        frame = ttk.LabelFrame(parent, text="Advanced Options", padding="10")
        frame.pack(fill=tk.X, pady=5)

        # Optimization
        self.optimize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame, text="Enable Optimizations",
            variable=self.optimize_var
        ).pack(anchor=tk.W)

        # Debug mode
        self.debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame, text="Debug Mode",
            variable=self.debug_var
        ).pack(anchor=tk.W)

        # Log level
        ttk.Label(frame, text="Log Level:").pack(anchor=tk.W)
        self.log_level_var = tk.StringVar(value="INFO")
        ttk.OptionMenu(
            frame, self.log_level_var, "INFO",
            "DEBUG", "INFO", "WARNING", "ERROR"
        ).pack(fill=tk.X)

    def _setup_output_console(self, parent):
        """Output console panel"""
        frame = ttk.LabelFrame(parent, text="System Output", padding="10")
        frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Text widget with scrollbar
        self.output_text = tk.Text(
            frame, wrap=tk.WORD, state=tk.DISABLED,
            font=('Consolas', 10), bg='black', fg='white'
        )

        scrollbar = ttk.Scrollbar(frame, command=self.output_text.yview)
        self.output_text.config(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _setup_control_buttons(self, parent):
        """Control buttons panel"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            frame, text="Start",
            command=self._start_system
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            frame, text="Stop",
            command=self._stop_system
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            frame, text="Clear",
            command=self._clear_output
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            frame, text="Exit",
            command=self._safe_shutdown
        ).pack(side=tk.RIGHT, padx=5)

    def _setup_status_bar(self):
        """Status bar at bottom"""
        self.status_var = tk.StringVar(value="Ready")

        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X)

        # System status
        ttk.Label(
            status_frame, textvariable=self.status_var,
            relief=tk.SUNKEN, anchor=tk.W
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Resource usage
        self.cpu_var = tk.StringVar(value="CPU: 0%")
        self.mem_var = tk.StringVar(value="MEM: 0%")

        ttk.Label(
            status_frame, textvariable=self.cpu_var,
            relief=tk.SUNKEN, anchor=tk.E, width=10
        ).pack(side=tk.LEFT)

        ttk.Label(
            status_frame, textvariable=self.mem_var,
            relief=tk.SUNKEN, anchor=tk.E, width=10
        ).pack(side=tk.LEFT)

        # Start monitoring
        self._start_resource_monitoring()

    def _start_resource_monitoring(self):
        """Start thread to monitor system resources"""

        def monitor():
            while self.running:
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().percent

                self.root.after(0, lambda: self._update_resource_display(cpu, mem))
                time.sleep(2)

        threading.Thread(
            target=monitor,
            daemon=True
        ).start()

    def _update_resource_display(self, cpu, mem):
        """Update resource usage display"""
        self.cpu_var.set(f"CPU: {cpu:.0f}%")
        self.mem_var.set(f"MEM: {mem:.0f}%")

    def _update_ui_state(self):
        """Update UI based on selected mode"""
        if self.mode_var.get() == "test":
            self.test_frame.pack(fill=tk.X, pady=5)
        else:
            self.test_frame.pack_forget()

    def _start_output_processor(self):
        """Start thread to process output queue"""

        def process_queue():
            while self.running:
                try:
                    line = self.output_queue.get(timeout=0.1)
                    self._append_output(line)
                except Empty:
                    continue

        threading.Thread(
            target=process_queue,
            daemon=True
        ).start()

    def _append_output(self, text):
        """Thread-safe output append"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)

    def _clear_output(self):
        """Clear output console"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)

    def _build_command(self):
        """Build command line based on UI settings"""
        cmd = ["python", "main.py"]

        # Mode
        mode = self.mode_var.get()
        cmd.extend(["--mode", mode])

        # Test options
        if mode == "test":
            cmd.extend([
                "--test-type", self.test_type_var.get(),
                "--scenario", self.scenario_var.get(),
                "--duration", self.duration_var.get()
            ])

        # Advanced options
        if self.optimize_var.get():
            cmd.append("--optimize")

        if self.debug_var.get():
            cmd.append("--debug")

        cmd.extend(["--log-level", self.log_level_var.get()])

        return cmd

    def _start_system(self):
        """Start the system process"""
        if self._is_process_running():
            messagebox.showwarning("Warning", "System is already running")
            return

        try:
            cmd = self._build_command()
            self._append_output(f"Starting: {' '.join(cmd)}\n")

            # Environment setup
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=env
            )

            # Start output reader
            self._start_output_reader()

            self.status_var.set(f"Running: {self.mode_var.get()} mode")
            logger.info(f"System started with PID: {self.process.pid}")

        except Exception as e:
            self._handle_error(f"Failed to start system: {str(e)}")

    def _is_process_running(self):
        """Check if process is running"""
        return self.process and self.process.poll() is None

    def _start_output_reader(self):
        """Start thread to read process output"""

        def reader():
            while self._is_process_running():
                line = self.process.stdout.readline()
                if line:
                    self.output_queue.put(line)
                else:
                    break

            # Process completed
            return_code = self.process.poll()
            self.root.after(0, lambda: self._process_completed(return_code))

        threading.Thread(
            target=reader,
            daemon=True
        ).start()

    def _process_completed(self, return_code):
        """Handle process completion"""
        status = "Completed" if return_code == 0 else f"Failed ({return_code})"
        self.status_var.set(status)
        self._append_output(f"\nProcess {status}\n")

        # Clean up
        if self.process:
            self.process.stdout.close()
            self.process = None

    def _stop_system(self):
        """Stop the running system"""
        if not self._is_process_running():
            messagebox.showwarning("Warning", "No system is running")
            return

        try:
            # First try graceful termination
            self.process.terminate()

            # Wait for 5 seconds
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if not terminated
                self.process.kill()
                self.process.wait()

            self._append_output("\nSystem stopped by user\n")
            self.status_var.set("Ready")
            logger.info("System stopped successfully")

        except Exception as e:
            self._handle_error(f"Failed to stop system: {str(e)}")

    def _handle_error(self, message):
        """Handle and display errors"""
        logger.error(message)
        self._append_output(f"\nERROR: {message}\n")
        messagebox.showerror("Error", message)

    def _safe_shutdown(self):
        """Safe shutdown procedure"""
        self.running = False

        # Stop running process
        if self._is_process_running():
            if messagebox.askyesno(
                    "Confirm",
                    "System is still running. Are you sure you want to exit?"
            ):
                self._stop_system()
            else:
                return

        # Close GUI
        self.root.destroy()
        logger.info("Application shutdown complete")


def main():
    """Main entry point"""
    root = tk.Tk()

    # Style configuration
    style = ttk.Style()
    style.theme_use('clam')

    # Configure fonts
    default_font = ('Tahoma', 10)
    root.option_add('*Font', default_font)

    # Create and run application
    app = SystemLauncherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()