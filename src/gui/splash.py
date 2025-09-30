"""
Splash Screen - Shows loading progress during application startup.
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
from pathlib import Path

class SplashScreen:
    """Application splash screen with progress indication."""
    
    def __init__(self):
        self.root = None
        self.progress_var = None
        self.status_var = None
        self.is_showing = False
        # Control flag to stop internal simulation when real progress events come
        self._stop_simulation = False
        
    def show(self):
        """Show the splash screen."""
        if self.is_showing:
            return
            
        self.is_showing = True
        
        # Create splash window
        self.root = tk.Tk()
        self.root.title("VRM AI Chatbot")
        self.root.geometry("500x300")
        self.root.resizable(False, False)
        
        # Center on screen
        self.root.eval('tk::PlaceWindow . center')
        
        # Remove window decorations
        self.root.overrideredirect(True)
        
        # Configure style
        self.root.configure(bg='#2c3e50')
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50', padx=40, pady=40)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="🤖 VRM AI Chatbot",
            font=("Arial", 24, "bold"),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        title_label.pack(pady=(0, 10))
        
        # Subtitle
        subtitle_label = tk.Label(
            main_frame,
            text="Your Personal 3D AI Companion",
            font=("Arial", 12),
            fg='#bdc3c7',
            bg='#2c3e50'
        )
        subtitle_label.pack(pady=(0, 30))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            main_frame,
            length=400,
            mode='determinate',
            variable=self.progress_var
        )
        progress_bar.pack(pady=(0, 10))
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing...")
        status_label = tk.Label(
            main_frame,
            textvariable=self.status_var,
            font=("Arial", 10),
            fg='#95a5a6',
            bg='#2c3e50'
        )
        status_label.pack()
        
        # Version info
        version_label = tk.Label(
            main_frame,
            text="Version 1.0.0 | Built with Python",
            font=("Arial", 8),
            fg='#7f8c8d',
            bg='#2c3e50'
        )
        version_label.pack(side=tk.BOTTOM, pady=(20, 0))
        
        # Start progress simulation
        self._start_progress_simulation()
        
        # Keep splash on top
        self.root.attributes('-topmost', True)
        self.root.update()
    
    def _start_progress_simulation(self):
        """Simulate loading progress."""
        def update_progress():
            steps = [
                (10, "Loading configuration..."),
                (25, "Initializing AI systems..."),
                (40, "Setting up voice synthesis..."), 
                (55, "Loading 3D renderer..."),
                (70, "Preparing VRM support..."),
                (85, "Finalizing components..."),
                (100, "Ready!")
            ]
            
            for progress, status in steps:
                if not self.is_showing:
                    break

                if self._stop_simulation:
                    # Real progress reporting has started; stop the simulated steps
                    break

                # Update UI directly from the worker thread. This mirrors the
                # simpler splash implementation and ensures the splash updates
                # even while the main thread is running the asyncio loop.
                try:
                    self.progress_var.set(progress)
                    self.status_var.set(status)
                    # Call update to flush UI changes
                    self.root.update()
                except Exception:
                    # If root is gone or closed, stop
                    break

                time.sleep(0.5)
            
            # Keep splash visible briefly
            time.sleep(1.0)
            # Hide splash (run on main thread via update to avoid race)
            try:
                self.root.after(0, self.hide)
            except Exception:
                self.hide()
        # Run in separate thread so initialization logic isn't blocked.
        thread = threading.Thread(target=update_progress, daemon=True)
        thread.start()

    def stop_simulation(self):
        """Stop the internal progress simulation (used when real init progress begins)."""
        self._stop_simulation = True
    
    def update_progress(self, progress: float, status: str):
        """Update progress bar and status."""
        if not self.is_showing or not self.root:
            return
        # Always schedule UI changes on the main thread
        try:
            # Ensure simulation thread stops
            self.stop_simulation()
            self.root.after(0, lambda: self._apply_progress(progress, status))
        except Exception:
            # Window might be closed
            return

    def _apply_progress(self, progress: float, status: str):
        """Apply progress updates on the Tk main loop."""
        if not self.is_showing or not self.root:
            return

        try:
            self.progress_var.set(progress)
            self.status_var.set(status)
            # Use update_idletasks to flush changes without forcing a full
            # event loop re-entry from a background thread.
            self.root.update_idletasks()
        except Exception:
            # Ignore errors during teardown
            pass
    
    def hide(self):
        """Hide the splash screen."""
        if not self.is_showing or not self.root:
            return
            
        try:
            self.root.destroy()
        except Exception:
            pass
            
        self.root = None
        self.is_showing = False
