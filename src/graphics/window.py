"""
Transparent Window System - Creates borderless, transparent overlay windows.
Supports Windows-specific features like click-through and always-on-top.
"""

import logging
import asyncio
from typing import Optional, Callable, Tuple
import tkinter as tk
from tkinter import ttk
import sys

# Windows-specific imports
if sys.platform == "win32":
    import win32gui
    import win32con
    import win32api
    from ctypes import windll, wintypes
    import ctypes

logger = logging.getLogger(__name__)

class TransparentWindow:
    """Transparent overlay window with advanced Windows integration."""
    
    def __init__(self, 
                 width: int = 800, 
                 height: int = 600,
                 transparency: float = 0.8,
                 always_on_top: bool = True,
                 click_through: bool = False,
                 follow_mouse: bool = False):
        
        self.width = width
        self.height = height
        self.transparency = transparency
        self.always_on_top = always_on_top
        self.click_through = click_through
        self.follow_mouse = follow_mouse
        
        # Tkinter components
        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.hwnd: Optional[int] = None
        
        # Window state
        self.visible = True
        self.x = 100
        self.y = 100
        
        # Event callbacks
        self.on_close: Optional[Callable] = None
        self.on_move: Optional[Callable] = None
        self.on_resize: Optional[Callable] = None
        
        logger.info("Transparent window initialized")
    
    async def initialize(self):
        """Initialize the transparent window."""
        try:
            # Create main window
            self.root = tk.Tk()
            self.root.title("VRM AI Character")
            
            # Configure window
            self.root.geometry(f"{self.width}x{self.height}+{self.x}+{self.y}")
            self.root.configure(bg='black')  # Will be made transparent
            
            # Remove window decorations
            self.root.overrideredirect(True)
            
            # Set transparency
            self.root.attributes('-alpha', self.transparency)
            
            # Set topmost if requested
            if self.always_on_top:
                self.root.attributes('-topmost', True)
            
            # Create canvas for 3D rendering
            self.canvas = tk.Canvas(
                self.root,
                width=self.width,
                height=self.height,
                bg='black',
                highlightthickness=0
            )
            self.canvas.pack(fill=tk.BOTH, expand=True)
            
            # Bind events
            self.root.bind('<Button-1>', self._on_click)
            self.root.bind('<B1-Motion>', self._on_drag)
            self.root.bind('<Double-Button-1>', self._on_double_click)
            self.root.bind('<Key>', self._on_key)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close_event)
            
            # Make window focusable for key events
            self.root.focus_set()
            
            # Windows-specific setup
            if sys.platform == "win32":
                await self._setup_windows_features()
            
            logger.info("Transparent window created successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize window: {e}")
            raise
    
    async def _setup_windows_features(self):
        """Setup Windows-specific window features."""
        try:
            # Get window handle
            self.root.update()  # Ensure window is created
            self.hwnd = int(self.root.frame(), 16)
            
            if not self.hwnd:
                logger.warning("Could not get window handle")
                return
            
            # Get current window style
            current_style = win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE)
            
            # Setup layered window for advanced transparency
            new_style = current_style | win32con.WS_EX_LAYERED
            
            if self.click_through:
                new_style |= win32con.WS_EX_TRANSPARENT
            
            if self.always_on_top:
                new_style |= win32con.WS_EX_TOPMOST
            
            # Apply new style
            win32gui.SetWindowLong(self.hwnd, win32con.GWL_EXSTYLE, new_style)
            
            # Set layered window attributes for better transparency control
            win32gui.SetLayeredWindowAttributes(
                self.hwnd,
                0,  # Transparency key (0 = use alpha)
                int(255 * self.transparency),  # Alpha value
                win32con.LWA_ALPHA
            )
            
            # Set window position in Z-order
            if self.always_on_top:
                win32gui.SetWindowPos(
                    self.hwnd,
                    win32con.HWND_TOPMOST,
                    0, 0, 0, 0,
                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
                )
            
            logger.info("Windows-specific features configured")
            
        except Exception as e:
            logger.error(f"Failed to setup Windows features: {e}")
    
    def _on_click(self, event):
        """Handle mouse click events."""
        logger.debug(f"Click at ({event.x}, {event.y})")
        # Store initial click position for dragging
        self._drag_start_x = event.x
        self._drag_start_y = event.y
    
    def _on_drag(self, event):
        """Handle window dragging."""
        if hasattr(self, '_drag_start_x'):
            # Calculate new window position
            new_x = self.root.winfo_x() + (event.x - self._drag_start_x)
            new_y = self.root.winfo_y() + (event.y - self._drag_start_y)
            
            # Move window
            self.root.geometry(f"{self.width}x{self.height}+{new_x}+{new_y}")
            
            if self.on_move:
                self.on_move(new_x, new_y)
    
    def _on_double_click(self, event):
        """Handle double-click events."""
        logger.debug("Double-click detected")
        # Could toggle features or show settings
    
    def _on_key(self, event):
        """Handle keyboard events."""
        logger.debug(f"Key pressed: {event.keysym}")
        
        # Handle special keys
        if event.keysym == 'Escape':
            self.hide()
        elif event.keysym == 'F11':
            self.toggle_fullscreen()
        elif event.keysym == 't':
            self.toggle_transparency()
    
    def _on_close_event(self):
        """Handle window close event."""
        if self.on_close:
            self.on_close()
        else:
            self.close()
    
    async def process_events(self):
        """Process window events (non-blocking)."""
        if self.root:
            try:
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                # Window has been destroyed
                self.root = None
    
    def set_transparency(self, transparency: float):
        """Set window transparency (0.0 = fully transparent, 1.0 = opaque)."""
        self.transparency = max(0.1, min(1.0, transparency))
        
        if self.root:
            self.root.attributes('-alpha', self.transparency)
        
        # Update Windows layered window attributes
        if sys.platform == "win32" and self.hwnd:
            try:
                win32gui.SetLayeredWindowAttributes(
                    self.hwnd,
                    0,
                    int(255 * self.transparency),
                    win32con.LWA_ALPHA
                )
            except Exception as e:
                logger.error(f"Failed to update transparency: {e}")
    
    def toggle_transparency(self):
        """Toggle between transparent and opaque."""
        new_transparency = 0.3 if self.transparency > 0.5 else 1.0
        self.set_transparency(new_transparency)
    
    def set_always_on_top(self, on_top: bool):
        """Set always-on-top behavior."""
        self.always_on_top = on_top
        
        if self.root:
            self.root.attributes('-topmost', on_top)
        
        # Update Windows Z-order
        if sys.platform == "win32" and self.hwnd:
            try:
                hwnd_insert_after = win32con.HWND_TOPMOST if on_top else win32con.HWND_NOTOPMOST
                win32gui.SetWindowPos(
                    self.hwnd,
                    hwnd_insert_after,
                    0, 0, 0, 0,
                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
                )
            except Exception as e:
                logger.error(f"Failed to update always-on-top: {e}")
    
    def set_click_through(self, click_through: bool):
        """Set click-through behavior."""
        self.click_through = click_through
        
        if sys.platform == "win32" and self.hwnd:
            try:
                current_style = win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE)
                
                if click_through:
                    new_style = current_style | win32con.WS_EX_TRANSPARENT
                else:
                    new_style = current_style & ~win32con.WS_EX_TRANSPARENT
                
                win32gui.SetWindowLong(self.hwnd, win32con.GWL_EXSTYLE, new_style)
            except Exception as e:
                logger.error(f"Failed to update click-through: {e}")
    
    def move_to(self, x: int, y: int):
        """Move window to specific position."""
        self.x = x
        self.y = y
        
        if self.root:
            self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")
    
    def resize(self, width: int, height: int):
        """Resize window."""
        self.width = width
        self.height = height
        
        if self.root:
            self.root.geometry(f"{width}x{height}+{self.x}+{self.y}")
        
        if self.canvas:
            self.canvas.configure(width=width, height=height)
    
    def hide(self):
        """Hide the window."""
        if self.root:
            self.root.withdraw()
            self.visible = False
    
    def show(self):
        """Show the window."""
        if self.root:
            self.root.deiconify()
            self.visible = True
    
    def toggle_visibility(self):
        """Toggle window visibility."""
        if self.visible:
            self.hide()
        else:
            self.show()
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self.root:
            current_state = self.root.attributes('-fullscreen')
            self.root.attributes('-fullscreen', not current_state)
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position relative to window."""
        if self.root:
            x = self.root.winfo_pointerx() - self.root.winfo_rootx()
            y = self.root.winfo_pointery() - self.root.winfo_rooty()
            return (x, y)
        return (0, 0)
    
    def get_window_position(self) -> Tuple[int, int]:
        """Get current window position."""
        if self.root:
            return (self.root.winfo_x(), self.root.winfo_y())
        return (self.x, self.y)
    
    def get_window_size(self) -> Tuple[int, int]:
        """Get current window size."""
        return (self.width, self.height)
    
    def bring_to_front(self):
        """Bring window to front."""
        if self.root:
            self.root.lift()
            self.root.focus_force()
    
    def minimize_to_tray(self):
        """Minimize window to system tray (Windows only)."""
        if sys.platform == "win32":
            # This would implement system tray functionality
            self.hide()
        else:
            self.root.iconify()
    
    def close(self):
        """Close the window."""
        if self.root:
            self.root.quit()
            self.root.destroy()
            self.root = None
    
    async def shutdown(self):
        """Shutdown the window system."""
        try:
            self.close()
            logger.info("Window shutdown complete")
        except Exception as e:
            logger.error(f"Error during window shutdown: {e}")

# Utility functions for window management
def get_screen_size() -> Tuple[int, int]:
    """Get screen dimensions."""
    if sys.platform == "win32":
        user32 = windll.user32
        screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        return screensize
    else:
        # Fallback for other platforms
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return (width, height)

def get_window_under_cursor():
    """Get window handle under cursor (Windows only)."""
    if sys.platform == "win32":
        try:
            point = win32gui.GetCursorPos()
            return win32gui.WindowFromPoint(point)
        except:
            return None
    return None
