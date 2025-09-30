import asyncio
import logging
import keyboard
from types import SimpleNamespace

from ..core.config import SystemConfig
from ..utils.event_bus import EventBus

logger = logging.getLogger(__name__)

class SystemIntegration:
    """Manages system-level integrations like global hotkeys."""

    def __init__(self, config: SystemConfig, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self._hotkey_listener = None
        self._is_running = False

    async def initialize(self):
        """Initializes the system integration, setting up hotkey listeners."""
        # Support both new and older configuration shapes. New configs may
        # provide `hotkeys` (list of objects with key/action/suppress).
        # Older configs provide a single `hotkey_toggle` string. Build a
        # compatible list for registration.
        hotkeys = getattr(self.config, 'hotkeys', None)
        if hotkeys is None:
            # Fallback to single toggle hotkey if available
            toggle = getattr(self.config, 'hotkey_toggle', None)
            if toggle:
                hotkeys = [SimpleNamespace(key=toggle, action='toggle_visibility', suppress=False)]
            else:
                logger.info("No hotkeys configured.")
                return

        self._is_running = True
        loop = asyncio.get_running_loop()

        for hotkey_action in hotkeys:
            try:
                # keyboard.add_hotkey is blocking, so run it in an executor
                await loop.run_in_executor(
                    None,
                    lambda: keyboard.add_hotkey(
                        hotkey_action.key,
                        lambda action=hotkey_action.action: self._on_hotkey_pressed(action),
                        suppress=hotkey_action.suppress
                    )
                )
                logger.info(f"Registered hotkey '{hotkey_action.key}' for action '{hotkey_action.action}'")
            except Exception as e:
                logger.error(f"Failed to register hotkey '{hotkey_action.key}': {e}")

        logger.info("System integration initialized.")

    def _on_hotkey_pressed(self, action: str):
        """Callback function for when a registered hotkey is pressed."""
        logger.info(f"Hotkey pressed for action: {action}")
        asyncio.run_coroutine_threadsafe(
            self.event_bus.publish("hotkey_pressed", action),
            asyncio.get_running_loop()
        )

    async def shutdown(self):
        """Shuts down the system integration, removing hotkey listeners."""
        self._is_running = False
        if self._hotkey_listener:
            # This is a placeholder, as the 'keyboard' library
            # doesn't have a simple way to stop all hotkeys at once.
            # We rely on the application exit to clean up.
            # In a more robust implementation, we might manage listeners individually.
            pass
        keyboard.unhook_all()
        logger.info("System integration shut down.")
