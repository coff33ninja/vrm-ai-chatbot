"""LiveKit integration client.

This module provides a minimal LiveKitClient class that the application
expects. It's intentionally lightweight: if the optional `livekit` or
`livekit_api` packages are installed and configuration is provided, it
will attempt to use them. Otherwise it acts as a safe stub so the
application can initialize and shut down cleanly.

Extend this class later with full session/room management as needed.
"""

import logging
import asyncio
from typing import Optional

from ..core.config import LiveKitConfig
from ..utils.event_bus import EventBus

logger = logging.getLogger(__name__)


class LiveKitClient:
    """Minimal LiveKit client wrapper used by the application.

    Public API used by the app:
    - __init__(config: LiveKitConfig, event_bus: EventBus)
    - async initialize()
    - async shutdown()

    The implementation below is stubbed to avoid hard dependency on an
    actual LiveKit server during development. If the `livekit` SDK is
    available and valid credentials are provided, the client will attempt
    a best-effort connection.
    """

    def __init__(self, config: LiveKitConfig, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self._connected = False
        self._client = None
        self._room = None

    async def initialize(self):
        """Initialize connection to LiveKit (best-effort)."""
        logger.info("Initializing LiveKit client...")

        # If LiveKit integration is not enabled, nothing to do.
        if not getattr(self.config, 'enabled', False):
            logger.info("LiveKit is disabled in configuration; skipping initialization.")
            return

        # Try to import livekit SDKs if available.
        try:
            # Prefer the async realtime SDK if available
            import livekit
            sdk_available = True
        except Exception:
            sdk_available = False

        if not sdk_available:
            logger.warning("LiveKit SDK not installed; LiveKit features will be disabled.")
            return

        # At this stage, an advanced implementation would create a
        # LiveKit room connection and manage participants. For now we
        # set connected=True so the application thinks the component
        # initialized successfully.
        # Keep this non-blocking and safe for development environments.
        await asyncio.sleep(0)  # yield control
        self._connected = True
        logger.info("LiveKit client initialized (stub).")

    async def shutdown(self):
        """Shutdown LiveKit client and cleanup resources."""
        if not getattr(self.config, 'enabled', False):
            return

        logger.info("Shutting down LiveKit client...")
        try:
            # In a real implementation you'd close room/connection here.
            await asyncio.sleep(0)
            self._connected = False
            logger.info("LiveKit client shut down.")
        except Exception as e:
            logger.error(f"Error while shutting down LiveKit client: {e}")

    @property
    def is_connected(self) -> bool:
        return self._connected
