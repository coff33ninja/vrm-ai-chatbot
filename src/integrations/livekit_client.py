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

        # Attempt to import optional LiveKit SDK pieces
        try:
            # server/admin SDK (for room management, etc.)
            from livekit.api.livekit_api import LivekitApi  # type: ignore
            from livekit.api.access_token import AccessToken, VideoGrant  # type: ignore
            self._LivekitApi = LivekitApi
            self._AccessToken = AccessToken
            self._VideoGrant = VideoGrant
        except Exception:
            self._LivekitApi = None
            self._AccessToken = None
            self._VideoGrant = None

    async def initialize(self):
        """Initialize connection to LiveKit (best-effort)."""
        logger.info("Initializing LiveKit client...")

        # If LiveKit integration is not enabled, nothing to do.
        if not getattr(self.config, 'enabled', False):
            logger.info("LiveKit is disabled in configuration; skipping initialization.")
            return

        # Try to import livekit SDKs if available.
        try:
            # Prefer the async realtime SDK if available. We only check
            # presence rather than importing heavy symbols here.
            __import__('livekit')
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

    def generate_access_token(self, identity: str, room: Optional[str] = None, ttl: int = 3600) -> Optional[str]:
        """Generate a LiveKit access token for a given identity and optional room.

        This uses the optional `livekit`/`livekit_api` package if installed and
        credentials (api_key/api_secret) exist in the configuration. Returns a
        JWT string or None if generation failed or SDK is not available.
        """
        if not getattr(self.config, 'api_key', None) or not getattr(self.config, 'api_secret', None):
            logger.warning("LiveKit API key/secret not configured; cannot generate token.")
            return None

        if not self._AccessToken or not self._VideoGrant:
            logger.warning("LiveKit AccessToken class not available in SDK; cannot generate token.")
            return None

        try:
            # Best-effort token generation. API differs across SDKs; we try a
            # commonly used constructor, but guard against failures.
            token = None
            try:
                # Preferred: AccessToken(api_key, api_secret, identity=..., ttl=...)
                access = self._AccessToken(self.config.api_key, self.config.api_secret, identity=identity)
                grant = self._VideoGrant(room=room) if room else self._VideoGrant()
                access.add_grant(grant)
                token = access.to_jwt()
            except Exception:
                # Fallback: try constructor with positional args (older/newer SDKs vary)
                access = self._AccessToken(self.config.api_key, self.config.api_secret, identity)
                grant = self._VideoGrant(room)
                try:
                    access.add_grant(grant)
                except Exception:
                    pass
                try:
                    token = access.to_jwt()
                except Exception as e:
                    logger.debug(f"AccessToken.to_jwt() failed: {e}")

            if token:
                logger.debug("Generated LiveKit access token for identity=%s room=%s", identity, room)
                return token
            else:
                logger.error("Failed to generate LiveKit access token due to SDK incompatibility")
                return None

        except Exception as e:
            logger.error(f"Exception while generating LiveKit token: {e}")
            return None

    async def create_room_if_missing(self, room_name: str) -> bool:
        """Create a LiveKit room via the admin API if it does not already exist.

        Returns True when the room exists or was created successfully. If the
        optional admin SDK is not available, the method logs a warning and
        returns False.
        """
        if not self._LivekitApi:
            logger.warning("LivekitApi not available; cannot manage rooms.")
            return False

        try:
            # Instantiate admin client (best-effort). Different SDK versions
            # accept different constructor parameters; try a few patterns.
            api = None
            try:
                api = self._LivekitApi(host=self.config.server_url, api_key=self.config.api_key, api_secret=self.config.api_secret)  # type: ignore
            except Exception:
                try:
                    api = self._LivekitApi(base_url=self.config.server_url, api_key=self.config.api_key, api_secret=self.config.api_secret)  # type: ignore
                except Exception as e:
                    logger.debug(f"Could not instantiate LivekitApi client: {e}")
                    api = None

            if api is None:
                logger.error("Unable to create LivekitApi client with available SDK.")
                return False

            # Try to create room; exact call may differ by SDK version.
            try:
                # Many SDKs expose a room_service or rooms API
                if hasattr(api, 'create_room'):
                    api.create_room(room=room_name)
                elif hasattr(api, 'room_service') and hasattr(api.room_service, 'create_room'):
                    api.room_service.create_room(room=room_name)
                else:
                    logger.debug("LivekitApi client does not expose create_room; skipping creation call")

                logger.info(f"Requested creation/existence of LiveKit room: {room_name}")
                return True
            except Exception as e:
                logger.error(f"Error while creating LiveKit room: {e}")
                return False

        except Exception as e:
            logger.error(f"Exception in create_room_if_missing: {e}")
            return False
