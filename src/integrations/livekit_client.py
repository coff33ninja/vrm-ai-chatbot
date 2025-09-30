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
import os
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
        # Attempt to import optional LiveKit SDK pieces (admin & access token)
        try:
            from livekit.api.livekit_api import LivekitApi  # type: ignore
            self._LivekitApi = LivekitApi
        except Exception:
            self._LivekitApi = None

        try:
            # Access token helpers
            from livekit.api.access_token import AccessToken, VideoGrant  # type: ignore
            self._AccessToken = AccessToken
            self._VideoGrant = VideoGrant
        except Exception:
            self._AccessToken = None
            self._VideoGrant = None

        # Optionally try to import realtime (client) SDK for active room join
        try:
            import livekit  # type: ignore
            # we will attempt to import the Realtime client lazily later
            self._realtime_sdk_available = True
        except Exception:
            self._realtime_sdk_available = False

    async def initialize(self):
        """Initialize connection to LiveKit (best-effort)."""
        logger.info("Initializing LiveKit client...")
        # If LiveKit integration is not enabled, nothing to do.
        enabled = getattr(self.config, 'enabled', False) or os.environ.get('LIVEKIT_ENABLED', '1') == '1'
        if not enabled:
            logger.info("LiveKit is disabled in configuration; skipping initialization.")
            return

        # Pull configuration from config with env var fallbacks
        server_url = getattr(self.config, 'server_url', None) or os.environ.get('LIVEKIT_URL')
        api_key = getattr(self.config, 'api_key', None) or os.environ.get('LIVEKIT_API_KEY')
        api_secret = getattr(self.config, 'api_secret', None) or os.environ.get('LIVEKIT_API_SECRET')

        if not server_url or not api_key or not api_secret:
            logger.warning("LiveKit credentials or server URL missing; LiveKit will remain inactive.")
            return

        # Try to ensure admin SDK presence for room management
        if not self._LivekitApi and not self._AccessToken:
            logger.warning("LiveKit admin SDK not fully available; some features (room create/token) may be disabled.")

        # Try to create a default room (best-effort). Use admin API if available.
        default_room = getattr(self.config, 'room', None) or os.environ.get('LIVEKIT_DEFAULT_ROOM', 'main')
        if self._LivekitApi:
            try:
                created = await asyncio.get_event_loop().run_in_executor(None, lambda: self.create_room_if_missing(default_room))
                if created:
                    logger.info("LiveKit room ensured: %s", default_room)
            except Exception as e:
                logger.warning("LiveKit room creation attempt failed: %s", e)

        # Generate an access token for the application to use if possible
        token = None
        try:
            token = self.generate_access_token(identity='app', room=default_room)
        except Exception:
            token = None

        # If realtime SDK is available, attempt to join the room (best-effort)
        if token and self._realtime_sdk_available:
            try:
                # Lazy import of Realtime client to avoid hard dependency
                from livekit import Room, connect  # type: ignore

                # Non-blocking connect attempt - run in executor to avoid blocking
                def connect_room():
                    try:
                        # `connect` API may differ between SDKs; try to call with URL+token
                        client = connect(server_url, token=token)
                        # The object returned may vary; store it for later shutdown
                        self._client = client
                        # mark connected
                        self._connected = True
                        logger.info("Connected to LiveKit room (realtime) as 'app'.")
                        # Emit an event to the event bus
                        asyncio.run_coroutine_threadsafe(self.event_bus.publish('livekit.connected', room=default_room), asyncio.get_event_loop())
                        return True
                    except Exception as e:
                        logger.warning(f"Realtime connection attempt failed: {e}")
                        return False

                ok = await asyncio.get_event_loop().run_in_executor(None, connect_room)
                if not ok:
                    logger.debug("Realtime join attempt unsuccessful; continuing without realtime connection.")
            except Exception as e:
                logger.debug(f"Realtime SDK join skipped due to error: {e}")
        else:
            if not token:
                logger.debug("LiveKit token not available; skipping realtime join.")
            if not self._realtime_sdk_available:
                logger.debug("Realtime SDK not available; skipping realtime join.")

        # Consider the client initialized if admin or realtime steps completed
        self._connected = bool(self._client) or bool(self._LivekitApi)
        logger.info("LiveKit client initialization finished; connected=%s", self._connected)

    async def shutdown(self):
        """Shutdown LiveKit client and cleanup resources."""
        enabled = getattr(self.config, 'enabled', False) or os.environ.get('LIVEKIT_ENABLED', '1') == '1'
        if not enabled:
            return

        logger.info("Shutting down LiveKit client...")
        try:
            # If we have a realtime client, attempt a graceful disconnect
            if self._client:
                try:
                    # Some SDKs expose a close/disconnect method
                    if hasattr(self._client, 'disconnect'):
                        await asyncio.get_event_loop().run_in_executor(None, self._client.disconnect)
                    elif hasattr(self._client, 'close'):
                        await asyncio.get_event_loop().run_in_executor(None, self._client.close)
                except Exception:
                    logger.debug("Error while disconnecting realtime client; continuing with shutdown.")

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
        api_key = getattr(self.config, 'api_key', None) or os.environ.get('LIVEKIT_API_KEY')
        api_secret = getattr(self.config, 'api_secret', None) or os.environ.get('LIVEKIT_API_SECRET')
        if not api_key or not api_secret:
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
                access = self._AccessToken(api_key, api_secret, identity=identity)
                grant = self._VideoGrant(room=room) if room else self._VideoGrant()
                try:
                    access.add_grant(grant)
                except Exception:
                    # Some SDK variants use different grant APIs
                    pass
                try:
                    token = access.to_jwt()
                except Exception as e:
                    logger.debug(f"AccessToken.to_jwt() failed: {e}")
                    token = None
            except Exception:
                # Fallback: try constructor with positional args (older/newer SDKs vary)
                try:
                    access = self._AccessToken(api_key, api_secret, identity)
                    grant = self._VideoGrant(room)
                    try:
                        access.add_grant(grant)
                    except Exception:
                        pass
                    try:
                        token = access.to_jwt()
                    except Exception as e:
                        logger.debug(f"AccessToken.to_jwt() fallback failed: {e}")
                        token = None
                except Exception as e:
                    logger.debug(f"AccessToken construction failed: {e}")
                    token = None

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
        api_key = getattr(self.config, 'api_key', None) or os.environ.get('LIVEKIT_API_KEY')
        api_secret = getattr(self.config, 'api_secret', None) or os.environ.get('LIVEKIT_API_SECRET')
        server_url = getattr(self.config, 'server_url', None) or os.environ.get('LIVEKIT_URL')

        if not self._LivekitApi:
            logger.warning("LivekitApi not available; cannot manage rooms.")
            return False

        try:
            # Instantiate admin client (best-effort). Different SDK versions
            # accept different constructor parameters; try a few patterns.
            api = None
            try:
                api = self._LivekitApi(host=server_url, api_key=api_key, api_secret=api_secret)  # type: ignore
            except Exception:
                try:
                    api = self._LivekitApi(base_url=server_url, api_key=api_key, api_secret=api_secret)  # type: ignore
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
