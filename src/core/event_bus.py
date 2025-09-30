"""
Event Bus - Central event system for component communication.
Provides async event handling between different parts of the application.
"""

import logging
import asyncio
from typing import Dict, List, Callable, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class EventBus:
    """Async event bus for component communication."""

    def __init__(self):
        self.listeners: Dict[str, List[Callable]] = defaultdict(list)
        self.running = False

    async def initialize(self):
        """Initialize the event bus."""
        self.running = True
        logger.info("Event bus initialized")

    def subscribe(self, event_name: str, callback: Callable):
        """Subscribe to an event."""
        self.listeners[event_name].append(callback)
        logger.debug(f"Subscribed to event: {event_name}")

    def unsubscribe(self, event_name: str, callback: Callable):
        """Unsubscribe from an event."""
        if callback in self.listeners[event_name]:
            self.listeners[event_name].remove(callback)
            logger.debug(f"Unsubscribed from event: {event_name}")

    async def emit(self, event_name: str, *args, **kwargs):
        """Emit an event to all listeners."""
        if not self.running:
            return

        listeners = self.listeners.get(event_name, [])
        if listeners:
            logger.debug(f"Emitting event: {event_name} to {len(listeners)} listeners")

            # Call all listeners
            for callback in listeners:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args, **kwargs)
                    else:
                        callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in event listener for {event_name}: {e}")

    async def shutdown(self):
        """Shutdown the event bus."""
        self.running = False
        self.listeners.clear()
        logger.info("Event bus shutdown")
