import os
import pytest
import jwt

from src.integrations.livekit_client import LiveKitClient
from src.utils.event_bus import EventBus


class DummyConfig:
    def __init__(self, api_key=None, api_secret=None, server_url=None, room=None, enabled=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.server_url = server_url
        self.room = room
        self.enabled = enabled


@pytest.fixture(autouse=True)
def clear_env():
    # Ensure env vars don't interfere across tests
    for k in ('LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET', 'LIVEKIT_URL', 'LIVEKIT_DEFAULT_ROOM'):
        os.environ.pop(k, None)
    yield


def test_pyjwt_fallback_generates_token():
    cfg = DummyConfig(api_key='testkey', api_secret='secret', server_url='https://example', room='r1')
    client = LiveKitClient(cfg, EventBus())

    token = client.generate_access_token(identity='alice', room='r1', ttl=3600)
    assert token is not None

    # decode with PyJWT to inspect claims
    decoded = jwt.decode(token, 'secret', algorithms=['HS256'], options={"verify_exp": False})
    assert decoded.get('iss') == 'testkey'
    assert decoded.get('sub') == 'alice'
    assert 'video' in decoded
    assert decoded['video'].get('room') == 'r1'
    assert decoded['video'].get('roomJoin') is True


def test_no_credentials_returns_none():
    cfg = DummyConfig(api_key=None, api_secret=None, server_url=None)
    client = LiveKitClient(cfg, EventBus())

    token = client.generate_access_token(identity='bob')
    assert token is None


def test_sdk_path_mocked(monkeypatch):
    # Mock an AccessToken class
    class FakeAccessToken:
        def __init__(self, key, secret, identity=None):
            self.key = key
            self.secret = secret
            self.identity = identity
            self.grants = []

        def add_grant(self, g):
            self.grants.append(g)

        def to_jwt(self):
            return 'sdk-token-for-' + (self.identity or 'unknown')

    class FakeVideoGrant:
        def __init__(self, room=None):
            self.room = room

    cfg = DummyConfig(api_key='k', api_secret='s', server_url='u')
    client = LiveKitClient(cfg, EventBus())

    # Inject fake SDK pieces
    monkeypatch.setattr(client, '_AccessToken', FakeAccessToken)
    monkeypatch.setattr(client, '_VideoGrant', FakeVideoGrant)

    token = client.generate_access_token(identity='svc', room='roomx')
    assert token == 'sdk-token-for-svc'
