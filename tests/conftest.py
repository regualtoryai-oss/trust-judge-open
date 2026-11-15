"""All pytest model traffic must use explicit local test transports."""
from contextlib import contextmanager
from contextvars import ContextVar
import ipaddress
import socket

import httpx
import pytest


# Explicit SDK integration cases require separately provisioned
# pinned tokenizer/model-runtime assets. They are not ordinary unit regressions
# and those upstream assets are not redistributed by the public edition.


def pytest_addoption(parser):
    parser.addoption("--run-dsh-integration", action="store_true", default=False,
                     help="Run optional SDK integrations after separately provisioning pinned tokenizer/runtime assets")


def pytest_configure(config):
    config.addinivalue_line("markers", "optional_dsh_runtime: needs separately provisioned pinned DSH/tokenizer runtime assets")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("optional_dsh_runtime") is not None:
            if not config.getoption("--run-dsh-integration"):
                item.add_marker(pytest.mark.skip(reason="optional DSH/tokenizer asset integration; provision assets and opt in with --run-dsh-integration"))

_expected_block = ContextVar("expected_test_network_block", default=False)


def _local(host):
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="strict")
    if host in {None, "localhost", "testserver"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        return False


@pytest.fixture(scope="session", autouse=True)
def forbid_real_network():
    patch = pytest.MonkeyPatch()
    violations = []
    def block(kind):
        if not _expected_block.get():
            violations.append(kind)
        raise OSError("real external network is forbidden in unit tests")
    original_dns, original_connect = socket.getaddrinfo, socket.socket.connect
    original_send, original_async_send = httpx.Client.send, httpx.AsyncClient.send
    def dns(host, *args, **kwargs):
        if not _local(host): block("dns")
        return original_dns(host, *args, **kwargs)
    def connect(sock, address):
        if isinstance(address, tuple) and not _local(address[0]): block("socket")
        return original_connect(sock, address)
    def explicit_transport(client):
        transport = client._transport
        return isinstance(transport, (httpx.MockTransport, httpx.ASGITransport, httpx.WSGITransport)) or type(transport).__module__.startswith("starlette.testclient")
    def send(client, request, *args, **kwargs):
        if not _local(request.url.host) and not explicit_transport(client):
            try: block("httpx")
            except OSError: raise httpx.ConnectError("real HTTP is forbidden in unit tests") from None
        return original_send(client, request, *args, **kwargs)
    async def async_send(client, request, *args, **kwargs):
        if not _local(request.url.host) and not explicit_transport(client):
            try: block("httpx_async")
            except OSError: raise httpx.ConnectError("real HTTP is forbidden in unit tests") from None
        return await original_async_send(client, request, *args, **kwargs)
    patch.setattr(socket, "getaddrinfo", dns)
    patch.setattr(socket.socket, "connect", connect)
    patch.setattr(httpx.Client, "send", send)
    patch.setattr(httpx.AsyncClient, "send", async_send)
    yield
    patch.undo()
    if violations:
        pytest.fail(f"{len(violations)} unexpected real-network attempts were blocked; fix the test transport")


@pytest.fixture
def expected_network_block():
    @contextmanager
    def expected():
        token = _expected_block.set(True)
        try: yield
        finally: _expected_block.reset(token)
    return expected
