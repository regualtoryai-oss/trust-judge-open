import socket
import httpx
import pytest


def test_accidental_real_http_and_dns_are_stopped_before_external_transport(expected_network_block):
    with expected_network_block():
        with pytest.raises(httpx.ConnectError, match="forbidden"):
            httpx.get("https://not-a-real-account.invalid/", timeout=.1)
        with pytest.raises(OSError, match="forbidden"):
            socket.getaddrinfo("not-a-real-account.invalid", 443)


async def test_explicit_fake_transport_remains_usable_but_real_async_transport_is_blocked(expected_network_block):
    with expected_network_block():
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.ConnectError, match="forbidden"):
                await client.get("https://not-a-real-account.invalid/")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200,json={"synthetic":True}))) as client:
        assert (await client.get("https://mock-provider.invalid/")).json()=={"synthetic":True}
