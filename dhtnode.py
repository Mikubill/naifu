import threading

import hivemind

dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
    start=True
)

print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
print("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))

for thread in threading.enumerate():
    try:
        thread.join()
    except RuntimeError:
        pass