#!/usr/bin/env python3
"""Small mTLS tunnel for one Porthos socket.

``run_porthos_mtls.py`` starts one server/client pair per directional channel,
so all six Porthos sockets of a three-party run travel over mutually
authenticated TLS.
"""

from __future__ import annotations

import argparse
import selectors
import socket
import ssl
import time


def address(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator:
        raise argparse.ArgumentTypeError("address must be HOST:PORT")
    return host, int(port)


def relay(left: socket.socket, right: socket.socket) -> None:
    selector = selectors.DefaultSelector()
    selector.register(left, selectors.EVENT_READ, right)
    selector.register(right, selectors.EVENT_READ, left)
    try:
        while selector.get_map():
            # SSL_read may decrypt more records than one recv() returns. Those
            # bytes live in OpenSSL rather than the kernel socket, so select()
            # will not wake again until SSLSocket.pending() is drained.
            buffered = [
                key for key in selector.get_map().values()
                if isinstance(key.fileobj, ssl.SSLSocket)
                and key.fileobj.pending()
            ]
            ready = {key.fd: key for key, _ in selector.select(
                timeout=0 if buffered else None)}
            ready.update({key.fd: key for key in buffered})
            for key in ready.values():
                if key.fd not in selector.get_map():
                    continue
                source, destination = key.fileobj, key.data
                block = source.recv(1024 * 1024)
                if not block:
                    selector.unregister(source)
                    try:
                        destination.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
                    continue
                destination.sendall(block)
    finally:
        left.close()
        right.close()


def relay_one_way(source: socket.socket, destination: socket.socket) -> None:
    """Relay a protocol channel whose reverse traffic uses another socket."""
    try:
        while True:
            block = source.recv(64 * 1024)
            if not block:
                try:
                    destination.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                return
            destination.sendall(block)
    finally:
        source.close()
        destination.close()


def listener(endpoint: tuple[str, int]) -> socket.socket:
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(endpoint)
    sock.listen(1)
    return sock


def connect_with_retry(
    endpoint: tuple[str, int], timeout_seconds: float
) -> socket.socket:
    """Connect once a lazily-created peer channel starts listening."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            return socket.create_connection(endpoint)
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("server", "client"))
    parser.add_argument("--listen", type=address, required=True)
    parser.add_argument("--connect", type=address, required=True)
    parser.add_argument("--ca", required=True)
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--server-name", default="porthos-p0")
    parser.add_argument("--connections", type=int, default=0,
                        help="stop after N relays; zero serves until terminated")
    parser.add_argument(
        "--relay-direction", choices=("duplex", "left-to-right", "right-to-left"),
        default="duplex",
        help="use a one-way relay when the protocol has directional sockets")
    parser.add_argument(
        "--connect-timeout", type=float, default=30.0,
        help="seconds to wait for a lazily-created peer channel")
    args = parser.parse_args()

    if args.mode == "server":
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_verify_locations(args.ca)
        context.load_cert_chain(args.cert, args.key)
        context.verify_mode = ssl.CERT_REQUIRED
    else:
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=args.ca)
        context.load_cert_chain(args.cert, args.key)

    served = 0
    with listener(args.listen) as incoming:
        while not args.connections or served < args.connections:
            local, _ = incoming.accept()
            if args.mode == "server":
                protected = context.wrap_socket(local, server_side=True)
                plain = connect_with_retry(args.connect, args.connect_timeout)
            else:
                protected = context.wrap_socket(
                    connect_with_retry(args.connect, args.connect_timeout),
                    server_hostname=args.server_name)
                plain = local
            if args.relay_direction == "left-to-right":
                relay_one_way(plain, protected)
            elif args.relay_direction == "right-to-left":
                relay_one_way(protected, plain)
            else:
                relay(plain, protected)
            served += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
