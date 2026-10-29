"""MongoDB connection helper with a mongomock fallback.

A real local MongoDB (127.0.0.1:27017) is preferred because it provides a real
query planner and index benchmarks.  If no server is reachable the practicals
still run against mongomock.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MongoHandle:
    client: object
    is_real: bool
    uri: str

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


def get_mongo(uri: str = "mongodb://127.0.0.1:27017/", timeout_ms: int = 2500) -> MongoHandle:
    try:
        from pymongo import MongoClient

        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        client.admin.command("ping")
        return MongoHandle(client=client, is_real=True, uri=uri)
    except Exception:
        import mongomock

        return MongoHandle(client=mongomock.MongoClient(), is_real=False, uri="mongomock://in-memory")
