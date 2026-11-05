"""Classified transport outcomes shared by bridge consumers and implementations."""


class BridgeFailure(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
