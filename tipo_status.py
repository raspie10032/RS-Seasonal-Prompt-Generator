"""Optional UI progress; no prompt/image/error contents are broadcast."""
from functools import wraps
import logging
import sys


def report_status(node_id, stage):
    server = getattr(sys.modules.get("server"), "PromptServer", None)
    instance = getattr(server, "instance", None)
    if node_id is not None and instance is not None:
        try:
            instance.send_sync("rs_tipo_status", {"node": str(node_id), "stage": stage})
        except Exception:
            # A disconnected UI must not fail inference or mask its exception.
            # Do not log exception text: transports may include private payloads.
            logging.warning("TIPO progress unavailable; inference result is unaffected")


def node_status(run):
    @wraps(run)
    def wrapped(*args, **kwargs):
        node_id = kwargs.get("unique_id")
        report_status(node_id, "validating")
        try:
            result = run(*args, **kwargs)
        except BaseException:
            report_status(node_id, "failed")
            raise
        report_status(node_id, "complete" if any(result) else "empty")
        return result
    return wrapped
