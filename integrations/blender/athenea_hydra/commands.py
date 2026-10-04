# Copyright (c) 2026 jesus luque.
"""`athenea mesh2splat`, `flatten` and `compare`, run inside Blender.

hdAthenea carries both commands as C entry points (apps/athenea/src/
Embedded.cpp): the plugin is built against Blender's own USD, so the stages
the commands read and write are that USD's and nothing loads a second one.
Their work is the engine's, on the GPU; this module only hands over the
arguments, runs the call on a worker thread (ctypes lets go of the GIL) and
queues the lines the command prints, for the UI to drain.

A `Job` is a list of commands run one after the other on one thread. Each
step may be marked optional: its failure is reported and the job goes on
(a shadow catcher that finds no ground does not undo the object's cloud).
"""

import ctypes
import os
import queue
import re
import threading
import time

# The entry points' contract this add-on speaks (athenea_embedded_abi).
ABI = 2
# A command's exit code when it was asked to stop.
CANCELLED = 4

_SinkType = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_library = None


def plugin_library():
    """hdAthenea, loaded through USD's registry and then opened by ctypes
    (the same image: dlopen counts a second open of a loaded file)."""
    global _library
    if _library is not None:
        return _library
    from . import plugin_dir, register_plugin
    if not register_plugin():
        raise RuntimeError("hdAthenea is not registered (see the add-on preferences)")
    from pxr import Plug
    plugin = Plug.Registry().GetPluginWithName("hdAthenea")
    if plugin is None:
        raise RuntimeError("USD knows no plugin named hdAthenea")
    if not plugin.isLoaded:
        plugin.Load()
    path = plugin.path or os.path.join(plugin_dir(), "hdAthenea", "hdAthenea.so")
    library = ctypes.CDLL(path)
    try:
        abi = library.athenea_embedded_abi
    except AttributeError:
        raise RuntimeError(f"{path} carries no commands (built without ATHENEA_HYDRA_COMMANDS)")
    abi.restype = ctypes.c_int
    abi.argtypes = []
    if abi() != ABI:
        raise RuntimeError(f"{path}: entry point ABI {abi()}, this add-on speaks {ABI}")
    library.athenea_request_cancel.restype = None
    library.athenea_request_cancel.argtypes = []
    for name in ("athenea_mesh2splat", "athenea_flatten", "athenea_compare"):
        entry = getattr(library, name)
        entry.restype = ctypes.c_int
        entry.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), _SinkType, ctypes.c_void_p]
    _library = library
    return library


def bundle_dirs():
    """Where the AOFX bundles are: beside the plugin in a package
    (`plugin/aofx`), or the build tree's `aofx`."""
    from . import plugin_dir
    directory = plugin_dir()
    if not directory:
        return []
    found = []
    for candidate in (os.path.join(os.path.dirname(directory), "aofx"),
                      os.path.join(os.path.dirname(os.path.dirname(directory)), "aofx")):
        if os.path.isdir(candidate) and candidate not in found:
            found.append(candidate)
    return found


class Step:
    """One command of a job, and the share of the job's progress it takes."""

    def __init__(self, command, args, label, weight=1.0, optional=False):
        self.command = command      # "mesh2splat", "flatten" or "compare"
        self.args = list(args)
        self.label = label          # what the panel calls it
        self.weight = weight
        self.optional = optional
        self.code = None            # its exit code, once run


# What a command's lines say it has reached, as a share of that command:
# the first pattern a line matches (in this order) moves the bar there.
_PHASES = (
    (re.compile(r"shadow catcher of"), 0.05),
    (re.compile(r"density per mesh|the cell is a pixel"), 0.08),
    (re.compile(r"textures decoded"), 0.15),
    (re.compile(r" splats of |-> \d+ splats"), 0.3),
    (re.compile(r"transfer baked in slices"), 0.35),
    (re.compile(r"transfer baked for|^mesh2splat: baked|traced in"), 0.9),
    (re.compile(r"^mesh2splat: wrote|^flatten: wrote"), 1.0),
)
# A progress line (core/Progress: "... 512/2048 points (25.0%) ..."), where
# the engine prints them: the bake's own share, between 0.35 and 0.9.
_PERCENT = re.compile(r"\((\d+(?:\.\d+)?)%\)")


class Job:
    """Steps run on a worker thread; lines and progress for the UI."""

    def __init__(self, steps):
        self.steps = steps
        self.lines = queue.Queue()
        self.log = []
        self.errors = []
        self.warnings = []
        self.progress = 0.0
        self.status = ""
        self.started = 0.0
        self.thread = None
        self.failed_step = None
        self.cancelled = False
        self._library = None
        self._current = 0
        self._within = 0.0
        self._sink = _SinkType(self._on_line)

    def _on_line(self, error, text, _user):
        # On the worker thread: only queue.
        self.lines.put(("line", int(error), text.decode("utf-8", "replace") if text else ""))

    def start(self):
        library = plugin_library()
        self._library = library

        def work():
            for index, step in enumerate(self.steps):
                if self.cancelled:
                    self.lines.put(("failed", index, ""))
                    return
                self.lines.put(("step", index, ""))
                encoded = [a.encode("utf-8") for a in step.args]
                argv = (ctypes.c_char_p * len(encoded))(*encoded)
                entry = getattr(library, "athenea_" + step.command)
                try:
                    step.code = entry(len(encoded), argv, self._sink, None)
                except Exception as error:   # a ctypes failure, not the command's
                    self.lines.put(("line", 1, f"{step.command}: {error}\n"))
                    step.code = 1
                if step.code != 0 and (not step.optional or step.code == CANCELLED):
                    self.lines.put(("failed", index, ""))
                    return
            self.lines.put(("done", len(self.steps), ""))

        for step in self.steps:
            self.log.append(f"athenea {step.command} " + " ".join(step.args))
        self.started = time.monotonic()
        self.thread = threading.Thread(target=work, name="athenea-commands", daemon=True)
        self.thread.start()

    def _fraction(self):
        total = sum(s.weight for s in self.steps) or 1.0
        done = sum(s.weight for s in self.steps[:self._current])
        here = self.steps[self._current].weight if self._current < len(self.steps) else 0.0
        return min((done + here * self._within) / total, 1.0)

    def drain(self):
        """The lines so far, into the log and the progress. True while running."""
        while True:
            try:
                kind, value, text = self.lines.get_nowait()
            except queue.Empty:
                break
            if kind == "step":
                self._current = value
                self._within = 0.0
                self.status = self.steps[value].label
            elif kind == "failed":
                self.failed_step = self.steps[value]
            elif kind == "done":
                self._current = value
                self._within = 0.0
            else:
                for line in text.splitlines():
                    self._line(value, line.rstrip())
            self.progress = max(self.progress, self._fraction())
        return self.thread is not None and self.thread.is_alive()

    def _line(self, error, line):
        if not line:
            return
        self.log.append(line)
        step = self.steps[self._current] if self._current < len(self.steps) else None
        if error:
            (self.warnings if step is not None and step.optional else self.errors).append(line)
        percent = _PERCENT.search(line)
        if percent:
            self._within = max(self._within, 0.35 + 0.55 * float(percent.group(1)) / 100.0)
        else:
            for pattern, share in _PHASES:
                if pattern.search(line):
                    self._within = max(self._within, share)
                    break
        label = step.label if step is not None else ""
        self.status = f"{label}: {line.split(': ', 1)[-1]}"[:160]

    def cancel(self):
        """Asks the running command to stop (it looks between its batches)
        and runs no further step."""
        self.cancelled = True
        if self._library is not None:
            self._library.athenea_request_cancel()

    def wait(self):
        """For a script (Blender in the background has no event loop)."""
        while self.drain():
            time.sleep(0.1)
        self.drain()

    @property
    def seconds(self):
        return time.monotonic() - self.started

    def failure(self):
        """The message of a failed required step, or None."""
        step = self.failed_step
        if step is None:
            return None
        if self.cancelled:
            return "cancelled"
        message = self.errors[-1] if self.errors else f"{step.command} exited with {step.code}"
        if step.code == 3:
            message += " (the GPU ran out of memory: a lower quality, or close what else holds the GPU)"
        return message
