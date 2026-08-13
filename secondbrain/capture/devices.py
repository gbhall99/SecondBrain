"""Audio input device discovery (wraps sounddevice; Mac/PortAudio at runtime)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InputDevice:
    index: int
    name: str
    channels: int
    default: bool


def list_input_devices() -> list[InputDevice]:
    """Enumerate available input devices. Requires the ``audio`` extra."""
    import sounddevice as sd  # lazy: needs PortAudio

    default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
    devices = []
    for idx, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            devices.append(
                InputDevice(
                    index=idx,
                    name=d["name"],
                    channels=d["max_input_channels"],
                    default=(idx == default_in),
                )
            )
    return devices


class DeviceNotFoundError(LookupError):
    """A configured input_device name matched no available input device."""


def resolve_device(name: str):
    """Return a sounddevice device index for ``name`` (or None for default).

    An unset/blank ``name`` is an intentional "use the system default" and
    returns None. A configured name that matches nothing raises
    :class:`DeviceNotFoundError` — silently falling back to the default mic
    would record the wrong room without anyone noticing.
    """
    if not name:
        return None
    for dev in list_input_devices():
        if dev.name == name or name.lower() in dev.name.lower():
            return dev.index
    raise DeviceNotFoundError(f"input device {name!r} not found (run `sb devices`)")
