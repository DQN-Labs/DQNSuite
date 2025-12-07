"""gui_utils

Small collection of helpers for creating widgets. Helpers create widgets
but do not pack/grid them; the caller places them so the layout can be
responsible for responsiveness.
"""
from __future__ import annotations

from typing import Dict, Tuple
import numbers

import customtkinter as ctk

# Default fonts (feel free to edit to your preferred font family/size)
# Primary choice: IBM Plex Mono. If it's not installed on the system,
# Tk will fall back to a similar monospace font.
DEFAULT_TITLE_FONT: Tuple[str, int] = ("IBM Plex Mono", 26)
DEFAULT_TEXT_FONT: Tuple[str, int] = ("IBM Plex Mono", 12)
DEFAULT_BUTTON_FONT: Tuple[str, int] = ("IBM Plex Mono", 13)
DEFAULT_LABEL_FONT: Tuple[str, int] = ("IBM Plex Mono", 11)


def create_frame(frame_id: str, container, frame_dict: Dict[str, ctk.CTkFrame]) -> ctk.CTkFrame:
    """Create a frame and register it in the provided dict.

    The frame is gridded at (0,0) with sticky="nsew" so it will expand
    when the container is resized.
    """

    frame = ctk.CTkFrame(container)
    frame_dict[frame_id] = frame
    frame.grid(row=0, column=0, sticky="nsew")
    return frame


def switch_frame(frame) -> None:
    """Raise the supplied frame to the front."""

    frame.tkraise()


def create_hyperparameter_slider(parent, text: str, from_: float, to: float, resolution: float, initial_value: float) -> ctk.CTkFrame:
    """Create a labeled slider inside a frame and return the frame.

    The returned frame contains three children: a left label, the slider,
    and a value label on the right. Caller should grid the frame.
    """

    frame = ctk.CTkFrame(parent)

    label = ctk.CTkLabel(frame, text=text, font=DEFAULT_LABEL_FONT)
    label.grid(row=0, column=0, padx=(10, 8), sticky="w")

    # Determine reasonable number_of_steps (must be int)
    try:
        steps = max(1, int(round((to - from_) / float(resolution))))
    except Exception:
        steps = 100

    slider = ctk.CTkSlider(frame, from_=from_, to=to, number_of_steps=steps, width=400)
    slider.set(initial_value)
    slider.grid(row=0, column=1, sticky="ew", padx=8)

    value_label = ctk.CTkLabel(frame, text=f"{initial_value:.5f}", font=DEFAULT_LABEL_FONT)
    value_label.grid(row=0, column=2, padx=(8, 10))

    frame.grid_columnconfigure(1, weight=1)

    def on_change(val: str | float) -> None:
        try:
            value_label.configure(text=f"{float(val):.5f}")
        except Exception:
            value_label.configure(text=str(val))

    # CTkSlider supports a command callback when the value changes
    slider.configure(command=on_change)

    # Attach convenience properties to the frame so callers can use
    # the same API as before (e.g. `lr_slider.get()`). This keeps the
    # external code minimal while enabling the grid-based layout.
    frame.slider = slider
    frame.get = lambda: slider.get()
    return frame


def create_title_label(parent, text: str, font: Tuple[str, int] | None = None) -> ctk.CTkLabel:
    """Create and return a title label (do not grid/pack it)."""
    # Use default title font unless overridden
    label_font = font if font is not None else DEFAULT_TITLE_FONT
    label = ctk.CTkLabel(parent, text=text, font=label_font)
    return label


def create_text_label(parent, text: str, wraplength: int = 0, pady: numbers.Number = 10) -> ctk.CTkLabel:
    """Create and return a text label without placing it."""

    label = ctk.CTkLabel(parent, text=text, wraplength=wraplength, font=DEFAULT_TEXT_FONT)
    return label


def create_nav_button(parent, text: str, frames: Dict[str, ctk.CTkFrame], frame_name: str) -> ctk.CTkButton:
    """Create and return a navigation button wired to switch frames.

    Buttons use a slightly larger, bold font for improved legibility.
    """

    button = ctk.CTkButton(parent, text=text, command=lambda: switch_frame(frames[frame_name]), font=DEFAULT_BUTTON_FONT)
    return button


def create_banner(parent, title: str, subtitle: str | None = None) -> ctk.CTkFrame:
    """Create a simple banner with title and subtitle.

    Returns a frame that the caller can grid. The banner uses the
    default title/text fonts so styling is consistent across the app.
    """
    banner = ctk.CTkFrame(parent)
    # Two-column layout: title/subtitle on the left, small action area on the right
    banner.grid_columnconfigure(0, weight=1)
    banner.grid_columnconfigure(1, weight=0)

    title_lbl = ctk.CTkLabel(banner, text=title, font=DEFAULT_TITLE_FONT)
    title_lbl.grid(row=0, column=0, sticky="w", padx=(12, 6), pady=(12, 6))

    if subtitle:
        sub_lbl = ctk.CTkLabel(banner, text=subtitle, font=DEFAULT_TEXT_FONT)
        sub_lbl.grid(row=1, column=0, sticky="w", padx=(12, 6), pady=(0, 12))

    # Right-side placeholder frame for small action buttons (filled by caller)
    action_frame = ctk.CTkFrame(banner)
    action_frame.grid(row=0, column=1, rowspan=2, sticky="e", padx=12, pady=6)

    # Expose the action_frame so callers can add buttons into the banner
    banner.action_frame = action_frame
    return banner
