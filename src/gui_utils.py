"""

# gui_utils.py

This file contains utility functions that are used throughout the application.

"""
import numbers

import customtkinter as ctk


def create_frame(frame_id: str, container, frame_dict: dict[str, ctk.CTkFrame]) -> ctk.CTkFrame:
    """
    Creates a frame as a child of the container and stores it in the frame_dict.

    :param frame_id: The unique ID of the frame
    :param container: The parent container
    :param frame_dict: The dict to store the frame in
    :return: ctk.CTkFrame
    """

    frame = ctk.CTkFrame(container)
    frame_dict[frame_id] = frame
    frame.grid(row=0, column=0, sticky="nsew")
    return frame


def switch_frame(frame) -> None:
    """
    Switches to the given frame in the GUI.

    :param frame: The frame to switch to
    :return:
    """

    frame.tkraise()

def create_hyperparameter_slider(parent, text, from_, to, resolution, initial_value) -> ctk.CTkSlider:
    frame = ctk.CTkFrame(parent)
    frame.pack(pady=5, fill="x")

    label = ctk.CTkLabel(frame, text=text)
    label.pack(side="left", padx=10)

    value_label = ctk.CTkLabel(frame, text=f"{initial_value:.5f}")
    value_label.pack(side="right", padx=10)

    slider = ctk.CTkSlider(frame, from_=from_, to=to, number_of_steps=(to - from_) / resolution, width=400)
    slider.set(initial_value)
    slider.pack(side="left", padx=10, fill="x", expand=True)

    def update_value_label(event):
        value_label.configure(text=f"{slider.get():.5f}")

    slider.bind("<Motion>", update_value_label)

    return slider

def create_title_label(parent, text: str, font: tuple = ("Helvetica", 24)) -> ctk.CTkLabel:
    """
    Creates a title using the given text.

    :param parent: The parent frame
    :param text: Text content
    :param font: Font style
    :return: ctk.CTkLabel
    """

    label = ctk.CTkLabel(parent, text=text, font=font)
    label.pack(pady=(50, 20))
    return label


def create_text_label(parent, text: str, wraplength: int = 0, pady: numbers.Number = 10) -> ctk.CTkLabel:
    label = ctk.CTkLabel(parent, text=text, wraplength=wraplength)
    label.pack(pady=pady)
    return label


def create_nav_button(parent, text, frames, frame_name) -> ctk.CTkButton:
    button = ctk.CTkButton(parent, text=text, command=lambda: switch_frame(frames[frame_name]))
    button.pack(pady=10)
    return button
