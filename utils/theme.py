import customtkinter as ctk


def is_light() -> bool:
    try:
        return str(ctk.get_appearance_mode()).lower().startswith("light")
    except Exception:
        return False


def surface_bg() -> str:
    # Slightly gray in light mode to avoid harsh white
    return "#F5F6F8" if is_light() else "#2B2B2B"


def panel_bg() -> str:
    # Panel slightly brighter/darker than surface for separation
    return "#FFFFFF" if is_light() else "#1E1E1E"


def border_color() -> str:
    return "#D0D5DD" if is_light() else "#3A3F44"


def text_color() -> str:
    return "#111111" if is_light() else "#FFFFFF"


def subtle_text_color() -> str:
    return "#344054" if is_light() else "#C9CDD2"


def plot_bg() -> str:
    # Use surface background for matplotlib figures so the square blends in
    return surface_bg()

