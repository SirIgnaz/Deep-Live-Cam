import os
import webbrowser
import customtkinter as ctk
from typing import Any, Callable, Dict, Tuple
import cv2
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras  # Add this import
from PIL import Image, ImageOps
import time
import json
import modules.globals
import modules.metadata
from modules.face_analyser import (
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    add_blank_map,
    has_valid_map,
    simplify_maps,
)
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    is_image,
    is_video,
    resolve_relative_path,
    has_image_extension,
)
from modules.video_capture import VideoCapturer
from modules.gettext import LanguageManager
import platform

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

ROOT = None
POPUP = None
POPUP_LIVE = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 960
PREVIEW_DEFAULT_HEIGHT = 540

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = (740,)
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = (890,)
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_MAX_HEIGHT = 100
MAPPER_PREVIEW_MAX_WIDTH = 100

DEFAULT_BUTTON_WIDTH = 200
DEFAULT_BUTTON_HEIGHT = 40

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

_ = None
preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None
popup_status_label = None
popup_status_label_live = None
source_label_dict = {}
target_label_dict = {}
source_label_dict_live = {}
target_label_dict_live = {}

img_ft, vid_ft = modules.globals.file_types


def init(start: Callable[[], None], destroy: Callable[[], None], lang: str) -> ctk.CTk:
    global ROOT, PREVIEW, _

    lang_manager = LanguageManager(lang)
    _ = lang_manager._
    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def _update_directml_face_enhancer_override(enabled: bool) -> None:
    from modules.processors.frame import face_enhancer as face_enhancer_module

    face_enhancer_module.set_directml_face_enhancer_override(enabled)


def _update_codeformer_backend_preferences(mask_blur: int, color_strength: float) -> None:
    from modules.processors.frame import face_enhancer as face_enhancer_module

    face_enhancer_module.update_codeformer_preferences(
        mask_blur=mask_blur, color_strength=color_strength
    )


def save_switch_states():
    switch_states = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "color_correction": modules.globals.color_correction,
        "allow_directml_face_enhancer": modules.globals.allow_directml_face_enhancer,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "codeformer_mask_blur": modules.globals.codeformer_mask_blur,
        "codeformer_color_strength": modules.globals.codeformer_color_strength,
    }
    with open("switch_states.json", "w") as f:
        json.dump(switch_states, f)


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            switch_states = json.load(f)
        modules.globals.keep_fps = switch_states.get("keep_fps", True)
        modules.globals.keep_audio = switch_states.get("keep_audio", True)
        modules.globals.keep_frames = switch_states.get("keep_frames", False)
        modules.globals.many_faces = switch_states.get("many_faces", False)
        modules.globals.map_faces = switch_states.get("map_faces", False)
        modules.globals.color_correction = switch_states.get("color_correction", False)
        modules.globals.allow_directml_face_enhancer = switch_states.get(
            "allow_directml_face_enhancer",
            modules.globals.allow_directml_face_enhancer,
        )
        modules.globals.nsfw_filter = switch_states.get("nsfw_filter", False)
        modules.globals.live_mirror = switch_states.get("live_mirror", False)
        modules.globals.live_resizable = switch_states.get("live_resizable", False)
        modules.globals.fp_ui = switch_states.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = switch_states.get("show_fps", False)
        modules.globals.mouth_mask = switch_states.get("mouth_mask", False)
        modules.globals.show_mouth_mask_box = switch_states.get(
            "show_mouth_mask_box", False
        )
        modules.globals.codeformer_mask_blur = int(
            switch_states.get("codeformer_mask_blur", modules.globals.codeformer_mask_blur)
        )
        modules.globals.codeformer_color_strength = float(
            switch_states.get(
                "codeformer_color_strength", modules.globals.codeformer_color_strength
            )
        )
    except FileNotFoundError:
        # If the file doesn't exist, use default values
        pass


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label, show_fps_switch

    load_switch_states()

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme(resolve_relative_path("ui.json"))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(
        f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
    )
    root.configure()
    root.protocol("WM_DELETE_WINDOW", lambda: destroy())
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    content = ctk.CTkFrame(root, corner_radius=18)
    content.grid(row=0, column=0, sticky="nsew", padx=24, pady=24)
    content.grid_columnconfigure((0, 1), weight=1)
    content.grid_rowconfigure(1, weight=1)
    content.grid_rowconfigure(3, weight=1)

    title_font = ctk.CTkFont(size=26, weight="bold")
    subtitle_font = ctk.CTkFont(size=15)

    header_frame = ctk.CTkFrame(content, fg_color="transparent")
    header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

    header_label = ctk.CTkLabel(
        header_frame,
        text=f"{modules.metadata.name}",
        font=title_font,
        anchor="w",
    )
    header_label.pack(fill="x")

    subtitle_label = ctk.CTkLabel(
        header_frame,
        text=_("Prepare your source and target to get started."),
        font=subtitle_font,
        anchor="w",
    )
    subtitle_label.pack(fill="x", pady=(6, 0))

    preview_container = ctk.CTkFrame(content, corner_radius=16)
    preview_container.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(18, 12))
    preview_container.grid_columnconfigure((0, 1), weight=1)
    preview_container.grid_rowconfigure(1, weight=1)

    source_card = ctk.CTkFrame(preview_container, corner_radius=14)
    source_card.grid(row=0, column=0, sticky="nsew", padx=(18, 9), pady=18)
    source_card.grid_columnconfigure(0, weight=1)
    source_card.grid_rowconfigure(1, weight=1)

    source_title = ctk.CTkLabel(
        source_card, text=_("Source"), anchor="w", font=ctk.CTkFont(weight="bold")
    )
    source_title.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))

    source_label = ctk.CTkLabel(source_card, text=None)
    source_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

    target_card = ctk.CTkFrame(preview_container, corner_radius=14)
    target_card.grid(row=0, column=1, sticky="nsew", padx=(9, 18), pady=18)
    target_card.grid_columnconfigure(0, weight=1)
    target_card.grid_rowconfigure(1, weight=1)

    target_title = ctk.CTkLabel(
        target_card, text=_("Target"), anchor="w", font=ctk.CTkFont(weight="bold")
    )
    target_title.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))

    target_label = ctk.CTkLabel(target_card, text=None)
    target_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

    action_frame = ctk.CTkFrame(content, fg_color="transparent")
    action_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
    action_frame.grid_columnconfigure(0, weight=1)
    action_frame.grid_columnconfigure(1, weight=0)
    action_frame.grid_columnconfigure(2, weight=1)

    select_face_button = ctk.CTkButton(
        action_frame,
        text=_("Select a face"),
        cursor="hand2",
        command=lambda: select_source_path(),
    )
    select_face_button.grid(row=0, column=0, sticky="ew", padx=(0, 12), pady=(0, 12))

    swap_faces_button = ctk.CTkButton(
        action_frame,
        text="↔",
        width=48,
        cursor="hand2",
        command=lambda: swap_faces_paths(),
    )
    swap_faces_button.grid(row=0, column=1, padx=6, pady=(0, 12))

    select_target_button = ctk.CTkButton(
        action_frame,
        text=_("Select a target"),
        cursor="hand2",
        command=lambda: select_target_path(),
    )
    select_target_button.grid(row=0, column=2, sticky="ew", padx=(12, 0), pady=(0, 12))

    options_frame = ctk.CTkFrame(content, corner_radius=16)
    options_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
    options_frame.grid_columnconfigure((0, 1), weight=1)

    left_option_column = ctk.CTkFrame(options_frame, fg_color="transparent")
    left_option_column.grid(row=0, column=0, sticky="nsew", padx=(18, 9), pady=18)

    right_option_column = ctk.CTkFrame(options_frame, fg_color="transparent")
    right_option_column.grid(row=0, column=1, sticky="nsew", padx=(9, 18), pady=18)

    for column in (left_option_column, right_option_column):
        column.grid_columnconfigure(0, weight=1)

    keep_fps_value = ctk.BooleanVar(value=modules.globals.keep_fps)
    keep_fps_checkbox = ctk.CTkSwitch(
        left_option_column,
        text=_("Keep fps"),
        variable=keep_fps_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "keep_fps", keep_fps_value.get()),
            save_switch_states(),
        ),
    )
    keep_fps_checkbox.grid(row=0, column=0, sticky="ew", pady=(0, 12))

    keep_frames_value = ctk.BooleanVar(value=modules.globals.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(
        left_option_column,
        text=_("Keep frames"),
        variable=keep_frames_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "keep_frames", keep_frames_value.get()),
            save_switch_states(),
        ),
    )
    keep_frames_switch.grid(row=1, column=0, sticky="ew", pady=12)

    enhancer_value = ctk.BooleanVar(value=modules.globals.fp_ui["face_enhancer"])
    enhancer_switch = ctk.CTkSwitch(
        left_option_column,
        text=_("Face Enhancer"),
        variable=enhancer_value,
        cursor="hand2",
        command=lambda: (
            update_tumbler("face_enhancer", enhancer_value.get()),
            save_switch_states(),
        ),
    )
    enhancer_switch.grid(row=2, column=0, sticky="ew", pady=12)

    map_faces = ctk.BooleanVar(value=modules.globals.map_faces)
    map_faces_switch = ctk.CTkSwitch(
        left_option_column,
        text=_("Map faces"),
        variable=map_faces,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "map_faces", map_faces.get()),
            save_switch_states(),
            close_mapper_window() if not map_faces.get() else None,
        ),
    )
    map_faces_switch.grid(row=3, column=0, sticky="ew", pady=12)

    mouth_mask_var = ctk.BooleanVar(value=modules.globals.mouth_mask)
    mouth_mask_switch = ctk.CTkSwitch(
        left_option_column,
        text=_("Mouth Mask"),
        variable=mouth_mask_var,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "mouth_mask", mouth_mask_var.get()),
            save_switch_states(),
        ),
    )
    mouth_mask_switch.grid(row=4, column=0, sticky="ew", pady=(12, 0))

    keep_audio_value = ctk.BooleanVar(value=modules.globals.keep_audio)
    keep_audio_switch = ctk.CTkSwitch(
        right_option_column,
        text=_("Keep audio"),
        variable=keep_audio_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "keep_audio", keep_audio_value.get()),
            save_switch_states(),
        ),
    )
    keep_audio_switch.grid(row=0, column=0, sticky="ew", pady=(0, 12))

    many_faces_value = ctk.BooleanVar(value=modules.globals.many_faces)
    many_faces_switch = ctk.CTkSwitch(
        right_option_column,
        text=_("Many faces"),
        variable=many_faces_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "many_faces", many_faces_value.get()),
            save_switch_states(),
        ),
    )
    many_faces_switch.grid(row=1, column=0, sticky="ew", pady=12)

    color_correction_value = ctk.BooleanVar(value=modules.globals.color_correction)
    color_correction_switch = ctk.CTkSwitch(
        right_option_column,
        text=_("Fix Blueish Cam"),
        variable=color_correction_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "color_correction", color_correction_value.get()),
            save_switch_states(),
        ),
    )
    color_correction_switch.grid(row=2, column=0, sticky="ew", pady=12)

    directml_face_enhancer_value = ctk.BooleanVar(
        value=modules.globals.allow_directml_face_enhancer
    )
    directml_face_enhancer_switch = ctk.CTkSwitch(
        right_option_column,
        text=_("Force DirectML Face Enhance"),
        variable=directml_face_enhancer_value,
        cursor="hand2",
        command=lambda: (
            _update_directml_face_enhancer_override(directml_face_enhancer_value.get()),
            save_switch_states(),
        ),
    )
    directml_face_enhancer_switch.grid(row=3, column=0, sticky="ew", pady=12)

    codeformer_settings_frame = ctk.CTkFrame(right_option_column, fg_color="transparent")
    codeformer_settings_frame.grid(row=4, column=0, sticky="ew", pady=12)
    codeformer_settings_frame.grid_columnconfigure(0, weight=1)
    codeformer_settings_frame.grid_columnconfigure(1, weight=0)
    codeformer_settings_frame.grid_columnconfigure(2, weight=0)

    codeformer_header = ctk.CTkLabel(
        codeformer_settings_frame,
        text=_("CodeFormer Face Enhance"),
        anchor="w",
        font=ctk.CTkFont(size=13, weight="bold"),
    )
    codeformer_header.grid(row=0, column=0, columnspan=3, sticky="ew")

    def _save_codeformer_settings() -> None:
        save_switch_states()
        _update_codeformer_backend_preferences(
            modules.globals.codeformer_mask_blur,
            modules.globals.codeformer_color_strength,
        )

    mask_blur_label = ctk.CTkLabel(
        codeformer_settings_frame,
        text=_("Mask blur"),
        anchor="w",
    )
    mask_blur_label.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(12, 0))

    mask_blur_slider = ctk.CTkSlider(
        codeformer_settings_frame,
        from_=0,
        to=151,
        number_of_steps=151,
    )
    mask_blur_slider.grid(row=2, column=0, columnspan=3, sticky="ew")
    mask_blur_slider.set(modules.globals.codeformer_mask_blur)

    mask_blur_value = ctk.StringVar(value=str(modules.globals.codeformer_mask_blur))
    mask_blur_entry = ctk.CTkEntry(
        codeformer_settings_frame,
        textvariable=mask_blur_value,
        width=70,
        justify="center",
    )
    mask_blur_entry.grid(row=3, column=0, sticky="ew", pady=(6, 0))

    def _set_mask_blur(value: float, *, from_slider: bool = False) -> None:
        try:
            blur_value = int(round(float(value)))
        except (TypeError, ValueError):
            blur_value = modules.globals.codeformer_mask_blur
        blur_value = max(0, min(151, blur_value))
        modules.globals.codeformer_mask_blur = blur_value
        mask_blur_value.set(str(blur_value))
        if not from_slider:
            mask_blur_slider.set(blur_value)
        _save_codeformer_settings()

    mask_blur_decrease = ctk.CTkButton(
        codeformer_settings_frame,
        text="-",
        width=30,
        command=lambda: _set_mask_blur(modules.globals.codeformer_mask_blur - 1),
    )
    mask_blur_decrease.grid(row=3, column=1, padx=(6, 3), pady=(6, 0))

    mask_blur_increase = ctk.CTkButton(
        codeformer_settings_frame,
        text="+",
        width=30,
        command=lambda: _set_mask_blur(modules.globals.codeformer_mask_blur + 1),
    )
    mask_blur_increase.grid(row=3, column=2, padx=(3, 0), pady=(6, 0))

    mask_blur_slider.configure(command=lambda value: _set_mask_blur(value, from_slider=True))
    mask_blur_entry.bind("<Return>", lambda _: _set_mask_blur(mask_blur_value.get()))
    mask_blur_entry.bind("<FocusOut>", lambda _: _set_mask_blur(mask_blur_value.get()))

    color_strength_label = ctk.CTkLabel(
        codeformer_settings_frame,
        text=_("Color match strength"),
        anchor="w",
    )
    color_strength_label.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 0))

    color_strength_slider = ctk.CTkSlider(
        codeformer_settings_frame,
        from_=0.0,
        to=1.0,
        number_of_steps=100,
    )
    color_strength_slider.grid(row=5, column=0, columnspan=3, sticky="ew")
    color_strength_slider.set(modules.globals.codeformer_color_strength)

    color_strength_value = ctk.StringVar(
        value=f"{modules.globals.codeformer_color_strength:.2f}"
    )
    color_strength_entry = ctk.CTkEntry(
        codeformer_settings_frame,
        textvariable=color_strength_value,
        width=70,
        justify="center",
    )
    color_strength_entry.grid(row=6, column=0, sticky="ew", pady=(6, 0))

    def _set_color_strength(value: float, *, from_slider: bool = False) -> None:
        try:
            strength = float(value)
        except (TypeError, ValueError):
            strength = modules.globals.codeformer_color_strength
        strength = max(0.0, min(1.0, round(strength, 2)))
        modules.globals.codeformer_color_strength = strength
        color_strength_value.set(f"{strength:.2f}")
        if not from_slider:
            color_strength_slider.set(strength)
        _save_codeformer_settings()

    color_strength_decrease = ctk.CTkButton(
        codeformer_settings_frame,
        text="-",
        width=30,
        command=lambda: _set_color_strength(
            modules.globals.codeformer_color_strength - 0.05
        ),
    )
    color_strength_decrease.grid(row=6, column=1, padx=(6, 3), pady=(6, 0))

    color_strength_increase = ctk.CTkButton(
        codeformer_settings_frame,
        text="+",
        width=30,
        command=lambda: _set_color_strength(
            modules.globals.codeformer_color_strength + 0.05
        ),
    )
    color_strength_increase.grid(row=6, column=2, padx=(3, 0), pady=(6, 0))

    color_strength_slider.configure(
        command=lambda value: _set_color_strength(value, from_slider=True)
    )
    color_strength_entry.bind(
        "<Return>", lambda _: _set_color_strength(color_strength_value.get())
    )
    color_strength_entry.bind(
        "<FocusOut>", lambda _: _set_color_strength(color_strength_value.get())
    )

    show_fps_value = ctk.BooleanVar(value=modules.globals.show_fps)
    show_fps_switch = ctk.CTkSwitch(
        right_option_column,
        text=_("Show FPS"),
        variable=show_fps_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "show_fps", show_fps_value.get()),
            save_switch_states(),
        ),
    )
    show_fps_switch.grid(row=5, column=0, sticky="ew", pady=12)

    show_mouth_mask_box_var = ctk.BooleanVar(value=modules.globals.show_mouth_mask_box)
    show_mouth_mask_box_switch = ctk.CTkSwitch(
        right_option_column,
        text=_("Show Mouth Mask Box"),
        variable=show_mouth_mask_box_var,
        cursor="hand2",
        command=lambda: (
            setattr(
                modules.globals,
                "show_mouth_mask_box",
                show_mouth_mask_box_var.get(),
            ),
            save_switch_states(),
        ),
    )
    show_mouth_mask_box_switch.grid(row=6, column=0, sticky="ew", pady=(12, 0))

    cta_frame = ctk.CTkFrame(content, fg_color="transparent")
    cta_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(18, 0))
    cta_frame.grid_columnconfigure((0, 1, 2), weight=1)

    start_button = ctk.CTkButton(
        cta_frame,
        text=_("Start"),
        cursor="hand2",
        command=lambda: analyze_target(start, root),
    )
    start_button.grid(row=0, column=0, sticky="ew", padx=(0, 12))

    stop_button = ctk.CTkButton(
        cta_frame, text=_("Destroy"), cursor="hand2", command=lambda: destroy()
    )
    stop_button.grid(row=0, column=1, sticky="ew", padx=12)

    preview_button = ctk.CTkButton(
        cta_frame, text=_("Preview"), cursor="hand2", command=lambda: toggle_preview()
    )
    preview_button.grid(row=0, column=2, sticky="ew", padx=(12, 0))

    camera_frame = ctk.CTkFrame(content, corner_radius=16)
    camera_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=18)
    camera_frame.grid_columnconfigure((0, 1, 2), weight=1)

    camera_label = ctk.CTkLabel(camera_frame, text=_("Select Camera:"), anchor="w")
    camera_label.grid(row=0, column=0, sticky="ew", padx=(18, 12), pady=18)

    available_cameras = get_available_cameras()
    camera_indices, camera_names = available_cameras

    if not camera_names or camera_names[0] == "No cameras found":
        camera_variable = ctk.StringVar(value="No cameras found")
        camera_optionmenu = ctk.CTkOptionMenu(
            camera_frame,
            variable=camera_variable,
            values=["No cameras found"],
            state="disabled",
        )
    else:
        camera_variable = ctk.StringVar(value=camera_names[0])
        camera_optionmenu = ctk.CTkOptionMenu(
            camera_frame, variable=camera_variable, values=camera_names
        )

    camera_optionmenu.grid(row=0, column=1, sticky="ew", padx=12, pady=18)

    live_button = ctk.CTkButton(
        camera_frame,
        text=_("Live"),
        cursor="hand2",
        command=lambda: webcam_preview(
            root,
            (
                camera_indices[camera_names.index(camera_variable.get())]
                if camera_names and camera_names[0] != "No cameras found"
                else None
            ),
        ),
        state=(
            "normal"
            if camera_names and camera_names[0] != "No cameras found"
            else "disabled"
        ),
    )
    live_button.grid(row=0, column=2, sticky="ew", padx=(12, 18), pady=18)

    status_frame = ctk.CTkFrame(content, fg_color="transparent")
    status_frame.grid(row=6, column=0, columnspan=2, sticky="ew")
    status_frame.grid_columnconfigure(0, weight=1)

    status_label = ctk.CTkLabel(status_frame, text=None, justify="center")
    status_label.grid(row=0, column=0, sticky="ew", pady=(0, 6))

    donate_label = ctk.CTkLabel(
        status_frame, text="Deep Live Cam", justify="center", cursor="hand2"
    )
    donate_label.grid(row=1, column=0, sticky="ew")
    donate_label.configure(
        text_color=ctk.ThemeManager.theme.get("URL").get("text_color")
    )
    donate_label.bind(
        "<Button>", lambda event: webbrowser.open("https://deeplivecam.net")
    )

    return root


def close_mapper_window():
    global POPUP, POPUP_LIVE
    if POPUP and POPUP.winfo_exists():
        POPUP.destroy()
        POPUP = None
    if POPUP_LIVE and POPUP_LIVE.winfo_exists():
        POPUP_LIVE.destroy()
        POPUP_LIVE = None


def analyze_target(start: Callable[[], None], root: ctk.CTk):
    if POPUP != None and POPUP.winfo_exists():
        update_status("Please complete pop-up or close it.")
        return

    if modules.globals.map_faces:
        modules.globals.source_target_map = []

        if is_image(modules.globals.target_path):
            update_status("Getting unique faces")
            get_unique_faces_from_target_image()
        elif is_video(modules.globals.target_path):
            update_status("Getting unique faces")
            get_unique_faces_from_target_video()

        if len(modules.globals.source_target_map) > 0:
            create_source_target_popup(start, root, modules.globals.source_target_map)
        else:
            update_status("No faces found in target")
    else:
        select_output_path(start)


def create_source_target_popup(
        start: Callable[[], None], root: ctk.CTk, map: list
) -> None:
    global POPUP, popup_status_label

    POPUP = ctk.CTkToplevel(root)
    POPUP.title(_("Source x Target Mapper"))
    POPUP.geometry(f"{POPUP_WIDTH}x{POPUP_HEIGHT}")
    POPUP.focus()

    def on_submit_click(start):
        if has_valid_map():
            POPUP.destroy()
            select_output_path(start)
        else:
            update_pop_status("At least 1 source with target is required!")

    scrollable_frame = ctk.CTkScrollableFrame(
        POPUP, width=POPUP_SCROLL_WIDTH, height=POPUP_SCROLL_HEIGHT
    )
    scrollable_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

    def on_select_source(map, button_num):
        update_popup_source(map, button_num)

    def on_adjust_target(map, button_num):
        update_popup_target(map, button_num)

    source_label_dict.clear()
    target_label_dict.clear()

    for item in map:
        id = item["id"]

        entry_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        entry_frame.grid(row=id, column=0, padx=30, pady=12, sticky="ew")
        entry_frame.grid_columnconfigure(0, weight=0)
        entry_frame.grid_columnconfigure(1, weight=0)
        entry_frame.grid_columnconfigure(2, weight=0)
        entry_frame.grid_columnconfigure(3, weight=0)

        select_source_button = ctk.CTkButton(
            entry_frame,
            text=_("Select source image"),
            command=lambda id=id: on_select_source(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        select_source_button.grid(row=0, column=0, padx=(0, 18), pady=(0, 6))

        source_label = ctk.CTkLabel(
            entry_frame,
            text=f"S-{id}",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        source_label.grid(row=0, column=1, padx=10, pady=(0, 6))
        source_label_dict[id] = source_label

        x_label = ctk.CTkLabel(
            entry_frame,
            text="×",
            width=30,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        x_label.grid(row=0, column=2, padx=6, pady=(0, 6))

        target_label = ctk.CTkLabel(
            entry_frame,
            text=f"T-{id}",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        target_label.grid(row=0, column=3, padx=10, pady=(0, 6))
        target_label_dict[id] = target_label

        adjust_target_button = ctk.CTkButton(
            entry_frame,
            text=_("Adjust target faces"),
            command=lambda id=id: on_adjust_target(map, id),
            width=DEFAULT_BUTTON_WIDTH,
        )
        adjust_target_button.grid(row=1, column=3, padx=10, pady=(6, 0), sticky="e")

        if "source" in item:
            source_crop = item["source"].get("cv2")
            if source_crop is not None and source_crop.size > 0:
                source_image = Image.fromarray(
                    cv2.cvtColor(source_crop, cv2.COLOR_BGR2RGB)
                )
                source_image = ImageOps.fit(
                    source_image,
                    (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT),
                    Image.LANCZOS,
                )
                source_photo = ctk.CTkImage(source_image, size=source_image.size)
                source_label.configure(image=source_photo, text="")
                source_label.image = source_photo

        target_crop = item.get("target", {}).get("cv2")
        if target_crop is not None and target_crop.size > 0:
            target_image = Image.fromarray(
                cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
            )
            target_image = ImageOps.fit(
                target_image,
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT),
                Image.LANCZOS,
            )
            target_photo = ctk.CTkImage(target_image, size=target_image.size)
            target_label.configure(image=target_photo, text="")
            target_label.image = target_photo

    popup_status_label = ctk.CTkLabel(POPUP, text=None, justify="center")
    popup_status_label.grid(row=1, column=0, pady=15)

    close_button = ctk.CTkButton(
        POPUP, text=_("Submit"), command=lambda: on_submit_click(start)
    )
    close_button.grid(row=2, column=0, pady=10)


def update_popup_source(map: list, button_num: int) -> list:
    global source_label_dict, RECENT_DIRECTORY_SOURCE

    label = source_label_dict.get(button_num)
    if label is None:
        return map

    source_path = ctk.filedialog.askopenfilename(
        title=_("select a source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if not source_path:
        return map

    cv2_img = cv2.imread(source_path)
    face = get_one_face(cv2_img)

    if not face:
        update_pop_status("Face could not be detected in last upload!")
        return map

    bbox = face.get("bbox") if isinstance(face, dict) else getattr(face, "bbox", None)
    if not bbox:
        update_pop_status("Face could not be detected in last upload!")
        return map

    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = [int(value) for value in (x_min, y_min, x_max, y_max)]
    crop = cv2_img[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        update_pop_status("Face could not be detected in last upload!")
        return map

    map[button_num]["source"] = {"cv2": crop, "face": face}
    RECENT_DIRECTORY_SOURCE = os.path.dirname(source_path)

    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    image = ImageOps.fit(
        image,
        (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT),
        Image.LANCZOS,
    )
    tk_image = ctk.CTkImage(image, size=image.size)
    label.configure(image=tk_image, text="")
    label.image = tk_image
    return map


def update_popup_target(map: list, button_num: int) -> list:
    global target_label_dict

    label = target_label_dict.get(button_num)
    if label is None:
        return map

    try:
        entry = map[button_num]
    except (IndexError, TypeError):
        return map

    target_frames = entry.get("target_faces_in_frame")
    if not target_frames:
        update_pop_status(_("No recorded target faces are available for this entry!"))
        return map

    def _face_bbox(face_obj):
        if isinstance(face_obj, dict):
            return face_obj.get("bbox")
        return getattr(face_obj, "bbox", None)

    def _face_score(face_obj) -> float:
        if isinstance(face_obj, dict):
            return float(face_obj.get("det_score", 0.0))
        return float(getattr(face_obj, "det_score", 0.0))

    frame_cache: Dict[str, np.ndarray] = {}
    candidates = []

    for frame_entry in target_frames:
        location = frame_entry.get("location")
        if not location:
            continue

        if location not in frame_cache:
            frame_image = cv2.imread(location)
            if frame_image is None:
                continue
            frame_cache[location] = frame_image
        else:
            frame_image = frame_cache[location]

        faces = frame_entry.get("faces", [])
        for idx, face in enumerate(faces):
            bbox = _face_bbox(face)
            if not bbox or len(bbox) < 4:
                continue

            x_min, y_min, x_max, y_max = bbox
            x_min = int(max(0, min(frame_image.shape[1] - 1, x_min)))
            y_min = int(max(0, min(frame_image.shape[0] - 1, y_min)))
            x_max = int(max(x_min + 1, min(frame_image.shape[1], x_max)))
            y_max = int(max(y_min + 1, min(frame_image.shape[0], y_max)))

            crop = frame_image[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue

            candidates.append(
                {
                    "key": (location, idx),
                    "frame": frame_entry.get("frame"),
                    "index": idx,
                    "face": face,
                    "crop": crop.copy(),
                    "score": _face_score(face),
                    "selection_preview": None,
                    "label_preview": None,
                }
            )

    if not candidates:
        update_pop_status(_("No recorded target faces are available for this entry!"))
        return map

    selections: Dict[Tuple[str, int], bool] = {
        candidate["key"]: True for candidate in candidates
    }

    selection_window = ctk.CTkToplevel(POPUP if POPUP is not None else ROOT)
    selection_window.title(_("Adjust target faces"))
    selection_window.geometry("420x520")
    selection_window.transient(POPUP if POPUP is not None else ROOT)
    selection_window.grab_set()
    selection_window.grid_columnconfigure(0, weight=1)

    preview_label = ctk.CTkLabel(selection_window, text="")
    preview_label.grid(row=0, column=0, padx=20, pady=(20, 10))

    info_var = ctk.StringVar(value="")
    info_label = ctk.CTkLabel(selection_window, textvariable=info_var)
    info_label.grid(row=1, column=0, padx=20, pady=(0, 10))

    include_var = ctk.BooleanVar(value=True)
    include_switch = ctk.CTkSwitch(
        selection_window,
        text=_("Include this face in the mapping"),
        variable=include_var,
    )
    include_switch.grid(row=2, column=0, padx=20, pady=(0, 10))

    nav_frame = ctk.CTkFrame(selection_window, fg_color="transparent")
    nav_frame.grid(row=3, column=0, padx=20, pady=(0, 10), sticky="ew")
    nav_frame.grid_columnconfigure(0, weight=0)
    nav_frame.grid_columnconfigure(1, weight=1)
    nav_frame.grid_columnconfigure(2, weight=0)

    prev_button = ctk.CTkButton(nav_frame, text=_("Previous"))
    prev_button.grid(row=0, column=0, padx=5)

    slider_steps = max(len(candidates) - 1, 1)
    slider = ctk.CTkSlider(nav_frame, from_=0, to=len(candidates) - 1, number_of_steps=slider_steps)
    slider.grid(row=0, column=1, padx=5, sticky="ew")

    next_button = ctk.CTkButton(nav_frame, text=_("Next"))
    next_button.grid(row=0, column=2, padx=5)

    action_frame = ctk.CTkFrame(selection_window, fg_color="transparent")
    action_frame.grid(row=4, column=0, padx=20, pady=(10, 20), sticky="ew")
    action_frame.grid_columnconfigure((0, 1), weight=1)

    status_label = ctk.CTkLabel(selection_window, text="")
    status_label.grid(row=5, column=0, padx=20, pady=(0, 10))

    def _ensure_selection_preview(candidate: Dict[str, Any]) -> ctk.CTkImage:
        if candidate["selection_preview"] is None:
            image = Image.fromarray(cv2.cvtColor(candidate["crop"], cv2.COLOR_BGR2RGB))
            image = ImageOps.fit(
                image,
                (
                    MAPPER_PREVIEW_MAX_WIDTH * 2,
                    MAPPER_PREVIEW_MAX_HEIGHT * 2,
                ),
                Image.LANCZOS,
            )
            candidate["selection_preview"] = ctk.CTkImage(image, size=image.size)
        return candidate["selection_preview"]

    def _ensure_label_preview(candidate: Dict[str, Any]) -> ctk.CTkImage:
        if candidate["label_preview"] is None:
            image = Image.fromarray(cv2.cvtColor(candidate["crop"], cv2.COLOR_BGR2RGB))
            image = ImageOps.fit(
                image,
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT),
                Image.LANCZOS,
            )
            candidate["label_preview"] = ctk.CTkImage(image, size=image.size)
        return candidate["label_preview"]

    current_index = 0

    def _update_status(message: str) -> None:
        status_label.configure(text=message)

    def _on_include_toggle() -> None:
        key = candidates[current_index]["key"]
        selections[key] = include_var.get()

    def _on_slider_change(value: float) -> None:
        show_candidate(int(round(value)), update_slider=False)

    def _show_previous() -> None:
        show_candidate(current_index - 1)

    def _show_next() -> None:
        show_candidate(current_index + 1)

    def show_candidate(index: int, *, update_slider: bool = True) -> None:
        nonlocal current_index

        index = max(0, min(len(candidates) - 1, index))
        current_index = index
        candidate = candidates[index]

        preview = _ensure_selection_preview(candidate)
        preview_label.configure(image=preview, text="")
        preview_label.image = preview

        frame_idx = candidate.get("frame")
        face_idx = candidate.get("index", 0) + 1
        score = candidate.get("score", 0.0)
        if frame_idx is not None:
            info_text = _("Frame {frame} • Face {face} • Confidence {score:.2f}").format(
                frame=frame_idx,
                face=face_idx,
                score=score,
            )
        else:
            info_text = _("Face {face} • Confidence {score:.2f}").format(
                face=face_idx,
                score=score,
            )
        info_var.set(info_text)

        include_switch.configure(command=None)
        include_var.set(selections[candidate["key"]])
        include_switch.configure(command=_on_include_toggle)

        prev_button.configure(state="normal" if index > 0 else "disabled")
        next_button.configure(state="normal" if index < len(candidates) - 1 else "disabled")

        if len(candidates) > 1:
            slider.configure(state="normal")
            if update_slider:
                slider.configure(command=None)
                slider.set(index)
                slider.configure(command=_on_slider_change)
        else:
            slider.configure(state="disabled")

        _update_status("")

    def _apply_selection() -> None:
        selected_candidates = [
            candidate for candidate in candidates if selections.get(candidate["key"], False)
        ]

        if not selected_candidates:
            _update_status(_("At least one target face must remain selected."))
            return

        filtered_frames = []
        for frame_entry in target_frames:
            location = frame_entry.get("location")
            faces = frame_entry.get("faces", [])
            filtered_faces = []
            for idx, face in enumerate(faces):
                if selections.get((location, idx), False):
                    filtered_faces.append(face)
            if filtered_faces:
                new_entry = dict(frame_entry)
                new_entry["faces"] = filtered_faces
                filtered_frames.append(new_entry)

        entry["target_faces_in_frame"] = filtered_frames

        best_candidate = max(selected_candidates, key=lambda c: c.get("score", 0.0))
        entry["target"] = {"cv2": best_candidate["crop"].copy(), "face": best_candidate["face"]}

        label_preview = _ensure_label_preview(best_candidate)
        label.configure(image=label_preview, text="")
        label.image = label_preview

        selection_window.grab_release()
        selection_window.destroy()
        update_pop_status(_("Target mapping updated!"))

    def _cancel_selection() -> None:
        selection_window.grab_release()
        selection_window.destroy()

    include_switch.configure(command=_on_include_toggle)
    prev_button.configure(command=_show_previous)
    next_button.configure(command=_show_next)
    slider.configure(command=_on_slider_change)

    apply_button = ctk.CTkButton(action_frame, text=_("Apply"), command=_apply_selection)
    apply_button.grid(row=0, column=0, padx=5, sticky="ew")

    cancel_button = ctk.CTkButton(action_frame, text=_("Cancel"), command=_cancel_selection)
    cancel_button.grid(row=0, column=1, padx=5, sticky="ew")

    selection_window.protocol("WM_DELETE_WINDOW", _cancel_selection)

    show_candidate(0)

    selection_window.wait_window()
    return map


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title(_("Preview"))
    preview.configure()
    preview.protocol("WM_DELETE_WINDOW", lambda: toggle_preview())
    preview.resizable(width=True, height=True)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill="both", expand=True)

    preview_slider = ctk.CTkSlider(
        preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value)
    )

    return preview


def update_status(text: str) -> None:
    status_label.configure(text=_(text))
    ROOT.update()


def update_pop_status(text: str) -> None:
    popup_status_label.configure(text=_(text))


def update_pop_live_status(text: str) -> None:
    popup_status_label_live.configure(text=_(text))


def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, update the frame processors
    if PREVIEW.state() == "normal":
        global frame_processors
        frame_processors = get_frame_processors_modules(
            modules.globals.frame_processors
        )


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(
        title=_("select a source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )
    if is_image(source_path):
        modules.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        image = render_image_preview(modules.globals.source_path, (200, 200))
        source_label.configure(image=image)
    else:
        modules.globals.source_path = None
        source_label.configure(image=None)


def swap_faces_paths() -> None:
    global RECENT_DIRECTORY_SOURCE, RECENT_DIRECTORY_TARGET

    source_path = modules.globals.source_path
    target_path = modules.globals.target_path

    if not is_image(source_path) or not is_image(target_path):
        return

    modules.globals.source_path = target_path
    modules.globals.target_path = source_path

    RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
    RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)

    PREVIEW.withdraw()

    source_image = render_image_preview(modules.globals.source_path, (200, 200))
    source_label.configure(image=source_image)

    target_image = render_image_preview(modules.globals.target_path, (200, 200))
    target_label.configure(image=target_image)


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET, img_ft, vid_ft

    PREVIEW.withdraw()
    target_path = ctk.filedialog.askopenfilename(
        title=_("select a target image or video"),
        initialdir=RECENT_DIRECTORY_TARGET,
        filetypes=[img_ft, vid_ft],
    )
    if is_image(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        image = render_image_preview(modules.globals.target_path, (200, 200))
        target_label.configure(image=image)
    elif is_video(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
    else:
        modules.globals.target_path = None
        target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT, img_ft, vid_ft

    if is_image(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title=_("save image output file"),
            filetypes=[img_ft],
            defaultextension=".png",
            initialfile="output.png",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    elif is_video(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title=_("save video output file"),
            filetypes=[vid_ft],
            defaultextension=".mp4",
            initialfile="output.mp4",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    else:
        output_path = None
    if output_path:
        modules.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(modules.globals.output_path)
        start()


def check_and_ignore_nsfw(target, destroy: Callable = None) -> bool:
    """Check if the target is NSFW.
    TODO: Consider to make blur the target.
    """
    from numpy import ndarray
    from modules.predicter import predict_image, predict_video, predict_frame

    if type(target) is str:  # image/video file path
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif type(target) is ndarray:  # frame object
        check_nsfw = predict_frame
    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(
                to_quit=False
            )  # Do not need to destroy the window frame if the target is NSFW
        update_status("Processing ignored!")
        return True
    else:
        return False


def fit_image_to_size(image, width: int, height: int):
    if width is None or height is None or width <= 0 or height <= 0:
        return image
    h, w, _ = image.shape
    ratio_h = 0.0
    ratio_w = 0.0
    ratio_w = width / w
    ratio_h = height / h
    # Use the smaller ratio to ensure the image fits within the given dimensions
    ratio = min(ratio_w, ratio_h)
    
    # Compute new dimensions, ensuring they're at least 1 pixel
    new_width = max(1, int(ratio * w))
    new_height = max(1, int(ratio * h))
    new_size = (new_width, new_height)

    return cv2.resize(image, dsize=new_size)


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(
        video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    if PREVIEW.state() == "normal":
        PREVIEW.withdraw()
    elif modules.globals.source_path and modules.globals.target_path:
        init_preview()
        update_preview()


def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()
    if is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill="x")
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if modules.globals.source_path and modules.globals.target_path:
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        for frame_processor in get_frame_processors_modules(
                modules.globals.frame_processors
        ):
            temp_frame = frame_processor.process_frame(
                get_one_face(cv2.imread(modules.globals.source_path)), temp_frame
            )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(
            image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        update_status("Processing succeed!")
        PREVIEW.deiconify()


def webcam_preview(root: ctk.CTk, camera_index: int):
    global POPUP_LIVE

    if POPUP_LIVE and POPUP_LIVE.winfo_exists():
        update_status("Source x Target Mapper is already open.")
        POPUP_LIVE.focus()
        return

    if not modules.globals.map_faces:
        if modules.globals.source_path is None:
            update_status("Please select a source image first")
            return
        create_webcam_preview(camera_index)
    else:
        modules.globals.source_target_map = []
        create_source_target_popup_for_webcam(
            root, modules.globals.source_target_map, camera_index
        )



def get_available_cameras():
    """Returns a list of available camera names and indices."""
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()

            # Create list of indices and names
            camera_indices = list(range(len(devices)))
            camera_names = devices

            # If no cameras found through DirectShow, try OpenCV fallback
            if not camera_names:
                # Try to open camera with index -1 and 0
                test_indices = [-1, 0]
                working_cameras = []

                for idx in test_indices:
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        working_cameras.append(f"Camera {idx}")
                        cap.release()

                if working_cameras:
                    return test_indices[: len(working_cameras)], working_cameras

            # If still no cameras found, return empty lists
            if not camera_names:
                return [], ["No cameras found"]

            return camera_indices, camera_names

        except Exception as e:
            print(f"Error detecting cameras: {str(e)}")
            return [], ["No cameras found"]
    else:
        # Unix-like systems (Linux/Mac) camera detection
        camera_indices = []
        camera_names = []

        if platform.system() == "Darwin":  # macOS specific handling
            # Try to open the default FaceTime camera first
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                camera_indices.append(0)
                camera_names.append("FaceTime Camera")
                cap.release()

            # On macOS, additional cameras typically use indices 1 and 2
            for i in [1, 2]:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_indices.append(i)
                    camera_names.append(f"Camera {i}")
                    cap.release()
        else:
            # Linux camera detection - test first 10 indices
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_indices.append(i)
                    camera_names.append(f"Camera {i}")
                    cap.release()

        if not camera_names:
            return [], ["No cameras found"]

        return camera_indices, camera_names


def create_webcam_preview(camera_index: int):
    global preview_label, PREVIEW

    cap = VideoCapturer(camera_index)
    if not cap.start(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT, 60):
        update_status("Failed to start camera")
        return

    preview_label.configure(width=PREVIEW_DEFAULT_WIDTH, height=PREVIEW_DEFAULT_HEIGHT)
    PREVIEW.deiconify()

    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
    source_image = None
    prev_time = time.time()
    fps_update_interval = 0.5
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        temp_frame = frame.copy()

        if modules.globals.live_mirror:
            temp_frame = cv2.flip(temp_frame, 1)

        if modules.globals.live_resizable:
            temp_frame = fit_image_to_size(
                temp_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height()
            )

        else:
            temp_frame = fit_image_to_size(
                temp_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height()
            )

        if not modules.globals.map_faces:
            if source_image is None and modules.globals.source_path:
                source_image = get_one_face(cv2.imread(modules.globals.source_path))

            for frame_processor in frame_processors:
                if frame_processor.NAME == "DLC.FACE-ENHANCER":
                    if modules.globals.fp_ui["face_enhancer"]:
                        temp_frame = frame_processor.process_frame(None, temp_frame)
                else:
                    temp_frame = frame_processor.process_frame(source_image, temp_frame)
        else:
            modules.globals.target_path = None
            for frame_processor in frame_processors:
                if frame_processor.NAME == "DLC.FACE-ENHANCER":
                    if modules.globals.fp_ui["face_enhancer"]:
                        temp_frame = frame_processor.process_frame_v2(temp_frame)
                else:
                    temp_frame = frame_processor.process_frame_v2(temp_frame)

        # Calculate and display FPS
        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= fps_update_interval:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time

        if modules.globals.show_fps:
            cv2.putText(
                temp_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        image = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageOps.contain(
            image, (temp_frame.shape[1], temp_frame.shape[0]), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        ROOT.update()

        if PREVIEW.state() == "withdrawn":
            break

    cap.release()
    PREVIEW.withdraw()


def create_source_target_popup_for_webcam(
        root: ctk.CTk, map: list, camera_index: int
) -> None:
    global POPUP_LIVE, popup_status_label_live

    POPUP_LIVE = ctk.CTkToplevel(root)
    POPUP_LIVE.title(_("Source x Target Mapper"))
    POPUP_LIVE.geometry(f"{POPUP_LIVE_WIDTH}x{POPUP_LIVE_HEIGHT}")
    POPUP_LIVE.focus()

    def on_submit_click():
        if has_valid_map():
            simplify_maps()
            update_pop_live_status("Mappings successfully submitted!")
            create_webcam_preview(camera_index)  # Open the preview window
        else:
            update_pop_live_status("At least 1 source with target is required!")

    def on_add_click():
        add_blank_map()
        refresh_data(map)
        update_pop_live_status("Please provide mapping!")

    def on_clear_click():
        clear_source_target_images(map)
        refresh_data(map)
        update_pop_live_status("All mappings cleared!")

    popup_status_label_live = ctk.CTkLabel(POPUP_LIVE, text=None, justify="center")
    popup_status_label_live.grid(row=1, column=0, pady=15)

    add_button = ctk.CTkButton(POPUP_LIVE, text=_("Add"), command=lambda: on_add_click())
    add_button.place(relx=0.1, rely=0.92, relwidth=0.2, relheight=0.05)

    clear_button = ctk.CTkButton(POPUP_LIVE, text=_("Clear"), command=lambda: on_clear_click())
    clear_button.place(relx=0.4, rely=0.92, relwidth=0.2, relheight=0.05)

    close_button = ctk.CTkButton(
        POPUP_LIVE, text=_("Submit"), command=lambda: on_submit_click()
    )
    close_button.place(relx=0.7, rely=0.92, relwidth=0.2, relheight=0.05)



def clear_source_target_images(map: list):
    global source_label_dict_live, target_label_dict_live

    for item in map:
        if "source" in item:
            del item["source"]
        if "target" in item:
            del item["target"]

    for button_num in list(source_label_dict_live.keys()):
        source_label_dict_live[button_num].destroy()
        del source_label_dict_live[button_num]

    for button_num in list(target_label_dict_live.keys()):
        target_label_dict_live[button_num].destroy()
        del target_label_dict_live[button_num]


def refresh_data(map: list):
    global POPUP_LIVE

    scrollable_frame = ctk.CTkScrollableFrame(
        POPUP_LIVE, width=POPUP_LIVE_SCROLL_WIDTH, height=POPUP_LIVE_SCROLL_HEIGHT
    )
    scrollable_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

    def on_sbutton_click(map, button_num):
        map = update_webcam_source(scrollable_frame, map, button_num)

    def on_tbutton_click(map, button_num):
        map = update_webcam_target(scrollable_frame, map, button_num)

    for item in map:
        id = item["id"]

        button = ctk.CTkButton(
            scrollable_frame,
            text=_("Select source image"),
            command=lambda id=id: on_sbutton_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=0, padx=30, pady=10)

        x_label = ctk.CTkLabel(
            scrollable_frame,
            text=f"X",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        x_label.grid(row=id, column=2, padx=10, pady=10)

        button = ctk.CTkButton(
            scrollable_frame,
            text=_("Select target image"),
            command=lambda id=id: on_tbutton_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=3, padx=20, pady=10)

        if "source" in item:
            image = Image.fromarray(
                cv2.cvtColor(item["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{id}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=id, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)

        if "target" in item:
            image = Image.fromarray(
                cv2.cvtColor(item["target"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            target_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"T-{id}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            target_image.grid(row=id, column=4, padx=20, pady=10)
            target_image.configure(image=tk_image)


def update_webcam_source(
        scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global source_label_dict_live

    source_path = ctk.filedialog.askopenfilename(
        title=_("select a source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "source" in map[button_num]:
        map[button_num].pop("source")
        source_label_dict_live[button_num].destroy()
        del source_label_dict_live[button_num]

    if source_path == "":
        return map
    else:
        cv2_img = cv2.imread(source_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["source"] = {
                "cv2": cv2_img[int(y_min): int(y_max), int(x_min): int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                cv2.cvtColor(map[button_num]["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=button_num, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)
            source_label_dict_live[button_num] = source_image
        else:
            update_pop_live_status("Face could not be detected in last upload!")
        return map


def update_webcam_target(
        scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global target_label_dict_live

    target_path = ctk.filedialog.askopenfilename(
        title=_("select a target image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "target" in map[button_num]:
        map[button_num].pop("target")
        target_label_dict_live[button_num].destroy()
        del target_label_dict_live[button_num]

    if target_path == "":
        return map
    else:
        cv2_img = cv2.imread(target_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["target"] = {
                "cv2": cv2_img[int(y_min): int(y_max), int(x_min): int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                cv2.cvtColor(map[button_num]["target"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            target_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"T-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            target_image.grid(row=button_num, column=4, padx=20, pady=10)
            target_image.configure(image=tk_image)
            target_label_dict_live[button_num] = target_image
        else:
            update_pop_live_status("Face could not be detected in last upload!")
        return map
