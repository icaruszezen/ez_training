# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ez_training (onedir mode)."""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ── paths ──────────────────────────────────────────────────────────────
SRC_DIR = os.path.join(SPECPATH, "src")

# ── data files ─────────────────────────────────────────────────────────
datas = []

# qfluentwidgets resources
datas += collect_data_files("qfluentwidgets", includes=["**/*.qss", "**/*.svg",
                                                         "**/*.png", "**/*.ttf",
                                                         "**/*.json"])

# labeling resources (icons, strings, predefined classes)
labeling_res = os.path.join(SRC_DIR, "ez_training", "labeling", "resources")
labeling_data = os.path.join(SRC_DIR, "ez_training", "labeling", "data")
datas += [
    (labeling_res, os.path.join("ez_training", "labeling", "resources")),
    (labeling_data, os.path.join("ez_training", "labeling", "data")),
]

# annotation script templates (shipped as data so users can edit them)
datas += [
    (os.path.join(SRC_DIR, "ez_training", "annotation_scripts"), os.path.join("ez_training", "annotation_scripts")),
]

# ── hidden imports ─────────────────────────────────────────────────────
hiddenimports = (
    collect_submodules("qfluentwidgets")
    + collect_submodules("lxml")
    + [
        "cv2",
        "PIL",
        "PIL.Image",
        "yaml",
        "requests",
        "matplotlib",
        "matplotlib.backends.backend_agg",
        "albumentations",
        "numpy",
    ]
)

# ── excludes (torch / ultralytics are too large) ──────────────────────
excludes = ["torch", "torchvision", "torchaudio", "ultralytics", "tkinter"]

# ── analysis ───────────────────────────────────────────────────────────
a = Analysis(
    [os.path.join(SRC_DIR, "ez_training", "main.py")],
    pathex=[SRC_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ez_training",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ez_training",
)
