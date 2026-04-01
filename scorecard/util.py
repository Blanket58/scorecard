import platform
import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import fontManager

EPS = np.finfo(np.float64).eps


def set_chinese_font():
    system = platform.system()
    candidates_font = {
        "Windows": ["Microsoft YaHei", "SimSun", "SimHei"],
        "Darwin": ["PingFang SC", "Hiragino Sans GB"],
        "Linux": ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei"],
    }
    available_fonts = [x.name for x in fontManager.ttflist]
    selected_font = None
    for font in candidates_font.get(system, []):
        if font in available_fonts:
            selected_font = font
            break
    if selected_font:
        plt.rcParams["font.sans-serif"] = selected_font
    else:
        warnings.warn(
            "Font not found. Chinese characters will be displayed as garbled text.",
            UserWarning,
            stacklevel=2,
        )
