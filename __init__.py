# This code was originally created with ChatGPT, and later substantially
# developed with Claude Code.
# 이 코드는 처음에 ChatGPT로 만들어졌고, 이후 Claude Code로 대폭
# 발전되었습니다.

# __init__.py

__version__ = "2.11.0"

from .seasonal_fashion_prompt_node import SeasonalFashionPromptNode
from .gemma_tipo_node import GemmaTipoNode
from .gemma_tipo_native_node import GemmaTipoNative
from .gemma_prompt_expand_node import GemmaPromptExpandNative
from .prompt_writer_node import RSPromptWriter
from .vrs_native_nodes import VRSNativeVisionLoader, VRSNativeMemoryLoader, VRSNativeBuild, VRSNativeRecall
from .vrs_folder_service import VRSFolderMemory, VRSFolderImportBatch, register_routes

NODE_CLASS_MAPPINGS = {
    "SeasonalFashionPromptNode": SeasonalFashionPromptNode,
    "GemmaTipoNode": GemmaTipoNode,
    "GemmaTipoNative": GemmaTipoNative,
    "GemmaPromptExpandNative": GemmaPromptExpandNative,
    "RSPromptWriter": RSPromptWriter,
    "VRSNativeVisionLoader": VRSNativeVisionLoader,
    "VRSNativeMemoryLoader": VRSNativeMemoryLoader,
    "VRSNativeBuild": VRSNativeBuild,
    "VRSNativeRecall": VRSNativeRecall,
    "VRSFolderMemory": VRSFolderMemory,
    "VRSFolderImportBatch": VRSFolderImportBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeasonalFashionPromptNode": "Seasonal Fashion Prompt Generator",
    "GemmaTipoNode": "Gemma TIPO (Prompt / Image → Tags)",
    "GemmaTipoNative": "Gemma TIPO Native (ComfyUI / Image → Tags)",
    "GemmaPromptExpandNative": "Gemma Prompt Expander Native (Tags / Natural Language)",
    "RSPromptWriter": "RS AI Prompt Writer (Korean / English → Prompt)",
    "VRSNativeVisionLoader": "VRS Native Vision Loader (Gemma FP16)",
    "VRSNativeMemoryLoader": "VRS Native Memory Loader",
    "VRSNativeBuild": "VRS Native Build / Update Experience",
    "VRSNativeRecall": "VRS Native Image Recall → Generation State",
    "VRSFolderMemory": "VRS Folder Memory (Latest Completed)",
    "VRSFolderImportBatch": "VRS Folder Import (Internal Batch)",
}

WEB_DIRECTORY = "./web"
register_routes()
