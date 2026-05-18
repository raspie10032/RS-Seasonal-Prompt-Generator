# This code was created by ChatGPT.
# 이 코드는 ChatGPT를 사용하여 만들어졌습니다.

# __init__.py

__version__ = "2.6.0"

from .seasonal_fashion_prompt_node import SeasonalFashionPromptNode
from .gemma_tipo_node import GemmaTipoNode
from .gemma_tipo_vision_node import GemmaTipoVisionNode

NODE_CLASS_MAPPINGS = {
    "SeasonalFashionPromptNode": SeasonalFashionPromptNode,
    "GemmaTipoNode": GemmaTipoNode,
    "GemmaTipoVisionNode": GemmaTipoVisionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeasonalFashionPromptNode": "Seasonal Fashion Prompt Generator",
    "GemmaTipoNode": "Gemma TIPO Prompt → Tags",
    "GemmaTipoVisionNode": "Gemma TIPO Vision (Image → Tags)",
}
