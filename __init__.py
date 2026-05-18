# This code was originally created with ChatGPT, and later substantially
# developed with Claude Code.
# 이 코드는 처음에 ChatGPT로 만들어졌고, 이후 Claude Code로 대폭
# 발전되었습니다.

# __init__.py

__version__ = "2.7.3"

from .seasonal_fashion_prompt_node import SeasonalFashionPromptNode
from .gemma_tipo_node import GemmaTipoNode

NODE_CLASS_MAPPINGS = {
    "SeasonalFashionPromptNode": SeasonalFashionPromptNode,
    "GemmaTipoNode": GemmaTipoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeasonalFashionPromptNode": "Seasonal Fashion Prompt Generator",
    "GemmaTipoNode": "Gemma TIPO (Prompt / Image → Tags)",
}
