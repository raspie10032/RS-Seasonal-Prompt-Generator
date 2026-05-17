# This code was created by ChatGPT.
# 이 코드는 ChatGPT를 사용하여 만들어졌습니다.

# __init__.py

__version__ = "2.0.0"

from .seasonal_fashion_prompt_node import SeasonalFashionPromptNode

NODE_CLASS_MAPPINGS = {
    "SeasonalFashionPromptNode": SeasonalFashionPromptNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeasonalFashionPromptNode": "Seasonal Fashion Prompt Generator"
}
