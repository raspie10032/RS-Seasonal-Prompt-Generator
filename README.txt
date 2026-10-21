RS-Seasonal-Prompt-Generator

Full usage / 전체 사용법:
  English: README.md
  한국어: README_KOR.md

Context expansion / 맥락 확장 — tags + natural-language instructions:
  PROMPT_EXPANSION.md
  examples/gemma-prompt-expand-native.api.json
  Optional personal Danbooru/e621 vocabulary; Korean augmentation, en/ja/ko fields.

Gemma TIPO Native (ComfyUI PyTorch; no separate llama.cpp CUDA wheel / 별도 llama.cpp CUDA 휠 불필요):
  NATIVE_TIPO.md
  examples/gemma-tipo-native.api.json (API graph)

Standalone VRS-INT / 독립 이미지·태그 기억:
  English: vrs_int/README.md
  한국어: vrs_int/README_KOR.md

Image/text pair examples / 이미지·텍스트 쌍 구성 예제:
  English: examples/vrs-int/README.md
  한국어: examples/vrs-int/README_KOR.md

VRS-INT ships code and fictional examples only. Prepare your own images,
metadata, and separate encoder. No starter memory or model weights are bundled.
VRS-INT는 코드와 가상 예제만 배포합니다. 이미지·태그·별도 인코더는 직접
준비합니다. 초기 기억과 모델 가중치는 포함하지 않습니다.

Originally created with ChatGPT, then developed with Claude Code and Codex.
처음 ChatGPT로 작성했고 이후 Claude Code와 Codex로 개발했습니다.

VRS inside ComfyUI / VRS 네이티브 구축·검색: vrs_int/COMFYUI_NATIVE.md
Folder/output collection / 폴더·output 수집과 시작 전 확인: vrs_int/FOLDERS.md
Lossless memory compression / 경험 무손실 압축: vrs_int/COMPRESSION.md
