Teddy Lee님의 코드를 참고하였습니다.

[![데모 영상](https://img.youtube.com/vi/VkcaigvTrug/0.jpg)](https://youtu.be/VkcaigvTrug)


## HuggingFace gguf 파일을 Ollama 로딩



### Modelfile

> EEVE-Korean-Instruct-10.8B-v1.0 예시
```
FROM ggml-model-Q5_K_M.gguf

TEMPLATE """{{- if .System }}
<s>{{ .System }}</s>
{{- end }}
<s>Human:
{{ .Prompt }}</s>
<s>Assistant:
"""

SYSTEM """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""

PARAMETER stop <s>
PARAMETER stop </s>
```

Ollama 모델 목록

```bash
ollama list
```

Ollama 모델 실행

```bash
ollama run EEVE-Korean-10.8B:latest
```

## LangServe 에서 Ollama 체인 생성

```bash
python server.py
```



## License

소스코드를 활용하실 때는 반드시 출처를 표기해 주시기 바랍니다.

```
MIT License

Copyright (c) 2024, 테디노트

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
