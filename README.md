# Vision Analyzer Bench

![Vision Analyzer Demo](https://github.com/lesteroliver911/vision-analyzer-bench/blob/main/vision-bench-groq-ollama-gemini.gif)

A tool for benchmarking Vision LLMs (Ollama, Groq, Gemini) to accelerate your AI development pipeline.

## Why Use Vision Analyzer Bench?

- **Fast Integration**: Get vision capabilities in your app within minutes
- **Cost Optimization**: Compare performance across providers to choose the most cost-effective solution
- **Extensible**: Easy to add new providers or customize analysis parameters
- **Self-Hosted**: Full control over your data and processing

## Quick Start
```bash
git clone https://github.com/lesteroliver911/vision-analyzer-bench.git
cd vision-analyzer-bench
pip install -r requirements.txt
streamlit run app.py
```

## Performance Benchmarks

| Provider | Average Response Time | Notes |
|----------|---------------------|--------|
| Groq     | < 3 seconds        | Fastest inference, consistent performance |
| Gemini   | < 16 seconds       | Good balance of speed/quality |
| Ollama   | Performance varies | Highly dependent on local hardware* |

\* Looking for community benchmarks on different hardware setups. Please submit your results via PR!

## Structured Outputs

All analysis results are returned as validated Pydantic models:

```python
from vision_analyzer import analyze_image_groq, ImageAnalysis

# Pydantic model ensures consistent structure
class ImageAnalysis(BaseModel):
    description: str
    key_points: List[str]
    detected_objects: List[str]
    detected_text: Optional[str] = None

# Usage
analysis: ImageAnalysis = analyze_image_groq("image.jpg", groq_client)
print(f"Description: {analysis.description}")
print(f"Objects: {analysis.detected_objects}")

# JSON serialization included
json_output = analysis.model_dump_json()
```

Benefits:
- Type-safe outputs
- Automatic validation
- JSON serialization
- IDE autocompletion
- OpenAPI compatibility

## Key Features

🚀 **Performance Metrics**
- Real-time processing speed
- Token usage tracking
- Temperature impact analysis

🔄 **Provider Support**
- Local: Ollama (free, self-hosted)
- Cloud: Groq (fast inference)
- Cloud: Gemini (cost-effective)

## Integration Example

```python
from vision_analyzer import analyze_image_groq

analysis = analyze_image_groq("path/to/image.jpg", groq_client)
print(f"Objects detected: {analysis.detected_objects}")
```

## Roadmap

- [ ] Add Claude support
- [ ] Batch processing capabilities
- [ ] API endpoint mode
- [ ] Cost estimation feature

## License

MIT
