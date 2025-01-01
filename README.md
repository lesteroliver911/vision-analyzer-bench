# Vision Analyzer Bench

![Vision Analyzer Demo](https://raw.githubusercontent.com/lesteroliver911/vision-analyzer-bench/main/demo.gif)

A production-ready tool for benchmarking Vision LLMs (Ollama, Groq, Gemini) to accelerate your AI development pipeline.

## Why Use Vision Analyzer Bench?

- **Fast Integration**: Get vision capabilities in your app within minutes
- **Cost Optimization**: Compare performance across providers to choose the most cost-effective solution
- **Production Ready**: Built with Pydantic models and proper error handling
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
