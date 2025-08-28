# PDF to Audiobook Converter

A Python tool that converts PDF documents to audiobooks using text-to-speech (TTS) technology. The script extracts text from PDFs and generates high-quality audio using the XTTSv2 model with optional voice cloning.

## Features

- Extract text from PDF files using PyMuPDF
- Generate natural-sounding speech using Coqui TTS (XTTSv2)
- Voice cloning support with custom speaker samples
- Resume functionality for interrupted conversions
- Multiple output formats (MP3, WAV)
- GPU acceleration support (CUDA)
- Automatic text chunking for optimal processing

## Requirements

- Python 3.11 (recommended)
- CUDA-capable GPU (optional, but recommended for faster processing)
- FFmpeg (required for MP3 output)

## Installation

1. Create and activate a conda environment:
```bash
conda create --name tts_audiobook_env python=3.11
conda activate tts_audiobook_env
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch with CUDA support (if you have a compatible GPU):
```bash
# Replace cu118 with your CUDA version (e.g., cu121)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. Ensure FFmpeg is installed and accessible in your system PATH for MP3 export.

## Usage

With voice cloning (recommended and tested):
```bash
python pdf_to_audiobook.py --pdf_path "input/book1.pdf" --output_path "output/my_book.mp3" --language "en" --speaker_wav "speech1.wav"
```

Basic usage (without voice cloning - untested):
```bash
python pdf_to_audiobook.py --pdf_path "input/book1.pdf" --output_path "output/my_book.mp3" --language "en"
```

### Arguments

- `--pdf_path`: Path to the input PDF file (required)
- `--output_path`: Path for the output audiobook file (required)
- `--language`: Language code (e.g., 'en', 'es', 'de', 'lt') (required)
- `--speaker_wav`: Path to speaker WAV file for voice cloning (optional, 5-20s, 16kHz mono recommended)
- `--model_name`: TTS model name (default: "tts_models/multilingual/multi-dataset/xtts_v2")
- `--output_format`: Output format - "mp3" or "wav" (default: "mp3")

## Resume Functionality

The script automatically creates temporary chunk files and can resume interrupted conversions. If the process is stopped, simply run the same command again to continue from where it left off.

## Project Structure

```
pdf-to-audio/
├── pdf_to_audiobook.py    # Main conversion script
├── requirements.txt       # Python dependencies
├── LICENSE              # Project license
└── README.md           # This file
```

## Notes

- The script uses approximately 250 characters per chunk by default
- CUDA GPU acceleration significantly speeds up processing
- Temporary chunk files are stored in `*_temp_chunks/` directories
- The script includes comprehensive logging for monitoring progress
- Voice cloning works best with clear, 5-20 second audio samples

## Troubleshooting

- **CUDA out of memory**: Reduce `MAX_CHUNK_CHAR_LIMIT` in the script
- **MP3 export fails**: Ensure FFmpeg is installed and in your PATH
- **Missing dependencies**: Install all requirements from `requirements.txt`
- **Resume issues**: Check that temporary chunk directories haven't been deleted

## License

See LICENSE file for details.
