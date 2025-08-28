#!/usr/bin/env python3
import fitz  # PyMuPDF
import torch
import os
import argparse
import logging
import re
from pathlib import Path
from pydub import AudioSegment
# Import TTS class using an alias to avoid potential name conflicts
from TTS.api import TTS as TtsApi
import time
import subprocess # For ffmpeg check
import types # For checking object types
from collections import Counter
from io import BytesIO
import unicodedata

# --- Configuration ---
# Max characters per chunk. XTTS v2 limit is ~400 tokens.
MAX_CHUNK_CHAR_LIMIT = 250
MIN_SPLIT_PART_LEN = 50

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.StreamHandler() # Output logs to console
    ]
)

# --- TOP LEVEL CHECK (Diagnostic - can be commented out after confirming TTS works) ---
# logging.info("--- Running top-level TtsApi check ---")
# try:
#     logging.info(f"Type of TtsApi at top level: {type(TtsApi)}")
#     logging.info(f"Is TtsApi callable at top level? {callable(TtsApi)}")
# except Exception as e:
#     logging.error(f"Top-level TtsApi check FAILED: {e}", exc_info=True)
# logging.info("--- End top-level TtsApi check ---")
# --- END TOP LEVEL CHECK ---


# --- Helper Functions ---

def check_ocr_dependencies():
    """Check if OCR dependencies are available."""
    try:
        import pytesseract
        from PIL import Image
        return True, None
    except ImportError as e:
        return False, str(e)

def extract_text_with_ocr(page, confidence_threshold=60):
    """Extract text from a page using OCR as fallback."""
    try:
        import pytesseract
        from PIL import Image

        # Get page as image
        mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data))

        # Perform OCR with confidence data
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        # Filter out low-confidence text
        text_parts = []
        for i, conf in enumerate(ocr_data['conf']):
            if int(conf) > confidence_threshold:
                text = ocr_data['text'][i].strip()
                if text:
                    text_parts.append(text)

        return ' '.join(text_parts)
    except Exception as e:
        logging.warning(f"OCR failed: {e}")
        return ""

def clean_text(text):
    """Clean and normalize extracted text."""
    if not text:
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Fix common OCR/extraction issues
    text = re.sub(r'([a-z])-\s*\n\s*([a-z])', r'\1\2', text)  # Fix hyphenated words
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Normalize paragraph breaks

    # Remove page numbers (common patterns)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone numbers
    text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)

    # Remove common headers/footers (basic patterns)
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # Skip very short lines that are likely page numbers or artifacts
        if len(line) < 3 and line.isdigit():
            continue
        # Skip lines that are mostly special characters
        if len(line) > 0 and len(re.sub(r'[^\w\s]', '', line)) / len(line) < 0.3:
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()

def detect_text_quality(text, min_word_length=3, min_alpha_ratio=0.7):
    """Detect if extracted text quality is good enough."""
    if not text or len(text) < 50:
        return False, "Text too short"

    words = text.split()
    if len(words) < 10:
        return False, "Too few words"

    # Check for reasonable word lengths
    avg_word_length = sum(len(word) for word in words) / len(words)
    if avg_word_length < min_word_length:
        return False, "Average word length too short"

    # Check alphabetic character ratio
    alpha_chars = sum(1 for char in text if char.isalpha())
    total_chars = len(text.replace(' ', ''))
    if total_chars > 0:
        alpha_ratio = alpha_chars / total_chars
        if alpha_ratio < min_alpha_ratio:
            return False, f"Too few alphabetic characters ({alpha_ratio:.2f})"

    return True, "Text quality acceptable"

def extract_text_structured(page):
    """Extract text using structured approach to maintain reading order."""
    try:
        # Get structured text data
        text_dict = page.get_text("dict")

        blocks = []
        for block in text_dict["blocks"]:
            if "lines" in block:  # Text block
                block_text = ""
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    block_text += line_text + "\n"

                if block_text.strip():
                    # Store block with its position for sorting
                    bbox = block["bbox"]
                    blocks.append({
                        'text': block_text.strip(),
                        'y': bbox[1],  # Top y-coordinate
                        'x': bbox[0],  # Left x-coordinate
                    })

        # Sort blocks by position (top to bottom, left to right)
        blocks.sort(key=lambda b: (b['y'], b['x']))

        # Combine sorted blocks
        return '\n\n'.join(block['text'] for block in blocks)
    except Exception as e:
        logging.warning(f"Structured text extraction failed: {e}")
        return page.get_text("text")

def extract_text_robust(pdf_path):
    """
    Robust PDF text extraction with multiple fallback methods.
    """
    logging.info(f"Starting robust text extraction from: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        logging.info(f"PDF opened successfully. Pages: {doc.page_count}")

        # Check if PDF is password protected
        if doc.needs_pass:
            doc.close()
            raise Exception("PDF is password protected")

        full_text = ""
        ocr_used = False
        low_quality_pages = 0

        # Check OCR availability once
        ocr_available, ocr_error = check_ocr_dependencies()
        if not ocr_available:
            logging.warning(f"OCR not available: {ocr_error}. Install with: pip install pytesseract pillow")

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_text = ""

            # Method 1: Try structured text extraction
            try:
                page_text = extract_text_structured(page)
            except Exception as e:
                logging.warning(f"Structured extraction failed for page {page_num + 1}: {e}")
                # Fallback to simple text extraction
                page_text = page.get_text("text")

            # Check text quality
            is_good_quality, quality_reason = detect_text_quality(page_text)

            # Method 2: Use OCR if text quality is poor and OCR is available
            if not is_good_quality and ocr_available:
                logging.info(f"Page {page_num + 1}: Text quality poor ({quality_reason}), trying OCR...")
                ocr_text = extract_text_with_ocr(page)

                # Compare OCR result quality
                ocr_is_good, _ = detect_text_quality(ocr_text)
                if ocr_is_good and len(ocr_text) > len(page_text):
                    page_text = ocr_text
                    ocr_used = True
                    logging.info(f"Page {page_num + 1}: Using OCR result")
                else:
                    low_quality_pages += 1
                    logging.warning(f"Page {page_num + 1}: Both extraction methods produced poor quality text")
            elif not is_good_quality:
                low_quality_pages += 1
                logging.warning(f"Page {page_num + 1}: Poor text quality ({quality_reason}) and OCR not available")

            # Clean the extracted text
            page_text = clean_text(page_text)

            if page_text:
                full_text += page_text + "\n\n"

            # Progress logging
            if (page_num + 1) % 50 == 0 or (page_num + 1) == doc.page_count:
                logging.info(f"Processed {page_num + 1}/{doc.page_count} pages...")

        doc.close()

        # Final text cleaning
        full_text = clean_text(full_text)

        # Summary logging
        if ocr_used:
            logging.info("OCR was used for some pages")
        if low_quality_pages > 0:
            logging.warning(f"{low_quality_pages} pages had low quality text extraction")

        # Final quality check
        if not full_text.strip():
            raise Exception("No text could be extracted from PDF")

        final_quality, final_reason = detect_text_quality(full_text)
        if not final_quality:
            logging.warning(f"Overall text quality warning: {final_reason}")

        logging.info(f"Text extraction completed. Total characters: {len(full_text)}")
        return full_text

    except Exception as e:
        logging.error(f"PDF text extraction failed: {e}")
        raise

def split_text_into_chunks(text):
    """
    Splits text into manageable chunks, aiming for a character limit
    while trying to respect sentence boundaries and word boundaries.
    """
    logging.info(f"Splitting text into chunks aiming for ~{MAX_CHUNK_CHAR_LIMIT} characters...")
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        logging.warning("No sentences found after initial punctuation split.")
        paragraphs = re.split(r'\s*\n\s*\n\s*', text)
        sentences = [p.strip() for p in paragraphs if p.strip()]
        if not sentences:
             logging.warning("No paragraphs found. Fallback 2: Splitting by single newline.")
             lines = re.split(r'\s*\n\s*', text)
             sentences = [l.strip() for l in lines if l.strip()]
             if not sentences:
                  logging.warning("No newlines found. Treating entire text as one block (will be split if too long).")
                  sentences = [text] if text else []

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(sentence) > MAX_CHUNK_CHAR_LIMIT:
            logging.warning(f"Sentence exceeds limit ({len(sentence)} > {MAX_CHUNK_CHAR_LIMIT}). Splitting it without breaking words...")
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            start = 0
            while start < len(sentence):
                end = min(start + MAX_CHUNK_CHAR_LIMIT, len(sentence))
                if end < len(sentence):
                    split_pos = sentence.rfind(' ', start, end)
                    if split_pos != -1 and split_pos > start and (split_pos - start) >= MIN_SPLIT_PART_LEN:
                         end = split_pos
                part = sentence[start:end].strip()
                if part:
                    chunks.append(part)
                start = end
            continue

        if current_chunk and len(current_chunk) + len(sentence) + 1 > MAX_CHUNK_CHAR_LIMIT:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    chunks = [c for c in chunks if c]
    logging.info(f"Refined splitting into {len(chunks)} chunks based on character limit approx {MAX_CHUNK_CHAR_LIMIT} and word boundaries.")
    return chunks

def create_output_dirs(output_path):
    """Creates output directory and a persistent temporary directory for chunks based on output filename."""
    try:
        output_file = Path(output_path)
        output_dir = output_file.parent
        # Create persistent temp dir name based on output filename stem
        temp_dir_name = f"{output_file.stem}_temp_chunks"
        temp_dir = output_dir / temp_dir_name # Changed naming scheme

        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True) # exist_ok=True allows resuming
        logging.info(f"Ensured output directory exists: {output_dir}")
        logging.info(f"Using persistent temporary directory for chunks: {temp_dir}")
        return temp_dir
    except Exception as e:
        logging.error(f"Failed to create directories: {e}")
        raise

def combine_audio_chunks(temp_dir, output_path, output_format="mp3"):
    """Combines WAV audio chunks from temp_dir into a single output file."""
    logging.info(f"Combining audio chunks from {temp_dir}...")
    # Ensure chunks are sorted numerically (00001, 00002, ...)
    chunk_files = sorted(temp_dir.glob("chunk_*.wav"))
    if not chunk_files:
        logging.warning("No audio chunks found to combine in temporary directory.")
        return False

    combined = AudioSegment.empty()
    logging.info(f"Found {len(chunk_files)} chunks to combine.")
    for i, chunk_file in enumerate(chunk_files):
        try:
            sound = AudioSegment.from_wav(chunk_file)
            combined += sound
        except Exception as e:
            # Log which file caused the error
            logging.error(f"Error loading or combining chunk {chunk_file.name}: {e}")
            return False # Indicate failure

    try:
        logging.info(f"Exporting combined audio to {output_path} (format: {output_format})...")
        combined.export(output_path, format=output_format, bitrate="192k")
        logging.info("Audio export successful.")
        return True
    except Exception as e:
        logging.error(f"Error exporting combined audio file: {e}")
        if output_format == "mp3":
            logging.error("Ensure ffmpeg or libav is installed and accessible in your system PATH for MP3 export.")
        return False

def cleanup_temp_dir(temp_dir):
    """Removes temporary audio chunk files and directory."""
    logging.info(f"Cleaning up temporary directory: {temp_dir}")
    try:
        if not temp_dir.exists():
            logging.warning(f"Temporary directory {temp_dir} not found for cleanup.")
            return
        n_deleted = 0
        for item in temp_dir.iterdir():
            if item.is_file() and item.name.startswith("chunk_") and item.name.endswith(".wav"):
                item.unlink()
                n_deleted +=1
        logging.info(f"Deleted {n_deleted} chunk files.")
        # Check if directory is empty before removing (might contain other files?)
        if not any(temp_dir.iterdir()):
             temp_dir.rmdir()
             logging.info("Temporary directory removed.")
        else:
             logging.warning(f"Temporary directory {temp_dir} not empty after deleting chunks, not removing directory.")
    except Exception as e:
        logging.error(f"Could not fully clean up temporary directory {temp_dir}: {e}")

def check_ffmpeg():
    """Checks if ffmpeg command is accessible."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, universal_newlines=True)
        if result.returncode == 0:
            logging.info("ffmpeg found.")
            return True
        else:
            logging.warning(f"ffmpeg command check failed (return code {result.returncode}). MP3 export might fail.")
            return False
    except FileNotFoundError:
        logging.warning("ffmpeg command not found. MP3 export will likely fail. Ensure ffmpeg is installed and in your system's PATH.")
        return False
    except Exception as e:
        logging.warning(f"An unexpected error occurred while checking for ffmpeg: {e}")
        return False

# --- Main Function ---

def pdf_to_audiobook(args):
    """Main function to handle PDF extraction, TTS, and audio combination with resume support."""
    overall_start_time = time.time()
    logging.info("="*50)
    logging.info("Starting PDF to Audiobook Conversion Process")
    logging.info("="*50)
    logging.info(f"Arguments received: PDF='{args.pdf_path}', Output='{args.output_path}', Lang='{args.language}', SpeakerWAV='{args.speaker_wav}', Format='{args.output_format}'")

    pdf_path = Path(args.pdf_path)
    output_path = Path(args.output_path)
    speaker_wav = Path(args.speaker_wav) if args.speaker_wav else None
    language = args.language.lower()
    model_name = args.model_name
    output_format = args.output_format.lower()
    temp_dir = None
    combine_success = False # Initialize defensively
    chunk_generation_success_count = 0 # Initialize here for broader scope
    total_chunks = 0 # Initialize here
    files_resumed = 0 # Initialize here

    try:
        # --- Validate Inputs ---
        if not pdf_path.is_file(): logging.error(f"Input PDF file not found: {pdf_path}"); return
        if speaker_wav and not speaker_wav.is_file(): logging.error(f"Speaker WAV file not found: {speaker_wav}"); return
        if output_format not in ["mp3", "wav"]: logging.error(f"Unsupported output format: {output_format}."); return
        if output_format == "mp3": check_ffmpeg()

        # --- Setup Directories (Persistent Temp Dir) ---
        temp_dir = create_output_dirs(output_path)

        # --- 1. Text Extraction ---
        section_start_time = time.time()
        logging.info(f"--- Step 1: Extracting Text from PDF: {pdf_path} ---")
        try:
            full_text = extract_text_robust(pdf_path)
            logging.info(f"Finished extracting text. Total time: {time.time() - section_start_time:.2f}s")
        except Exception as e:
            logging.error(f"Failed to extract text from PDF: {e}");
            return
        if not full_text.strip():
            logging.error("Extracted text is empty.");
            return

        # --- 2. Text Chunking ---
        section_start_time = time.time()
        logging.info("--- Step 2: Chunking Text ---")
        text_chunks = split_text_into_chunks(full_text)
        total_chunks = len(text_chunks) # Set total chunks here
        if not text_chunks: logging.error("Failed to split text into chunks."); return
        logging.info(f"Text chunking finished. Total time: {time.time() - section_start_time:.2f}s")

        # --- 3. TTS Initialization ---
        # (Moved TTS init before the loop to avoid re-initializing if resuming)
        section_start_time = time.time()
        logging.info("--- Step 3: Initializing Text-to-Speech Engine ---")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Attempting to use device: {device}")
        if device == "cpu": logging.warning("CUDA not available. TTS will run on CPU (slower).")
        tts = None # Initialize tts variable
        try:
            tts_instance = TtsApi(model_name=model_name, progress_bar=True)
            tts_instance.to(device)
            logging.info(f"TTS model '{model_name}' loaded and moved to device '{device}'.")
            tts = tts_instance
            speaker_args = {}
            if speaker_wav:
                logging.info(f"Using voice cloning with speaker WAV: {speaker_wav}")
                speaker_args['speaker_wav'] = str(speaker_wav)
            else:
                logging.info("No speaker WAV provided. Using model's default voice.")
            if hasattr(tts, 'languages') and tts.languages:
                 if language not in tts.languages: logging.warning(f"Language '{language}' not explicitly listed: {tts.languages}.")
                 else: logging.info(f"Language '{language}' confirmed supported.")
            else: logging.info("Model language list N/A.")
            logging.info(f"TTS initialization finished. Total time: {time.time() - section_start_time:.2f}s")
        except Exception as e:
            logging.error(f"Failed to initialize TTS model: {e}", exc_info=True)
            return

        # --- 4. Audio Generation (with Resume) ---
        section_start_time = time.time()
        logging.info("--- Step 4: Generating Audio Chunks ---")
        # chunk_generation_success_count = 0 # Moved initialization up
        # files_resumed = 0 # Moved initialization up
        estimated_total_tts_time = 0

        for i, chunk in enumerate(text_chunks):
            chunk_filename = temp_dir / f"chunk_{i+1:05d}.wav"

            # --- Resume Check ---
            if chunk_filename.exists() and chunk_filename.stat().st_size > 0:
                # Check if file exists and is not empty
                if i == 0 or (i + 1) % 50 == 0 or (i+1) == total_chunks : # Log skipping periodically
                     logging.info(f"Skipping chunk {i+1}/{total_chunks}: Output file already exists (resuming).")
                files_resumed += 1
                chunk_generation_success_count += 1 # Count resumed as success
                continue # Move to the next chunk
            # --- End Resume Check ---

            # Only generate if file doesn't exist or is empty
            chunk_start_time = time.time()
            logging.info(f"Generating audio for chunk {i+1}/{total_chunks} ({len(chunk)} chars) -> {chunk_filename.name}")

            if not chunk.strip(): logging.warning(f"Skipping empty chunk {i+1}."); continue

            try:
                tts.tts_to_file( text=chunk, file_path=str(chunk_filename), language=language, **speaker_args)
                chunk_end_time = time.time(); chunk_duration = chunk_end_time - chunk_start_time
                estimated_total_tts_time += chunk_duration
                # Calculate estimated remaining time based only on *newly generated* chunks
                processed_new_chunks = (i + 1) - files_resumed
                if processed_new_chunks > 0:
                     avg_time_per_new_chunk = estimated_total_tts_time / processed_new_chunks
                     remaining_chunks_to_generate = total_chunks - chunk_generation_success_count
                     estimated_remaining_time = avg_time_per_new_chunk * remaining_chunks_to_generate
                     logging.info(f"Successfully generated chunk {i+1}/{total_chunks} (took {chunk_duration:.2f}s). Estimated time remaining for new chunks: {estimated_remaining_time:.2f}s")
                else: # Only resumed so far
                     logging.info(f"Successfully generated chunk {i+1}/{total_chunks} (took {chunk_duration:.2f}s).")

                chunk_generation_success_count += 1
            except RuntimeError as e:
                if "CUDA out of memory" in str(e): logging.error(f"CUDA OOM error chunk {i+1}! Reduce MAX_CHUNK_CHAR_LIMIT?")
                else: logging.error(f"Runtime error chunk {i+1}: {e}")
                logging.error(f"Failed chunk text: '{chunk[:100]}...'"); logging.error("Stopping generation."); break
            except Exception as e:
                logging.error(f"General error chunk {i+1}: {e}", exc_info=True)
                logging.error(f"Failed chunk text: '{chunk[:100]}...'"); logging.error("Stopping generation."); break

        logging.info(f"Audio generation loop finished. {chunk_generation_success_count}/{total_chunks} chunks successful ({files_resumed} resumed). Total TTS time for new chunks: {estimated_total_tts_time:.2f}s")

        # --- 5. Combine Audio ---
        section_start_time = time.time()
        logging.info("--- Step 5: Combining Audio Chunks ---")
        # Only attempt combining if *all* chunks are marked as successful (generated or resumed)
        if chunk_generation_success_count == total_chunks:
            combine_success = combine_audio_chunks(temp_dir, output_path, output_format)
            logging.info(f"Audio combination process finished. Success: {combine_success}. Total time: {time.time() - section_start_time:.2f}s")

            # --- 6. Cleanup ---
            # Only cleanup if combination was successful
            if combine_success:
                 section_start_time = time.time()
                 logging.info("--- Step 6: Cleaning Up Temporary Files ---")
                 cleanup_temp_dir(temp_dir)
                 logging.info(f"Cleanup finished. Total time: {time.time() - section_start_time:.2f}s")
            else:
                 logging.error(f"Audio combination failed. Temporary files kept in {temp_dir} for inspection.")
        else:
             logging.warning(f"Not all chunks ({chunk_generation_success_count}/{total_chunks}) were processed successfully. Skipping combination and cleanup. Re-run the script to resume.")
             combine_success = False # Ensure this reflects the state

        # --- Final Summary ---
        overall_end_time = time.time()
        logging.info("="*50)
        if combine_success and chunk_generation_success_count == total_chunks:
            logging.info(f"Audiobook conversion process completed successfully!")
            logging.info(f"Final audiobook saved to: {output_path}")
        else:
            logging.error("Audiobook conversion process finished with errors or was incomplete.")
            if chunk_generation_success_count != total_chunks: logging.error(f"=> Only {chunk_generation_success_count}/{total_chunks} audio chunks were ready.")
            if not combine_success and chunk_generation_success_count == total_chunks : logging.error("=> Audio combination failed.") # Only log combination fail if all chunks were theoretically ready
        logging.info(f"Total execution time for this run: {overall_end_time - overall_start_time:.2f} seconds.")
        logging.info("="*50)

    except Exception as e:
        logging.error(f"An unexpected critical error occurred: {e}", exc_info=True)
    finally:
        logging.info("Script execution finished or exited.")
        # No automatic cleanup in finally - relies on successful combination step


# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a PDF file (non-scanned) to an audiobook using local TTS (XTTSv2 default) with resume support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Arguments remain the same
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to the input PDF file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the final audiobook file (e.g., output/mybook.mp3). Temp files stored nearby.")
    parser.add_argument("--speaker_wav", type=str, default=None, help="Path to a speaker WAV file (~5-20s, 16kHz mono) for voice cloning. Uses model default if omitted.")
    parser.add_argument("--language", type=str, required=True, help="Language code for the text/speech (e.g., 'en', 'es', 'de', 'lt').")
    parser.add_argument("--model_name", type=str, default="tts_models/multilingual/multi-dataset/xtts_v2", help="Name of the TTS model to use.")
    parser.add_argument("--output_format", type=str, default="mp3", choices=["mp3", "wav"], help="Output audio format. MP3 requires ffmpeg.")

    # Dependency Check
    missing_deps = []
    try: import TTS
    except ImportError: missing_deps.append("TTS (pip install TTS)")
    try: import pydub
    except ImportError: missing_deps.append("pydub (pip install pydub)")
    try: import fitz
    except ImportError: missing_deps.append("PyMuPDF (pip install PyMuPDF)")
    try: import torch
    except ImportError: missing_deps.append("torch (install via pytorch.org)")

    if missing_deps:
        print("\nERROR: Missing required Python libraries:"); [print(f" - {dep}") for dep in missing_deps]; print("Install them first."); exit(1)

    # Check optional OCR dependencies
    optional_deps = []
    try: import pytesseract
    except ImportError: optional_deps.append("pytesseract (pip install pytesseract)")
    try: from PIL import Image
    except ImportError: optional_deps.append("Pillow (pip install Pillow)")

    if optional_deps:
        print("\nINFO: Optional OCR dependencies not found (for scanned PDFs):"); [print(f" - {dep}") for dep in optional_deps]
        print("OCR functionality will be disabled. Install these for better scanned PDF support.\n")

    args = parser.parse_args()
    pdf_to_audiobook(args)
