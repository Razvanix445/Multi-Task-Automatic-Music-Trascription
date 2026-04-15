#!/usr/bin/env python3
import os
import uuid
import json
import time
import tempfile
import threading
from datetime import datetime
from pathlib import Path
import atexit

import numpy as np
import tensorflow as tf
import librosa
from flask import Flask, request, jsonify, send_file, g
from flask_cors import CORS
import mido
from pydub import AudioSegment
import boto3
from werkzeug.utils import secure_filename
import logging
import gc
import subprocess
import shutil
from scipy import ndimage
from scipy.signal import find_peaks

from typing import List, Tuple, Dict, Any

# Import your existing modules
from models.model_loader import ModelLoader
from utils.utils import weighted_binary_crossentropy, focal_loss, F1Score
from models.architecture import acoustic_feature_extractor, vertical_dependencies_layer, lstm_with_attention, \
    onset_subnetwork, frame_subnetwork, offset_subnetwork, velocity_subnetwork, build_model
from postprocessing.postprocessing import MusicTranscriptionPostprocessor

# Configure logging for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print(f"Python executable: {os.sys.executable}")

# =============================================================================
# CONFIGURATION & SETUP - ALL ORIGINAL SETTINGS
# =============================================================================

# Use absolute paths for directories
UPLOAD_FOLDER = '/app/uploads'
OUTPUT_FOLDER = '/app/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# AWS S3 Configuration - EXACTLY as original
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_BUCKET_NAME = "flutter-audio-uploads"
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")

s3_client = None
if AWS_ACCESS_KEY and AWS_SECRET_KEY:
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        logger.info("✅ S3 client initialized")
    except Exception as e:
        logger.warning(f"⚠️ S3 initialization failed: {e}")


# =============================================================================
# ENHANCED STARTUP SEQUENCE - ALL ORIGINAL FUNCTIONALITY
# =============================================================================

def setup_virtual_display():
    """Set up virtual display for MuseScore in headless environment"""
    display = ':99'

    try:
        os.environ['DISPLAY'] = display

        try:
            result = subprocess.run(['pgrep', 'Xvfb'], capture_output=True)
            if result.returncode != 0:
                logger.info("🖥️ Starting virtual display...")
                xvfb_process = subprocess.Popen([
                    'Xvfb', display,
                    '-screen', '0', '1024x768x24',
                    '-ac', '+extension', 'GLX'
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                time.sleep(2)
                atexit.register(lambda: xvfb_process.terminate())
                logger.info(f"✅ Virtual display started: {display}")
            else:
                logger.info(f"✅ Virtual display already running: {display}")

        except Exception as e:
            logger.warning(f"⚠️ Virtual display setup warning: {e}")

    except Exception as e:
        logger.error(f"❌ Display setup error: {e}")

    return display


def comprehensive_musescore_test():
    """Complete MuseScore test including conversion - ORIGINAL FUNCTION"""
    logger.info("🎼 Running comprehensive MuseScore test...")

    commands_to_test = ['musescore3', 'musescore', 'mscore3', 'mscore']
    working_command = None

    for cmd in commands_to_test:
        try:
            result = subprocess.run([cmd, '--version'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                working_command = cmd
                logger.info(f"✅ {cmd} version check passed: {result.stdout.strip()}")
                break
        except Exception as e:
            logger.debug(f"   {cmd}: {e}")

    if not working_command:
        return False, "No working MuseScore command found"

    # Test conversion with minimal MIDI
    try:
        test_midi = os.path.join(OUTPUT_FOLDER, 'test_minimal.mid')
        minimal_midi_bytes = bytes([
            0x4D, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,  # MThd header
            0x00, 0x00, 0x00, 0x01, 0x00, 0x60,  # Format 0, 1 track, 96 tpqn
            0x4D, 0x54, 0x72, 0x6B, 0x00, 0x00, 0x00, 0x0B,  # MTrk header
            0x00, 0x90, 0x40, 0x40,  # Note on C4
            0x48, 0x80, 0x40, 0x40,  # Note off C4
            0x00, 0xFF, 0x2F, 0x00  # End of track
        ])

        with open(test_midi, 'wb') as f:
            f.write(minimal_midi_bytes)

        test_xml = os.path.join(OUTPUT_FOLDER, 'test_output.musicxml')
        result = subprocess.run([working_command, '-o', test_xml, test_midi],
                                capture_output=True, text=True, timeout=20)

        if result.returncode == 0 and os.path.exists(test_xml):
            logger.info("✅ MusicXML conversion test passed")

            test_pdf = os.path.join(OUTPUT_FOLDER, 'test_output.pdf')
            result = subprocess.run([working_command, '-o', test_pdf, test_xml],
                                    capture_output=True, text=True, timeout=20)

            if result.returncode == 0 and os.path.exists(test_pdf):
                logger.info("✅ PDF conversion test passed")
                conversion_success = True
            else:
                logger.warning(f"⚠️ PDF conversion failed: {result.stderr}")
                conversion_success = False

            # Clean up test files
            for test_file in [test_midi, test_xml, test_pdf]:
                if os.path.exists(test_file):
                    os.remove(test_file)

            return conversion_success, working_command
        else:
            logger.error(f"❌ MusicXML conversion failed: {result.stderr}")
            return False, f"{working_command} conversion failed"

    except Exception as e:
        logger.error(f"❌ Conversion test error: {e}")
        return False, str(e)


# Run startup sequence
logger.info("=" * 60)
logger.info("🚀 STARTING WAVE2NOTES WITH MUSESCORE SUPPORT")
logger.info("=" * 60)

display = setup_virtual_display()
conversion_works, musescore_status = comprehensive_musescore_test()

if conversion_works:
    logger.info(f"🎼 ✅ MuseScore fully operational: {musescore_status}")
    logger.info("   📄 Sheet music generation enabled")
else:
    logger.info(f"🎼 ❌ MuseScore issues: {musescore_status}")
    logger.info("   📄 Sheet music generation disabled")

# Verify directories
logger.info("📁 Checking directories...")
for folder_name, folder_path in [("Upload", UPLOAD_FOLDER), ("Output", OUTPUT_FOLDER)]:
    try:
        os.makedirs(folder_path, mode=0o755, exist_ok=True)
        if os.access(folder_path, os.W_OK):
            logger.info(f"✅ {folder_name} folder ready: {folder_path}")
        else:
            logger.warning(f"⚠️ {folder_name} folder not writable: {folder_path}")
    except Exception as e:
        logger.error(f"❌ {folder_name} folder error: {e}")

logger.info("=" * 60)
logger.info("🎵 Ready for piano transcription!")
logger.info("=" * 60)


# =============================================================================
# IMPROVED MODEL LOADER WRAPPER
# =============================================================================

class StableModelLoader:
    """Wrapper around your existing ModelLoader for stability"""

    def __init__(self):
        self.model_loader = ModelLoader()
        self.lock = threading.Lock()
        self.last_reset_time = 0
        self.reset_cooldown = 30  # 30 seconds between resets

    def get_model(self):
        """Get model with stability improvements"""
        with self.lock:
            try:
                return self.model_loader.get_model()
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                # Only reset if cooldown period has passed
                current_time = time.time()
                if current_time - self.last_reset_time > self.reset_cooldown:
                    logger.info("Attempting model reset after cooldown")
                    try:
                        self.model_loader.reset()
                        self.last_reset_time = current_time
                        return self.model_loader.get_model()
                    except Exception as reset_error:
                        logger.error(f"Model reset failed: {reset_error}")
                        raise
                else:
                    logger.warning("Reset attempted too soon, using cooldown")
                    raise

    def is_model_ready(self):
        """Check model status safely"""
        try:
            return self.model_loader.is_model_ready()
        except Exception:
            return False

    def safe_reset(self):
        """Safe reset with cooldown"""
        with self.lock:
            current_time = time.time()
            if current_time - self.last_reset_time > self.reset_cooldown:
                try:
                    self.model_loader.reset()
                    self.last_reset_time = current_time
                    logger.info("🔄 Model reset completed")
                    return True
                except Exception as e:
                    logger.error(f"Reset failed: {e}")
                    return False
            else:
                logger.warning("Reset blocked by cooldown")
                return False


# Initialize stable model loader
stable_model_loader = StableModelLoader()
logger.info("Model loader initialized - model will load on first recordings endpoint request")


# =============================================================================
# ALL ORIGINAL UTILITY FUNCTIONS - PRESERVED
# =============================================================================

def detect_request_platform(request):
    """Detect if request is from web browser or mobile app - ORIGINAL"""
    user_agent = request.headers.get('User-Agent', '').lower()

    is_web = any(browser in user_agent for browser in [
        'mozilla', 'chrome', 'safari', 'firefox', 'edge', 'webkit'
    ])

    is_mobile_app = 'flutter' in user_agent or 'dart' in user_agent

    return {
        'is_web': is_web,
        'is_mobile_app': is_mobile_app,
        'user_agent': user_agent
    }


def safe_file_processing(file, platform_type):
    """Safely process files based on platform - ORIGINAL"""
    try:
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"

        if platform_type == 'web':
            audio_path = os.path.join(UPLOAD_FOLDER, f"web_{unique_filename}")
        else:
            audio_path = os.path.join(UPLOAD_FOLDER, f"mobile_{unique_filename}")

        file.save(audio_path)

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise Exception(f"File not saved properly: {audio_path}")

        logger.info(f"✅ File saved safely: {audio_path}")
        return audio_path

    except Exception as e:
        logger.error(f"❌ File processing error: {e}")
        raise


def check_musescore_with_display():
    """Check MuseScore with proper display setup - ORIGINAL"""
    try:
        if not os.environ.get('DISPLAY'):
            os.environ['DISPLAY'] = ':99'

        try:
            subprocess.run(['Xvfb', ':99', '-screen', '0', '1024x768x24'],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           timeout=2)
        except:
            pass

        commands_to_test = ['musescore3', 'musescore', 'mscore3', 'mscore']

        for cmd in commands_to_test:
            try:
                result = subprocess.run([cmd, '--version'],
                                        capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"✅ {cmd} is available: {result.stdout.strip()}")
                    return True, cmd, result.stdout.strip()
            except Exception as e:
                logger.debug(f"   {cmd}: {e}")
                continue

        return False, None, "MuseScore not responding"

    except Exception as e:
        logger.error(f"❌ Display setup error: {e}")
        return False, None, str(e)


def check_musescore_installation():
    """Updated function that works with your Dockerfile setup - ORIGINAL"""
    return check_musescore_with_display()[:2]


def pitch_to_note_name(pitch):
    """Convert MIDI pitch number to note name - ORIGINAL"""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (pitch // 12) - 1
    note = note_names[pitch % 12]
    return f"{note}{octave}"


def clean_up_notes(notes_list, min_duration=0.05, merge_gap=0.08, confidence_threshold=0.4):
    """Filter and merge notes to improve MIDI quality - ORIGINAL"""
    filtered_notes = []
    for note in notes_list:
        if note["duration"] >= min_duration and note["velocity"] >= confidence_threshold:
            filtered_notes.append(note)

    filtered_notes.sort(key=lambda x: (x["pitch"], x["time"]))

    merged_notes = []
    i = 0
    while i < len(filtered_notes):
        current_note = filtered_notes[i]

        j = i + 1
        while j < len(filtered_notes) and filtered_notes[j]["pitch"] == current_note["pitch"]:
            next_note = filtered_notes[j]

            gap = next_note["time"] - (current_note["time"] + current_note["duration"])
            if gap <= merge_gap:
                current_note["duration"] = (next_note["time"] + next_note["duration"]) - current_note["time"]
                current_note["velocity"] = max(current_note["velocity"], next_note["velocity"])
                current_note["velocity_midi"] = max(current_note["velocity_midi"], next_note["velocity_midi"])
                j += 1
            else:
                break

        merged_notes.append(current_note)
        i = j

    return merged_notes


def extract_notes_from_predictions(predictions):
    """Enhanced note extraction using sophisticated postprocessing - ORIGINAL"""
    logger.info("🎼 Extracting notes with enhanced postprocessing...")

    postprocessor = MusicTranscriptionPostprocessor(
        onset_threshold=0.3,
        frame_threshold=0.3,
        min_note_duration=0.05,
        max_note_duration=8.0,
        time_resolution=0.032
    )

    refined_notes = postprocessor.process_predictions(predictions)
    return refined_notes


def print_detailed_notes(notes):
    """Print detailed information about detected notes for debugging - ORIGINAL"""
    logger.info("\n===== DETECTED NOTES (BACKEND) =====")
    logger.info(f"Total notes detected: {len(notes)}")

    if len(notes) > 0:
        pitches = [note['pitch'] for note in notes]
        times = [note['time'] for note in notes]
        durations = [note['duration'] for note in notes]
        velocities = [note['velocity'] for note in notes]

        logger.info(f"Pitch range: {min(pitches)} to {max(pitches)}")
        logger.info(f"Time range: {min(times):.2f}s to {max(times):.2f}s")
        logger.info(f"Duration range: {min(durations):.2f}s to {max(durations):.2f}s")
        logger.info(f"Velocity range: {min(velocities):.2f} to {max(velocities):.2f}")

        for i, note in enumerate(notes[:5]):  # Show first 5 notes
            logger.info(f"Note {i + 1}: name={note['note_name']}, time={note['time']:.3f}s, "
                        f"duration={note['duration']:.3f}s, velocity={note['velocity']:.2f}, "
                        f"pitch={note['pitch']}")

    logger.info("===== END OF NOTES =====\n")


def create_midi_from_notes(notes, output_path):
    """Create a MIDI file from the detected notes - ORIGINAL"""
    logger.info(f"Creating MIDI file with {len(notes)} notes")

    for i, note in enumerate(notes[:5]):
        logger.debug(f"Note {i}: time={note.get('time', 'N/A')}, "
                     f"duration={note.get('duration', 'N/A')}, "
                     f"pitch={note.get('pitch', 'N/A')}, "
                     f"velocity={note.get('velocity_midi', 'N/A')}")

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

    ticks_per_beat = 480
    tempo = 500000
    ticks_per_second = ticks_per_beat / (tempo / 1000000)

    notes = sorted(notes, key=lambda x: x['time'])
    events = []

    for note in notes:
        if note['time'] < 0 or note['duration'] <= 0:
            logger.debug(f"Skipping invalid note: time={note['time']}, duration={note['duration']}")
            continue

        onset_time_ticks = int(max(0, note['time'] * ticks_per_second))
        offset_time_ticks = onset_time_ticks + int(max(1, note['duration'] * ticks_per_second))

        velocity_raw = note['velocity_midi']
        if velocity_raw > 5:
            velocity_raw = 100

        velocity = max(0, min(127, velocity_raw))

        events.append((onset_time_ticks, 'note_on', note['pitch'], velocity))
        events.append((offset_time_ticks, 'note_off', note['pitch'], 0))

    events.sort()

    last_time = 0
    for abs_time, msg_type, pitch, velocity in events:
        delta_time = max(0, abs_time - last_time)

        if msg_type == 'note_on':
            track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=delta_time))
        else:
            track.append(mido.Message('note_off', note=pitch, velocity=velocity, time=delta_time))

        last_time = abs_time

    mid.save(output_path)
    return output_path


def extract_mel_spectrogram(audio_path, sr=16000, n_mels=229, hop_length=512, n_fft=2048):
    """Extract mel spectrogram - ORIGINAL"""
    y, _ = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec


def convert_audio_to_wav(input_path, output_path, sample_rate=16000):
    """Simple function to convert audio files to WAV format - ORIGINAL"""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(sample_rate)

        if audio.channels > 1:
            audio = audio.set_channels(1)

        audio.export(output_path, format="wav")
        logger.info(f"Converted {input_path} to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        return False


def convert_m4a_to_wav(input_path, output_path, sample_rate=16000):
    """Convert specifically M4A to WAV format - ORIGINAL"""
    try:
        audio = AudioSegment.from_file(input_path, format="m4a")
        audio = audio.set_frame_rate(sample_rate)

        if audio.channels > 1:
            audio = audio.set_channels(1)

        audio.export(output_path, format="wav")
        logger.info(f"Converted {input_path} to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        return False


def convert_midi_to_musicxml(midi_path, output_path):
    """Convert MIDI to MusicXML using your container's MuseScore - ORIGINAL"""
    try:
        if not os.environ.get('DISPLAY'):
            os.environ['DISPLAY'] = ':99'

        cmd = ['musescore3', '-o', output_path, midi_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0 and os.path.exists(output_path):
            return True, "Conversion successful"
        else:
            return False, f"MuseScore error: {result.stderr or 'Unknown error'}"

    except subprocess.TimeoutExpired:
        return False, "MuseScore conversion timed out"
    except Exception as e:
        return False, f"Conversion error: {str(e)}"


def convert_musicxml_to_pdf(musicxml_path, pdf_path):
    """Convert MusicXML to PDF using your container's MuseScore - ORIGINAL"""
    try:
        if not os.environ.get('DISPLAY'):
            os.environ['DISPLAY'] = ':99'

        cmd = ['musescore3', '-o', pdf_path, musicxml_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0 and os.path.exists(pdf_path):
            return True, "PDF conversion successful"
        else:
            return False, f"PDF conversion error: {result.stderr or 'Unknown error'}"

    except Exception as e:
        return False, f"PDF conversion error: {str(e)}"


def process_spectrogram_for_model(mel_spec):
    """Process the spectrogram to fit model input requirements - ORIGINAL"""
    expected_height = 229
    expected_width = 626

    if mel_spec.shape[0] != expected_height:
        mel_spec = tf.image.resize(
            tf.expand_dims(mel_spec, 0),
            [expected_height, mel_spec.shape[1]]
        )[0]

    if mel_spec.shape[1] < expected_width:
        padding = expected_width - mel_spec.shape[1]
        mel_spec = np.pad(mel_spec, ((0, 0), (0, padding)), mode='constant')
    elif mel_spec.shape[1] > expected_width:
        mel_spec = mel_spec[:, :expected_width]

    mel_spec = tf.transpose(mel_spec)
    mel_spec = tf.expand_dims(mel_spec, axis=0)
    mel_spec = tf.expand_dims(mel_spec, axis=-1)

    return mel_spec


# =============================================================================
# ALL ORIGINAL S3 HELPER FUNCTIONS - PRESERVED
# =============================================================================

def get_file_extension(filename):
    """Extract file extension from filename - ORIGINAL"""
    return '.' + filename.rsplit('.', 1)[1].lower() if '.' in filename else ''


def save_file_locally(file, filename):
    """Save uploaded file to local directory temporarily - ORIGINAL"""
    local_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(local_path)
    return local_path


def upload_file_to_s3(local_path, s3_path, content_type):
    """Upload file from local path to S3 - ORIGINAL"""
    if s3_client:
        s3_client.upload_file(
            local_path,
            AWS_BUCKET_NAME,
            s3_path,
            ExtraArgs={'ContentType': content_type}
        )


def clean_up_local_file(local_path):
    """Remove temporary local file - ORIGINAL"""
    try:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
    except Exception as e:
        logger.debug(f"Cleanup warning: {e}")


def save_metadata_to_s3(metadata, s3_path):
    """Save metadata JSON to S3 - ORIGINAL"""
    if not s3_client:
        return

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        json.dump(metadata, temp_file, indent=2)
        temp_file_path = temp_file.name

    s3_client.upload_file(
        temp_file_path,
        AWS_BUCKET_NAME,
        s3_path,
        ExtraArgs={'ContentType': 'application/json'}
    )

    os.remove(temp_file_path)


def save_generated_files_to_s3(user_id, recording_id, result_data):
    """Save generated MIDI and PDF files to S3 and update metadata - ORIGINAL"""
    if not s3_client:
        return result_data

    try:
        logger.info("💾 Saving generated files to S3...")

        recording_folder = f"users/{user_id}/recordings/{recording_id}"
        metadata_key = f"{recording_folder}/metadata.json"

        try:
            metadata_response = s3_client.get_object(
                Bucket=AWS_BUCKET_NAME,
                Key=metadata_key
            )
            metadata = json.loads(metadata_response['Body'].read().decode('utf-8'))
        except Exception as e:
            logger.error(f"❌ Could not load metadata: {e}")
            return result_data

        files_saved = 0

        # Save MIDI file if it exists
        if 'midi_file' in result_data and result_data['midi_file']:
            midi_url = result_data['midi_file']
            if midi_url.startswith('/api/download/'):
                midi_filename = midi_url.replace('/api/download/', '')
                midi_local_path = os.path.join(OUTPUT_FOLDER, midi_filename)

                if os.path.exists(midi_local_path):
                    midi_s3_path = f"{recording_folder}/transcription.mid"
                    upload_file_to_s3(midi_local_path, midi_s3_path, 'audio/midi')

                    midi_s3_url = f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com/{midi_s3_path}"

                    metadata['files']['midi'] = {
                        'filename': 'transcription.mid',
                        'original_name': 'AI_Generated_Transcription.mid',
                        'content_type': 'audio/midi',
                        's3_path': midi_s3_path,
                        'url': midi_s3_url,
                        'generated_date': datetime.now().isoformat(),
                        'generated_by': 'ai_transcription'
                    }

                    result_data['midi_file'] = midi_s3_url

                    logger.info(f"✅ MIDI saved to S3: {midi_s3_path}")
                    files_saved += 1

                    try:
                        os.remove(midi_local_path)
                    except:
                        pass

        # Save PDF file if it exists
        if 'sheet_music' in result_data and result_data['sheet_music']:
            sheet_info = result_data['sheet_music']
            if 'fileUrl' in sheet_info and sheet_info['fileUrl']:
                pdf_url = sheet_info['fileUrl']
                if pdf_url.startswith('/api/download/'):
                    pdf_filename = pdf_url.replace('/api/download/', '')

                    # Treat filename from URL as untrusted and enforce OUTPUT_FOLDER boundary
                    output_root = os.path.realpath(OUTPUT_FOLDER)
                    pdf_local_path = os.path.realpath(os.path.join(output_root, pdf_filename))

                    if os.path.commonpath([output_root, pdf_local_path]) != output_root:
                        logger.warning(f"⚠️ Skipping invalid PDF path outside output folder: {pdf_filename}")
                        continue

                    if os.path.exists(pdf_local_path):
                        pdf_s3_path = f"{recording_folder}/sheet_music.pdf"
                        upload_file_to_s3(pdf_local_path, pdf_s3_path, 'application/pdf')

                        pdf_s3_url = f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com/{pdf_s3_path}"

                        metadata['files']['pdf'] = {
                            'filename': 'sheet_music.pdf',
                            'original_name': 'AI_Generated_Sheet_Music.pdf',
                            'content_type': 'application/pdf',
                            's3_path': pdf_s3_path,
                            'url': pdf_s3_url,
                            'generated_date': datetime.now().isoformat(),
                            'generated_by': 'ai_transcription'
                        }

                        result_data['sheet_music']['fileUrl'] = pdf_s3_url

                        logger.info(f"✅ PDF saved to S3: {pdf_s3_path}")
                        files_saved += 1

                        try:
                            os.remove(pdf_local_path)
                        except:
                            pass

        if files_saved > 0:
            metadata['last_transcription'] = {
                'date': datetime.now().isoformat(),
                'files_saved': files_saved
            }

            save_metadata_to_s3(metadata, metadata_key)
            logger.info(f"✅ Metadata updated with {files_saved} new files")

        return result_data

    except Exception as e:
        logger.error(f"❌ Error saving files to S3: {e}")
        return result_data


# =============================================================================
# CORE TRANSCRIPTION FUNCTION - ALL ORIGINAL LOGIC
# =============================================================================

def calculate_chunk_duration_and_overlap():
    """
    Calculate the optimal chunk duration and overlap for processing.
    
    Returns:
        chunk_duration (float): Duration of each chunk in seconds
        overlap_duration (float): Overlap between chunks to avoid missing notes
    """
    # Based on your model's expected input size
    expected_width = 626  # time frames
    hop_length = 512
    sr = 16000
    
    # Calculate actual duration the model can handle
    chunk_duration = (expected_width * hop_length) / sr  # ≈ 20.032 seconds
    
    # Use 2-second overlap to catch notes that might be split between chunks
    overlap_duration = 2.0
    
    logger.info(f"📏 Chunk settings: {chunk_duration:.1f}s duration, {overlap_duration:.1f}s overlap")
    
    return chunk_duration, overlap_duration


def split_audio_into_chunks(audio_path: str) -> List[Tuple[str, float, float]]:
    """
    Split a long audio file into processable chunks.
    
    Args:
        audio_path (str): Path to the audio file
        
    Returns:
        List of tuples: (chunk_file_path, start_time, end_time)
    """
    try:
        # Load the full audio to get its duration
        y, sr = librosa.load(audio_path, sr=16000)
        total_duration = len(y) / sr
        
        logger.info(f"🎵 Audio duration: {total_duration:.1f} seconds")
        
        chunk_duration, overlap_duration = calculate_chunk_duration_and_overlap()
        
        # If audio is short enough, no chunking needed
        if total_duration <= chunk_duration:
            logger.info("✅ Audio fits in single chunk, no splitting needed")
            return [(audio_path, 0.0, total_duration)]
        
        # Calculate chunk boundaries
        chunks = []
        start_time = 0.0
        chunk_index = 0
        
        while start_time < total_duration:
            # Calculate end time for this chunk
            end_time = min(start_time + chunk_duration, total_duration)
            
            # Extract chunk audio data
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            chunk_audio = y[start_sample:end_sample]
            
            # Save chunk to temporary file
            chunk_filename = f"chunk_{chunk_index}_{uuid.uuid4()}.wav"
            chunk_path = os.path.join(UPLOAD_FOLDER, chunk_filename)
            
            # Save the chunk as WAV file
            import soundfile as sf
            sf.write(chunk_path, chunk_audio, sr)
            
            chunks.append((chunk_path, start_time, end_time))
            
            logger.info(f"📦 Chunk {chunk_index}: {start_time:.1f}s - {end_time:.1f}s -> {chunk_path}")
            
            # Move to next chunk with overlap
            # For the last chunk, don't add overlap
            if end_time < total_duration:
                start_time = end_time - overlap_duration
            else:
                break
                
            chunk_index += 1
        
        logger.info(f"✂️ Split audio into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"❌ Error splitting audio: {e}")
        # Fallback: return original file as single chunk
        return [(audio_path, 0.0, 0.0)]


def process_single_chunk(chunk_path: str, start_offset: float) -> List[Dict]:
    """
    Process a single audio chunk and return notes with adjusted timing.
    
    Args:
        chunk_path (str): Path to the chunk audio file
        start_offset (float): Time offset of this chunk in the original audio
        
    Returns:
        List of note dictionaries with corrected timestamps
    """
    try:
        logger.info(f"🔄 Processing chunk: {os.path.basename(chunk_path)} (offset: {start_offset:.1f}s)")
        
        # Extract mel spectrogram for this chunk
        mel_spec = extract_mel_spectrogram(chunk_path)
        mel_spec = process_spectrogram_for_model(mel_spec)
        
        # Get current model and make prediction
        current_model = stable_model_loader.get_model()
        predictions = current_model.predict(mel_spec)
        
        # Extract notes from predictions
        notes = extract_notes_from_predictions(predictions)
        
        # Adjust note timing by adding the chunk's start offset
        adjusted_notes = []
        for note in notes:
            adjusted_note = note.copy()
            adjusted_note['time'] = note['time'] + start_offset
            adjusted_notes.append(adjusted_note)
        
        logger.info(f"✅ Chunk processed: {len(adjusted_notes)} notes found")
        return adjusted_notes
        
    except Exception as e:
        logger.error(f"❌ Error processing chunk {chunk_path}: {e}")
        return []


def remove_duplicate_notes(all_notes: List[Dict], overlap_duration: float = 2.0) -> List[Dict]:
    """
    Remove duplicate notes that appear in overlapping regions between chunks.
    
    Args:
        all_notes: Combined list of all notes from all chunks
        overlap_duration: Duration of overlap between chunks
        
    Returns:
        List of unique notes with duplicates removed
    """
    if not all_notes:
        return []
    
    logger.info(f"🔍 Removing duplicates from {len(all_notes)} total notes...")
    
    # Sort notes by time first
    all_notes.sort(key=lambda x: x['time'])
    
    unique_notes = []
    
    for note in all_notes:
        is_duplicate = False
        
        # Check if this note is too similar to any recent note
        for existing_note in unique_notes[-10:]:  # Only check last 10 notes for efficiency
            time_diff = abs(note['time'] - existing_note['time'])
            pitch_diff = abs(note['pitch'] - existing_note['pitch'])
            
            # Consider it a duplicate if:
            # - Same pitch and very close in time (within overlap region)
            # - Time difference is less than 0.5 seconds
            if pitch_diff == 0 and time_diff < 0.5:
                is_duplicate = True
                logger.debug(f"   Duplicate found: {note['note_name']} at {note['time']:.2f}s")
                break
        
        if not is_duplicate:
            unique_notes.append(note)
    
    removed_count = len(all_notes) - len(unique_notes)
    logger.info(f"✅ Removed {removed_count} duplicate notes, {len(unique_notes)} unique notes remain")
    
    return unique_notes


def process_long_audio_in_chunks(audio_path: str) -> List[Dict]:
    """
    Main function to process long audio files by splitting into chunks.
    
    Args:
        audio_path (str): Path to the audio file
        
    Returns:
        List of all detected notes with correct timing
    """
    try:
        logger.info(f"🎼 Starting chunked processing for: {audio_path}")
        
        # Split audio into manageable chunks
        chunks = split_audio_into_chunks(audio_path)
        
        if len(chunks) == 1:
            # No chunking needed, process normally
            logger.info("📋 Single chunk processing")
            mel_spec = extract_mel_spectrogram(audio_path)
            mel_spec = process_spectrogram_for_model(mel_spec)
            current_model = stable_model_loader.get_model()
            predictions = current_model.predict(mel_spec)
            return extract_notes_from_predictions(predictions)
        
        # Process each chunk
        all_notes = []
        chunk_files_to_cleanup = []
        
        for i, (chunk_path, start_time, end_time) in enumerate(chunks):
            logger.info(f"🔄 Processing chunk {i+1}/{len(chunks)}: {start_time:.1f}s - {end_time:.1f}s")
            
            # Process this chunk
            chunk_notes = process_single_chunk(chunk_path, start_time)
            all_notes.extend(chunk_notes)
            
            # Mark chunk file for cleanup (but not if it's the original file)
            if chunk_path != audio_path:
                chunk_files_to_cleanup.append(chunk_path)
        
        # Remove duplicate notes from overlapping regions
        _, overlap_duration = calculate_chunk_duration_and_overlap()
        unique_notes = remove_duplicate_notes(all_notes, overlap_duration)
        
        # Clean up temporary chunk files
        for chunk_file in chunk_files_to_cleanup:
            try:
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
                    logger.debug(f"🧹 Cleaned up chunk file: {os.path.basename(chunk_file)}")
            except Exception as cleanup_e:
                logger.warning(f"⚠️ Could not clean up {chunk_file}: {cleanup_e}")
        
        logger.info(f"🎉 Chunked processing complete: {len(unique_notes)} total notes from {len(chunks)} chunks")
        return unique_notes
        
    except Exception as e:
        logger.error(f"❌ Error in chunked processing: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to regular processing
        logger.info("🔄 Falling back to regular processing...")
        mel_spec = extract_mel_spectrogram(audio_path)
        mel_spec = process_spectrogram_for_model(mel_spec)
        current_model = stable_model_loader.get_model()
        predictions = current_model.predict(mel_spec)
        return extract_notes_from_predictions(predictions)


# Modified perform_transcription function to use chunking
def perform_transcription_with_chunking(audio_file_path, title="Piano Transcription", sheet_format="pdf", tempo=120):
    """
    Enhanced transcription function that handles long audio files by chunking.
    This replaces your original perform_transcription function.
    """
    try:
        logger.info(f"🎵 Starting enhanced transcription for: {audio_file_path}")

        # Convert to WAV if needed
        wav_path = os.path.splitext(audio_file_path)[0] + ".wav"
        if not audio_file_path.lower().endswith('.wav'):
            convert_audio_to_wav(audio_file_path, wav_path)
        else:
            wav_path = audio_file_path

        # Check audio duration to decide processing method
        y, sr = librosa.load(wav_path, sr=16000)
        total_duration = len(y) / sr
        
        logger.info(f"⏱️ Audio duration: {total_duration:.1f} seconds")
        
        if total_duration > 22:
            logger.info(f"📏 Long audio detected ({total_duration:.1f}s), using chunked processing")
            notes = process_long_audio_in_chunks(wav_path)
        else:
            logger.info(f"📏 Short audio ({total_duration:.1f}s), using standard processing")
            # Use original processing method for short audio
            mel_spec = extract_mel_spectrogram(wav_path)
            mel_spec = process_spectrogram_for_model(mel_spec)
            current_model = stable_model_loader.get_model()
            predictions = current_model.predict(mel_spec)
            notes = extract_notes_from_predictions(predictions)

        logger.info(f"🎼 Total notes extracted: {len(notes)}")

        # Create MIDI file from all notes
        midi_filename = f"{uuid.uuid4()}.mid"
        midi_path = os.path.join(OUTPUT_FOLDER, midi_filename)
        create_midi_from_notes(notes, midi_path)

        # Generate sheet music if possible
        musescore_available, musescore_info = check_musescore_installation()
        sheet_music_result = None

        if musescore_available and os.path.exists(midi_path):
            try:
                logger.info("🎼 Generating sheet music from full-length MIDI...")

                sheet_uuid = str(uuid.uuid4())
                musicxml_filename = f"{sheet_uuid}.musicxml"
                musicxml_path = os.path.join(OUTPUT_FOLDER, musicxml_filename)

                pdf_filename = f"{sheet_uuid}.pdf"
                pdf_path = os.path.join(OUTPUT_FOLDER, pdf_filename)

                success, message = convert_midi_to_musicxml(midi_path, musicxml_path)

                if success and sheet_format.lower() == 'pdf':
                    pdf_success, pdf_message = convert_musicxml_to_pdf(musicxml_path, pdf_path)

                    if pdf_success:
                        sheet_music_result = {
                            "fileUrl": f"/api/download/{pdf_filename}",
                            "format": "pdf",
                            "title": title
                        }
                        logger.info(f"✅ Full-length sheet music generated: {pdf_filename}")
                    else:
                        logger.error(f"❌ PDF generation failed: {pdf_message}")
                elif success:
                    sheet_music_result = {
                        "fileUrl": f"/api/download/{musicxml_filename}",
                        "format": "musicxml",
                        "title": title
                    }
                    logger.info(f"✅ Full-length MusicXML generated: {musicxml_filename}")
                else:
                    logger.error(f"❌ Sheet music generation failed: {message}")

            except Exception as sheet_e:
                logger.error(f"❌ Sheet music generation error: {sheet_e}")

        # Clean up temporary WAV file
        try:
            if wav_path != audio_file_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception as cleanup_e:
            logger.warning(f"⚠️ Warning: Could not clean up temporary wav file: {cleanup_e}")

        result_data = {
            "success": True,
            "notes": notes,
            "midi_file": f"/api/download/{midi_filename}",
            "musescore_available": musescore_available,
            "sheet_music": sheet_music_result,
            "debug_info": {
                "total_duration": total_duration,
                "processing_method": "chunked" if total_duration > 22 else "standard",
                "notes_extracted": len(notes),
                "sheet_music_generated": sheet_music_result is not None
            }
        }

        logger.info(f"🎉 Enhanced transcription complete: {len(notes)} notes, Duration: {total_duration:.1f}s")
        return True, result_data, None

    except Exception as e:
        logger.error(f"❌ Error in enhanced transcription: {e}")
        import traceback
        traceback.print_exc()
        return False, None, str(e)

def perform_transcription(audio_file_path, title="Piano Transcription", sheet_format="pdf", tempo=120):
    """Core transcription logic - ALL ORIGINAL FUNCTIONALITY"""
    try:
        logger.info(f"🎵 Starting transcription for: {audio_file_path}")

        wav_path = os.path.splitext(audio_file_path)[0] + ".wav"
        if not audio_file_path.lower().endswith('.wav'):
            convert_audio_to_wav(audio_file_path, wav_path)
        else:
            wav_path = audio_file_path

        logger.info("🔊 Extracting mel spectrogram...")
        mel_spec = extract_mel_spectrogram(wav_path)
        mel_spec = process_spectrogram_for_model(mel_spec)
        logger.info(f"📊 Processed spectrogram shape: {mel_spec.shape}")

        try:
            logger.info("🤖 Loading AI model...")
            current_model = stable_model_loader.get_model()
            logger.info("✅ Model loaded successfully for transcription")
        except Exception as model_e:
            logger.error(f"❌ Error loading model: {model_e}")
            return False, None, f"Model loading failed: {str(model_e)}"

        logger.info("🧠 Running AI model prediction...")
        predictions = current_model.predict(mel_spec)
        logger.info("✅ Model prediction completed")

        logger.info(f"🔍 Model output debug:")
        logger.info(f"  - Predictions type: {type(predictions)}")
        logger.info(f"  - Number of outputs: {len(predictions)}")
        for i, pred in enumerate(predictions):
            logger.info(f"  - Output {i} shape: {pred.shape}")

        try:
            notes = extract_notes_from_predictions(predictions)
            logger.info(f"🎼 Extracted {len(notes)} notes successfully")
        except Exception as extraction_error:
            logger.error(f"❌ Note extraction failed: {extraction_error}")
            return False, None, f"Note extraction failed: {str(extraction_error)}"

        midi_filename = f"{uuid.uuid4()}.mid"
        midi_path = os.path.join(OUTPUT_FOLDER, midi_filename)
        create_midi_from_notes(notes, midi_path)

        musescore_available, musescore_info = check_musescore_installation()

        sheet_music_result = None

        if musescore_available and os.path.exists(midi_path):
            try:
                logger.info("🎼 Generating sheet music...")

                sheet_uuid = str(uuid.uuid4())
                musicxml_filename = f"{sheet_uuid}.musicxml"
                musicxml_path = os.path.join(OUTPUT_FOLDER, musicxml_filename)

                pdf_filename = f"{sheet_uuid}.pdf"
                pdf_path = os.path.join(OUTPUT_FOLDER, pdf_filename)

                success, message = convert_midi_to_musicxml(midi_path, musicxml_path)

                if success and sheet_format.lower() == 'pdf':
                    pdf_success, pdf_message = convert_musicxml_to_pdf(musicxml_path, pdf_path)

                    if pdf_success:
                        sheet_music_result = {
                            "fileUrl": f"/api/download/{pdf_filename}",
                            "format": "pdf",
                            "title": title
                        }
                        logger.info(f"✅ Sheet music generated: {pdf_filename}")
                    else:
                        logger.error(f"❌ PDF generation failed: {pdf_message}")
                elif success:
                    sheet_music_result = {
                        "fileUrl": f"/api/download/{musicxml_filename}",
                        "format": "musicxml",
                        "title": title
                    }
                    logger.info(f"✅ MusicXML generated: {musicxml_filename}")
                else:
                    logger.error(f"❌ Sheet music generation failed: {message}")

            except Exception as sheet_e:
                logger.error(f"❌ Sheet music generation error: {sheet_e}")

        try:
            if wav_path != audio_file_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception as cleanup_e:
            logger.warning(f"⚠️ Warning: Could not clean up temporary wav file: {cleanup_e}")

        result_data = {
            "success": True,
            "notes": notes,
            "midi_file": f"/api/download/{midi_filename}",
            "musescore_available": musescore_available,
            "sheet_music": sheet_music_result,
            "debug_info": {
                "model_outputs": len(predictions),
                "notes_extracted": len(notes),
                "sheet_music_generated": sheet_music_result is not None
            }
        }

        logger.info(
            f"🎉 Transcription complete: {len(notes)} notes, MIDI: ✅, Sheet: {'✅' if sheet_music_result else '❌'}")

        return True, result_data, None

    except Exception as e:
        logger.error(f"❌ Error in transcription: {e}")
        import traceback
        traceback.print_exc()
        return False, None, str(e)


# =============================================================================
# FLASK APPLICATION SETUP
# =============================================================================

app = Flask(__name__)
CORS(app)


@app.before_request
def before_request():
    """Handle platform differences and periodic cleanup"""
    platform_info = detect_request_platform(request)
    g.platform_info = platform_info

    logger.debug(f"🔍 Request from: {platform_info['user_agent'][:50]}...")
    logger.debug(f"📱 Platform: {'Web Browser' if platform_info['is_web'] else 'Mobile App'}")

    # Periodic cleanup (10% chance)
    if np.random.random() < 0.1:
        try:
            # Clean old files
            current_time = time.time()
            for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
                for file_path in Path(folder).glob('*'):
                    if file_path.is_file():
                        age_minutes = (current_time - file_path.stat().st_mtime) / 60
                        if age_minutes > 30:  # Remove files older than 30 minutes
                            try:
                                file_path.unlink()
                            except:
                                pass
        except Exception as e:
            logger.debug(f"Cleanup warning: {e}")

    if request.method == 'OPTIONS':
        return '', 200


@app.after_request
def after_request(response):
    """Ensure proper CORS headers for all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


@app.errorhandler(Exception)
def handle_exception(e):
    """IMPROVED error handling - no more cascading failures"""
    logger.error(f"❌ Error occurred: {e}", exc_info=True)

    # Don't automatically reset model - this was causing cascades
    # Only log and return clean JSON response

    return jsonify({
        "success": False,
        "error": "Server error occurred",
        "details": str(e)
    }), 500


# =============================================================================
# ALL ORIGINAL ENDPOINTS - COMPLETE PRESERVATION
# =============================================================================

@app.route('/hello', methods=['GET'])
def hello():
    """Hello endpoint - ORIGINAL"""
    return jsonify({"message": "Hello, World!"}), 200


@app.route('/api/musescore-status', methods=['GET'])
def check_musescore_status():
    """Check if MuseScore is available for sheet music generation - ORIGINAL"""
    try:
        is_available, version_info = check_musescore_installation()
        return jsonify({
            "available": is_available,
            "version": version_info,
            "features": ["pdf", "musicxml"] if is_available else []
        })
    except Exception as e:
        return jsonify({
            "available": False,
            "error": str(e),
            "features": []
        })


@app.route('/upload', methods=['POST'])
def upload_recording_with_files():
    """Enhanced upload endpoint that handles multiple file types - ALL ORIGINAL"""
    try:
        logger.info("📤 Enhanced upload request received")

        if 'userId' not in request.form:
            return jsonify({"error": "User ID is required"}), 400

        user_id = request.form['userId']
        title = request.form.get('title', 'Untitled Recording')
        description = request.form.get('description', '')

        if 'audio_file' not in request.files:
            return jsonify({"error": "Audio file is required"}), 400

        audio_file = request.files['audio_file']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400

        image_file = request.files.get('image_file')
        pdf_file = request.files.get('pdf_file')
        midi_file = request.files.get('midi_file')

        logger.info(f"📋 Upload details:")
        logger.info(f"   User: {user_id}")
        logger.info(f"   Title: {title}")
        logger.info(f"   Audio: {audio_file.filename}")
        logger.info(f"   Image: {image_file.filename if image_file else 'None'}")
        logger.info(f"   PDF: {pdf_file.filename if pdf_file else 'None'}")
        logger.info(f"   MIDI: {midi_file.filename if midi_file else 'None'}")

        recording_id = str(uuid.uuid4())
        timestamp = datetime.now()

        recording_folder = f"users/{user_id}/recordings/{recording_id}"

        metadata = {
            'recording_id': recording_id,
            'user_id': user_id,
            'title': title,
            'description': description,
            'upload_date': timestamp.isoformat(),
            'created_date': timestamp.strftime('%Y-%m-%d'),
            'files': {}
        }

        uploaded_files = {}

        # Process audio file (required)
        audio_extension = get_file_extension(audio_file.filename)
        audio_s3_path = f"{recording_folder}/audio{audio_extension}"
        local_audio_path = save_file_locally(audio_file, f"audio_{recording_id}{audio_extension}")

        upload_file_to_s3(local_audio_path, audio_s3_path, audio_file.content_type)
        metadata['files']['audio'] = {
            'filename': f"audio{audio_extension}",
            'original_name': audio_file.filename,
            'content_type': audio_file.content_type,
            's3_path': audio_s3_path,
            'url': f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com/{audio_s3_path}"
        }
        uploaded_files['audio'] = metadata['files']['audio']['url']
        clean_up_local_file(local_audio_path)

        # Process image file (optional)
        if image_file and image_file.filename:
            image_extension = get_file_extension(image_file.filename)
            image_s3_path = f"{recording_folder}/image{image_extension}"
            local_image_path = save_file_locally(image_file, f"image_{recording_id}{image_extension}")

            upload_file_to_s3(local_image_path, image_s3_path, image_file.content_type)
            metadata['files']['image'] = {
                'filename': f"image{image_extension}",
                'original_name': image_file.filename,
                'content_type': image_file.content_type,
                's3_path': image_s3_path,
                'url': f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com/{image_s3_path}"
            }
            uploaded_files['image'] = metadata['files']['image']['url']
            clean_up_local_file(local_image_path)

        # Process PDF file (optional)
        if pdf_file and pdf_file.filename:
            pdf_s3_path = f"{recording_folder}/sheet_music.pdf"
            local_pdf_path = save_file_locally(pdf_file, f"pdf_{recording_id}.pdf")

            upload_file_to_s3(local_pdf_path, pdf_s3_path, 'application/pdf')
            metadata['files']['pdf'] = {
                'filename': 'sheet_music.pdf',
                'original_name': pdf_file.filename,
                'content_type': 'application/pdf',
                's3_path': pdf_s3_path,
                'url': f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com/{pdf_s3_path}"
            }
            uploaded_files['pdf'] = metadata['files']['pdf']['url']
            clean_up_local_file(local_pdf_path)

        # Process MIDI file (optional)
        if midi_file and midi_file.filename:
            midi_s3_path = f"{recording_folder}/transcription.mid"
            local_midi_path = save_file_locally(midi_file, f"midi_{recording_id}.mid")

            upload_file_to_s3(local_midi_path, midi_s3_path, 'audio/midi')
            metadata['files']['midi'] = {
                'filename': 'transcription.mid',
                'original_name': midi_file.filename,
                'content_type': 'audio/midi',
                's3_path': midi_s3_path,
                'url': f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com/{midi_s3_path}"
            }
            uploaded_files['midi'] = metadata['files']['midi']['url']
            clean_up_local_file(local_midi_path)

        # Save metadata.json to S3
        metadata_s3_path = f"{recording_folder}/metadata.json"
        save_metadata_to_s3(metadata, metadata_s3_path)

        logger.info(f"✅ Upload successful for recording {recording_id}")

        return jsonify({
            "success": True,
            "message": f"Recording '{title}' uploaded successfully",
            "recording_id": recording_id,
            "files": uploaded_files,
            "metadata": metadata
        }), 200

    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/recordings/<user_id>', methods=['GET'])
def get_user_recordings_enhanced(user_id):
    """Enhanced endpoint to get all recordings with their files - ALL ORIGINAL"""
    try:
        logger.info(f"📋 Getting recordings for user: {user_id}")

        # Load model on first recordings request
        if not stable_model_loader.is_model_ready():
            logger.info("Loading model on recordings endpoint request...")
            try:
                stable_model_loader.get_model()
                logger.info("Model loaded successfully!")
            except Exception as e:
                logger.warning(f"Model loading failed: {e}")

        if not s3_client:
            return jsonify({
                "success": True,
                "userId": user_id,
                "recordings": [],
                "model_ready": stable_model_loader.is_model_ready(),
                "total_recordings": 0,
                "message": "S3 not configured"
            }), 200

        user_recordings_prefix = f"users/{user_id}/recordings/"

        response = s3_client.list_objects_v2(
            Bucket=AWS_BUCKET_NAME,
            Prefix=user_recordings_prefix,
            Delimiter='/'
        )

        recordings = []

        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                recording_folder = prefix['Prefix']
                recording_id = recording_folder.split('/')[-2]

                try:
                    metadata_key = f"{recording_folder}metadata.json"
                    metadata_response = s3_client.get_object(
                        Bucket=AWS_BUCKET_NAME,
                        Key=metadata_key
                    )
                    metadata = json.loads(metadata_response['Body'].read().decode('utf-8'))

                    recordings.append({
                        'recording_id': recording_id,
                        'metadata': metadata,
                        'files': metadata.get('files', {}),
                        'title': metadata.get('title', 'Untitled'),
                        'upload_date': metadata.get('upload_date'),
                        'description': metadata.get('description', ''),
                        'user_id': metadata.get('user_id', user_id),
                        # For backward compatibility with existing UI
                        'url': metadata.get('files', {}).get('audio', {}).get('url', ''),
                        'has_image': 'image' in metadata.get('files', {}),
                        'has_pdf': 'pdf' in metadata.get('files', {}),
                        'has_midi': 'midi' in metadata.get('files', {})
                    })

                except Exception as e:
                    logger.error(f"❌ Error reading metadata for recording {recording_id}: {e}")
                    continue

        recordings.sort(key=lambda x: x.get('upload_date', ''), reverse=True)

        logger.info(f"🎵 Found {len(recordings)} recordings for user {user_id}")

        return jsonify({
            "success": True,
            "userId": user_id,
            "recordings": recordings,
            "model_ready": stable_model_loader.is_model_ready(),
            "total_recordings": len(recordings)
        }), 200

    except Exception as e:
        logger.error(f"❌ Error in enhanced recordings endpoint: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/recordings/<recording_id>', methods=['PUT'])
def update_recording_metadata(recording_id):
    """Update recording metadata and optionally replace image - ALL ORIGINAL"""
    try:
        logger.info(f"📝 Updating recording {recording_id}")

        if 'userId' not in request.form:
            return jsonify({"error": "User ID is required"}), 400

        user_id = request.form['userId']
        title = request.form.get('title', 'Untitled Recording')
        description = request.form.get('description', '')

        logger.info(f"👤 User: {user_id}")
        logger.info(f"🏷️ New title: {title}")
        logger.info(f"📝 New description: {description}")

        recording_folder = f"users/{user_id}/recordings/{recording_id}"
        metadata_key = f"{recording_folder}/metadata.json"

        if not s3_client:
            return jsonify({"error": "S3 not configured"}), 503

        try:
            metadata_response = s3_client.get_object(
                Bucket=AWS_BUCKET_NAME,
                Key=metadata_key
            )
            metadata = json.loads(metadata_response['Body'].read().decode('utf-8'))
            logger.info("📋 Loaded existing metadata")
        except Exception as e:
            logger.error(f"❌ Could not load existing metadata: {e}")
            return jsonify({"error": "Recording not found or access denied"}), 404

        metadata['title'] = title
        metadata['description'] = description
        metadata['last_modified'] = datetime.now().isoformat()

        image_file = request.files.get('image_file')
        if image_file and image_file.filename:
            logger.info(f"🖼️ Processing new image: {image_file.filename}")

            image_extension = get_file_extension(image_file.filename)
            local_image_path = save_file_locally(image_file, f"image_update_{recording_id}{image_extension}")

            image_s3_path = f"{recording_folder}/image{image_extension}"
            upload_file_to_s3(local_image_path, image_s3_path, image_file.content_type)

            metadata['files']['image'] = {
                'filename': f"image{image_extension}",
                'original_name': image_file.filename,
                'content_type': image_file.content_type,
                's3_path': image_s3_path,
                'url': f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com/{image_s3_path}"
            }

            clean_up_local_file(local_image_path)
            logger.info("✅ Image updated successfully")

        save_metadata_to_s3(metadata, metadata_key)
        logger.info("✅ Metadata updated successfully")

        return jsonify({
            "success": True,
            "message": f"Recording '{title}' updated successfully",
            "recording_id": recording_id,
            "metadata": metadata
        }), 200

    except Exception as e:
        logger.error(f"❌ Update error: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/recordings/<user_id>/<recording_id>/transcribe', methods=['POST'])
def generate_transcription_for_recording(user_id, recording_id):
    """Generate AI transcription for an existing recording - ALL ORIGINAL"""
    try:
        logger.info(f"🤖 Generating transcription for recording {recording_id}")

        data = request.get_json() or {}
        title = data.get('title', 'Piano Transcription')
        sheet_format = data.get('sheet_format', 'pdf')
        tempo = int(data.get('tempo', 120))

        if not s3_client:
            return jsonify({"error": "S3 not configured"}), 503

        recording_folder = f"users/{user_id}/recordings/{recording_id}"
        metadata_key = f"{recording_folder}/metadata.json"

        try:
            metadata_response = s3_client.get_object(
                Bucket=AWS_BUCKET_NAME,
                Key=metadata_key
            )
            metadata = json.loads(metadata_response['Body'].read().decode('utf-8'))
            logger.info("📋 Loaded recording metadata")
        except Exception as e:
            logger.error(f"❌ Could not load recording metadata: {e}")
            return jsonify({"error": "Recording not found"}), 404

        audio_info = metadata.get('files', {}).get('audio')
        if not audio_info:
            return jsonify({"error": "Audio file not found in recording"}), 404

        audio_s3_path = audio_info['s3_path']
        logger.info(f"📥 Audio file S3 path: {audio_s3_path}")

        temp_audio_path = None

        try:
            audio_extension = audio_info.get('filename', 'audio.m4a').split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_extension}') as temp_audio:
                temp_audio_path = temp_audio.name

            s3_client.download_file(
                AWS_BUCKET_NAME,
                audio_s3_path,
                temp_audio_path
            )

            logger.info(f"📥 Downloaded audio file for transcription: {temp_audio_path}")

            success, result_data, error_message = perform_transcription_with_chunking(
                temp_audio_path, title, sheet_format, tempo
            )

            if success:
                result_data['recording_id'] = recording_id
                result_data['user_id'] = user_id

                updated_result_data = save_generated_files_to_s3(user_id, recording_id, result_data)

                logger.info(f"✅ Transcription completed for recording {recording_id}")
                if 'midi_file' in updated_result_data:
                    logger.info(f"   MIDI: {updated_result_data['midi_file']}")
                if 'sheet_music' in updated_result_data and updated_result_data['sheet_music']:
                    logger.info(f"   PDF: {updated_result_data['sheet_music'].get('fileUrl', 'None')}")

                return jsonify(updated_result_data), 200
            else:
                logger.error(f"❌ Transcription failed: {error_message}")
                return jsonify({"error": error_message}), 500

        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                    logger.info("🧹 Cleaned up temporary audio file")
                except Exception as cleanup_e:
                    logger.warning(f"⚠️ Warning: Could not clean up temp file: {cleanup_e}")

    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint - ORIGINAL"""
    return jsonify({
        "status": "ok",
        "model_loaded": stable_model_loader.is_model_ready(),
        "timestamp": datetime.now().isoformat(),
        "message": "Model loads on first recordings endpoint request"
    })


@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio_with_sheet_music():
    """Enhanced transcribe endpoint that includes sheet music generation - ALL ORIGINAL"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        platform_info = g.get('platform_info', detect_request_platform(request))
        platform_type = 'web' if platform_info['is_web'] else 'mobile'

        logger.info(f"🎵 Processing audio from: {platform_type}")

        # Check if model needs reset due to platform switch
        if hasattr(g, 'last_platform') and g.last_platform != platform_type:
            logger.info(f"🔄 Platform switch detected ({g.last_platform} -> {platform_type})")
            # Don't automatically reset - just log the switch

        g.last_platform = platform_type

        sheet_format = request.form.get('sheet_format', 'pdf')
        title = request.form.get('title', 'Piano Transcription')
        tempo = int(request.form.get('tempo', '120'))

        audio_path = safe_file_processing(file, platform_type)

        logger.info(f"🎵 Processing uploaded audio file from {platform_type}: {os.path.basename(audio_path)}")

        success, result_data, error_message = perform_transcription_with_chunking(
            audio_path, title, sheet_format, tempo
        )

        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as cleanup_e:
            logger.warning(f"⚠️ Warning: Could not clean up uploaded file: {cleanup_e}")

        if success:
            result_data['platform'] = platform_type
            return jsonify(result_data)
        else:
            return jsonify({"error": error_message}), 500

    except Exception as e:
        logger.error(f"❌ Error in transcription endpoint: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({"error": str(e)}), 500


@app.route('/api/convert-midi-to-sheet', methods=['POST'])
def convert_midi_to_sheet():
    """Convert an existing MIDI file to sheet music - ALL ORIGINAL"""
    if 'midi' not in request.files:
        return jsonify({"error": "No MIDI file provided"}), 400

    file = request.files['midi']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        format_type = request.form.get('format', 'pdf')
        title = request.form.get('title', 'Piano Sheet Music')

        musescore_available, musescore_info = check_musescore_installation()

        if not musescore_available:
            return jsonify({
                "error": "MuseScore is not available for sheet music generation",
                "musescore_info": musescore_info
            }), 400

        original_filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4()}_{original_filename}"
        midi_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(midi_path)

        if format_type.lower() == 'pdf':
            musicxml_filename = f"{os.path.splitext(filename)[0]}.musicxml"
            musicxml_path = os.path.join(OUTPUT_FOLDER, musicxml_filename)

            pdf_filename = f"{os.path.splitext(filename)[0]}.pdf"
            pdf_path = os.path.join(OUTPUT_FOLDER, pdf_filename)

            success, message = convert_midi_to_musicxml(midi_path, musicxml_path)

            if success:
                pdf_success, pdf_message = convert_musicxml_to_pdf(musicxml_path, pdf_path)

                if pdf_success:
                    sheet_music_result = {
                        "fileUrl": f"/api/download/{pdf_filename}",
                        "format": "pdf",
                        "title": title
                    }
                else:
                    return jsonify({"error": f"PDF conversion failed: {pdf_message}"}), 500
            else:
                return jsonify({"error": f"MusicXML conversion failed: {message}"}), 500
        else:
            musicxml_filename = f"{os.path.splitext(filename)[0]}.musicxml"
            musicxml_path = os.path.join(OUTPUT_FOLDER, musicxml_filename)

            success, message = convert_midi_to_musicxml(midi_path, musicxml_path)

            if success:
                sheet_music_result = {
                    "fileUrl": f"/api/download/{musicxml_filename}",
                    "format": "musicxml",
                    "title": title
                }
            else:
                return jsonify({"error": f"MusicXML conversion failed: {message}"}), 500

        try:
            if os.path.exists(midi_path):
                os.remove(midi_path)
        except Exception as cleanup_e:
            logger.warning(f"Warning: Could not clean up MIDI file: {cleanup_e}")

        return jsonify({
            "success": True,
            "sheet_music": sheet_music_result,
            "musescore_available": True
        })

    except Exception as e:
        logger.error(f"Error in MIDI to sheet conversion: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_midi(filename):
    """Download the generated MIDI file - ORIGINAL"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, secure_filename(filename))
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {e}")
        return jsonify({"error": str(e)}), 404


@app.route('/process-audio', methods=['POST'])
def process_audio():
    """Process uploaded audio file and return detected notes - ALL ORIGINAL"""
    if 'file' not in request.files:
        logger.error("No file part in the request.")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file.")
        return jsonify({"error": "No selected file"}), 400

    try:
        file_ext = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        audio_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(audio_path)
        logger.info(f"File saved at: {audio_path}")

        wav_file_path = os.path.splitext(audio_path)[0] + ".wav"
        if file_ext != '.wav':
            logger.info(f"Converting {file_ext} to WAV...")
            if file_ext == '.m4a':
                success = convert_m4a_to_wav(audio_path, wav_file_path)
            else:
                success = convert_audio_to_wav(audio_path, wav_file_path)

            if not success:
                return jsonify({"error": "Failed to convert audio file"}), 500
        else:
            wav_file_path = audio_path

        mel_spec = extract_mel_spectrogram(wav_file_path)
        expected_height = 229
        expected_width = 625

        logger.info(f"Original spectrogram shape: {mel_spec.shape}")

        if mel_spec.shape[0] != expected_height:
            logger.info(f"Resizing frequency dimension from {mel_spec.shape[0]} to {expected_height}")
            mel_spec = tf.image.resize(
                tf.expand_dims(mel_spec, 0),
                [expected_height, mel_spec.shape[1]]
            )[0]

        if mel_spec.shape[1] < expected_width:
            padding = expected_width - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, padding)), mode='constant')
            logger.info(f"Padded time dimension to {mel_spec.shape}")
        elif mel_spec.shape[1] > expected_width:
            mel_spec = mel_spec[:, :expected_width]
            logger.info(f"Trimmed time dimension to {mel_spec.shape}")

        mel_spec = tf.transpose(mel_spec)
        mel_spec = tf.expand_dims(mel_spec, axis=0)
        mel_spec = tf.expand_dims(mel_spec, axis=-1)
        logger.info(f"Spectrogram shape for model input: {mel_spec.shape}")

        try:
            current_model = stable_model_loader.get_model()
            logger.info("Model loaded successfully for processing")
            logger.info("Running model prediction...")
            predictions = current_model.predict(mel_spec)
            logger.info("Model prediction completed")
            notes = extract_notes_from_predictions(predictions)
            logger.info(f"Extracted {len(notes)} notes")

        except Exception as model_error:
            logger.error(f"Model failed: {model_error}, generating test notes...")
            logger.info("Creating test notes (C major scale)...")
            notes = []
            for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
                notes.append({
                    "note_name": pitch_to_note_name(pitch),
                    "time": float(i * 0.5),
                    "duration": 0.4,
                    "velocity": 0.8,
                    "velocity_midi": 100,
                    "pitch": pitch,
                    "frequency": librosa.midi_to_hz(pitch)
                })

        logger.info("\n===== EXTRACTED NOTES SUMMARY =====")
        logger.info(f"Total notes extracted: {len(notes)}")
        if len(notes) > 0:
            logger.info(f"Time range: {notes[0]['time']:.2f}s to {notes[-1]['time']:.2f}s")
            logger.info(f"Pitch range: {min([n['pitch'] for n in notes])} to {max([n['pitch'] for n in notes])}")

        midi_filename = f"{os.path.splitext(unique_filename)[0]}.mid"
        midi_output_path = os.path.join(OUTPUT_FOLDER, midi_filename)
        create_midi_from_notes(notes, midi_output_path)
        logger.info(f"MIDI file created at: {midi_output_path}")

        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(wav_file_path) and wav_file_path != audio_path:
                os.remove(wav_file_path)
        except Exception as cleanup_e:
            logger.warning(f"Warning: Could not clean up temporary files: {cleanup_e}")

        return jsonify({
            "success": True,
            "notes": notes,
            "midi_file": f"/api/download/{midi_filename}"
        }), 200

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/reset-server-state', methods=['POST'])
def reset_server_state():
    """Emergency endpoint to reset server state when corruption occurs - IMPROVED"""
    try:
        logger.info("🔄 Manually resetting server state...")

        # Safe reset with cooldown
        reset_success = stable_model_loader.safe_reset()

        # Clear temporary files
        temp_files_cleared = 0
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            try:
                for filename in os.listdir(folder):
                    if filename.startswith(('temp_', 'web_', 'mobile_', 'test_')):
                        try:
                            file_path = os.path.join(folder, filename)
                            os.remove(file_path)
                            temp_files_cleared += 1
                        except Exception as file_error:
                            logger.warning(f"⚠️ Could not remove {filename}: {file_error}")
            except Exception as folder_error:
                logger.warning(f"⚠️ Could not access folder {folder}: {folder_error}")

        try:
            gc.collect()
        except:
            pass

        logger.info(f"✅ Server state reset complete - cleared {temp_files_cleared} temp files")

        return jsonify({
            "success": True,
            "message": "Server state reset successfully",
            "temp_files_cleared": temp_files_cleared,
            "model_reset": reset_success,
            "model_ready": stable_model_loader.is_model_ready()
        })

    except Exception as e:
        logger.error(f"❌ Reset failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # For Hugging Face Spaces, use the PORT environment variable
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)