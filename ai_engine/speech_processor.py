"""
MODULE FOUR: Experimental Speech Analysis Engine.
Uses Librosa for acoustic feature extraction (duration, non-silent pause segmentation,
RMS signal-to-noise quality) and optional Faster-Whisper transcription with sequence similarity
against a reference clinical sentence.
"""
from difflib import SequenceMatcher
import io
import tempfile
from pathlib import Path
from typing import Any, Dict

import librosa
import numpy as np
import soundfile as sf

from config import SPEECH_REFERENCE_SENTENCE

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except Exception:
    HAS_WHISPER = False


class SpeechAnalysisProcessor:
    """Analyses recorded or uploaded patient speech audio with explicit acoustic quality checks."""

    def __init__(self) -> None:
        self._whisper_model = None

    def _get_whisper_model(self):
        if not HAS_WHISPER:
            return None
        if self._whisper_model is None:
            try:
                self._whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            except Exception:
                return None
        return self._whisper_model

    def analyse_audio_bytes(
        self,
        audio_bytes: bytes,
        reference_sentence: str = SPEECH_REFERENCE_SENTENCE,
        manual_transcription_fallback: str = "",
    ) -> Dict[str, Any]:
        """Extracts acoustic pause/rate metrics via Librosa and transcription similarity."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = Path(tmp.name)

            y, sr = librosa.load(str(tmp_path), sr=16000, mono=True)
        except Exception as exc:
            return {
                "audio_quality_flag": "INVALID_AUDIO_FILE",
                "duration_sec": 0.0,
                "pause_count": 0,
                "speaking_rate_wpm": 0.0,
                "similarity_score": 0.0,
                "transcription": "",
                "message": f"Could not decode audio file: {exc}",
            }
        finally:
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        duration_sec = float(librosa.get_duration(y=y, sr=sr))
        if duration_sec < 0.8:
            return {
                "audio_quality_flag": "TOO_SHORT_UNRELIABLE",
                "duration_sec": round(duration_sec, 2),
                "pause_count": 0,
                "speaking_rate_wpm": 0.0,
                "similarity_score": 0.0,
                "transcription": "",
                "message": "Recording is shorter than 0.8 seconds; insufficient audio for analysis.",
            }

        rms = float(np.sqrt(np.mean(y**2)))
        quality_flag = "VALID_AUDIO" if rms >= 0.008 else "LOW_VOLUME_NOISY"

        # Non-silent interval detection via librosa
        intervals = librosa.effects.split(y, top_db=25)
        pause_count = max(0, len(intervals) - 1)
        active_speech_samples = sum(int(end - start) for start, end in intervals)
        active_speech_sec = float(active_speech_samples) / float(sr)
        pause_ratio = round(
            max(0.0, (duration_sec - active_speech_sec) / max(duration_sec, 1e-6)), 3
        )

        # Transcription via Faster-Whisper (if installed & model cached) or fallback
        transcription = manual_transcription_fallback.strip()
        whisper_model = self._get_whisper_model()
        if whisper_model is not None and not transcription:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wtmp:
                    sf.write(wtmp.name, y, sr)
                    segments, _ = whisper_model.transcribe(wtmp.name, beam_size=1)
                    transcription = " ".join(seg.text.strip() for seg in segments).strip()
                Path(wtmp.name).unlink(missing_ok=True)
            except Exception:
                pass

        word_count = len(transcription.split()) if transcription else max(1, int(active_speech_sec * 2.2))
        speaking_rate_wpm = round((word_count / max(duration_sec, 1e-6)) * 60.0, 1)

        if transcription:
            clean_ref = reference_sentence.lower().strip()
            clean_hyp = transcription.lower().strip()
            similarity = round(SequenceMatcher(None, clean_ref, clean_hyp).ratio(), 3)
        else:
            similarity = 0.0

        return {
            "audio_quality_flag": quality_flag,
            "duration_sec": round(duration_sec, 2),
            "active_speech_sec": round(active_speech_sec, 2),
            "pause_count": int(pause_count),
            "pause_ratio": pause_ratio,
            "speaking_rate_wpm": speaking_rate_wpm,
            "similarity_score": similarity,
            "transcription": transcription or "[Acoustic features extracted; enter/verify transcript below]",
            "reference_sentence": reference_sentence,
            "message": (
                "LIMITATION NOTE: Accent, native language, ambient noise, microphone clipping, "
                "and non-stroke speech conditions affect transcription and acoustic metrics. "
                "Transcription mismatches must NEVER be automatically diagnosed as stroke dysarthria."
            ),
        }
