import noisereduce as nr
import scipy.io.wavfile as wav
import numpy as np
import os

def denoise_audio(input_file='extracted_audio.wav', output_file='cleaned_output.wav'):
    """
    Apply noise reduction to the given input WAV file and save the cleaned version.

    Args:
        input_file (str): Path to the noisy input audio file (.wav)
        output_file (str): Path to save the cleaned output audio file (.wav)
    """
    if not os.path.exists(input_file):
        print(f"❌ Input file '{input_file}' not found.")
        return

    try:
        print(f"🔊 Reading input audio from {input_file}")
        rate, data = wav.read(input_file)

        if data.ndim == 2:
            print("🎧 Stereo audio detected. Converting to mono.")
            data = np.mean(data, axis=1).astype(data.dtype)

        print("🧹 Reducing noise...")
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
        wav.write(output_file, rate, reduced_noise.astype(np.int16))
        print(f"✅ Denoised audio saved to {output_file}")

    except Exception as e:
        print(f"❌ Error during noise cancellation: {e}")
