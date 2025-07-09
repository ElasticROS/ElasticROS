#!/usr/bin/env python3
"""
Example: Human-robot dialogue with ElasticROS
Demonstrates elastic computing for speech recognition tasks
"""

import numpy as np
import time
import wave
import struct
from typing import Dict, List, Tuple, Optional
import threading
import queue

# ElasticROS imports
from elasticros_core import ElasticNode, PressNode, ReleaseNode

# Optional imports for real audio
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("PyAudio not available - using simulated audio")


class SpeechRecognitionPressNode(PressNode):
    """Local computation for speech recognition"""
    
    def _initialize(self):
        """Initialize local audio processing"""
        self.sample_rate = 16000
        self.frame_size = 512
        self.mel_bins = 128
        
        # Simple VAD (Voice Activity Detection) parameters
        self.energy_threshold = 0.01
        self.silence_frames = 20
        
    def _process(self, data: np.ndarray, compute_ratio: float) -> Dict:
        """
        Process audio for speech recognition.
        
        Args:
            data: Audio samples
            compute_ratio: How much to process locally
            
        Returns:
            Processed data or features
        """
        if compute_ratio == 0.0:
            # No local processing
            return {
                'raw_audio': data,
                'sample_rate': self.sample_rate
            }
            
        # Step 1: Basic preprocessing (always done if any local processing)
        # Remove DC offset
        audio = data - np.mean(data)
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
            
        if compute_ratio <= 0.3:
            # Minimal processing
            return {
                'preprocessed_audio': audio,
                'sample_rate': self.sample_rate,
                'processing_level': 'basic'
            }
            
        # Step 2: Voice Activity Detection
        energy = np.sum(audio ** 2) / len(audio)
        is_speech = energy > self.energy_threshold
        
        if not is_speech and compute_ratio < 1.0:
            # No speech detected - might skip cloud processing
            return {
                'preprocessed_audio': audio,
                'is_speech': False,
                'energy': energy,
                'processing_level': 'vad'
            }
            
        if compute_ratio <= 0.7:
            # VAD + basic features
            return {
                'preprocessed_audio': audio,
                'is_speech': is_speech,
                'energy': energy,
                'processing_level': 'vad'
            }
            
        # Step 3: Feature extraction (full local processing)
        # Compute spectrogram
        spectrogram = self._compute_spectrogram(audio)
        
        # Compute MFCC-like features (simplified)
        features = self._extract_features(spectrogram)
        
        return {
            'features': features,
            'spectrogram': spectrogram,
            'is_speech': is_speech,
            'processing_level': 'full',
            'duration': len(audio) / self.sample_rate
        }
        
    def _compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute simple spectrogram"""
        # Simplified - in practice use proper STFT
        n_frames = len(audio) // self.frame_size
        spec = np.zeros((self.mel_bins, n_frames))
        
        for i in range(n_frames):
            frame = audio[i * self.frame_size:(i + 1) * self.frame_size]
            # Simple frequency analysis
            fft = np.fft.rfft(frame * np.hanning(len(frame)))
            spec[:, i] = np.abs(fft[:self.mel_bins])
            
        return spec
        
    def _extract_features(self, spectrogram: np.ndarray) -> np.ndarray:
        """Extract audio features from spectrogram"""
        # Simplified MFCC-like features
        # Log mel spectrogram
        log_spec = np.log(spectrogram + 1e-10)
        
        # DCT to get cepstral coefficients (simplified)
        n_mfcc = 13
        features = np.zeros((n_mfcc, log_spec.shape[1]))
        
        for i in range(log_spec.shape[1]):
            features[:, i] = np.real(np.fft.rfft(log_spec[:, i]))[:n_mfcc]
            
        return features


class SpeechRecognitionReleaseNode(ReleaseNode):
    """Cloud computation for speech recognition using deep learning"""
    
    def _initialize(self):
        """Initialize cloud speech recognition models"""
        print("Loading speech recognition model in cloud...")
        time.sleep(1.5)  # Simulate model loading
        
        # In practice, would load actual ASR model like:
        # self.model = load_whisper_model() or load_wav2vec2()
        self.model_loaded = True
        
        # Vocabulary for mock recognition
        self.vocabulary = [
            "hello", "robot", "move", "forward", "backward", "left", "right",
            "stop", "start", "grasp", "release", "help", "status", "battery",
            "shutdown", "navigate", "to", "the", "kitchen", "bedroom", "door"
        ]
        
    def _process(self, data: Dict, compute_ratio: float) -> Dict:
        """
        Complete speech recognition in cloud.
        
        Args:
            data: Input from Press node
            compute_ratio: Cloud computation ratio
            
        Returns:
            Speech recognition results
        """
        start_time = time.time()
        
        # Check if speech was detected
        if data.get('is_speech') is False:
            return {
                'text': "",
                'confidence': 0.0,
                'is_speech': False,
                'processing_time': time.time() - start_time
            }
            
        # Extract audio based on processing level
        if 'raw_audio' in data:
            # Need full processing
            audio = data['raw_audio']
            features = self._extract_cloud_features(audio)
            
        elif 'features' in data:
            # Use pre-extracted features
            features = data['features']
            
        elif 'preprocessed_audio' in data:
            # Partial processing done
            audio = data['preprocessed_audio']
            features = self._extract_cloud_features(audio)
            
        else:
            raise ValueError("No audio data received")
            
        # Run inference (simulated)
        if self.use_gpu:
            time.sleep(0.1)  # Fast GPU inference
        else:
            time.sleep(0.3)  # Slower CPU inference
            
        # Generate mock transcription
        transcription = self._generate_mock_transcription(features, data.get('duration', 1.0))
        
        processing_time = time.time() - start_time
        
        return {
            'text': transcription['text'],
            'confidence': transcription['confidence'],
            'words': transcription['words'],
            'processing_time': processing_time,
            'processed_on': self.instance_type,
            'preprocessing_level': data.get('processing_level', 'cloud'),
            'timestamp': time.time()
        }
        
    def _extract_cloud_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features using cloud resources"""
        # Simulate cloud feature extraction
        # In practice, might use more sophisticated models
        time.sleep(0.05)
        
        # Mock features
        n_features = 128
        n_frames = len(audio) // 512
        return np.random.randn(n_features, n_frames)
        
    def _generate_mock_transcription(self, features: np.ndarray, duration: float) -> Dict:
        """Generate mock speech recognition results"""
        # Estimate number of words based on duration
        # Average speech rate: 150 words per minute
        expected_words = int(duration * 150 / 60)
        expected_words = max(1, min(expected_words, 10))
        
        # Generate random but coherent phrase
        words = []
        word_list = []
        
        # Common robot commands
        command_templates = [
            ["move", "forward"],
            ["turn", "left"],
            ["turn", "right"],
            ["stop", "moving"],
            ["grasp", "the", "object"],
            ["go", "to", "the", "kitchen"],
            ["check", "battery", "status"],
            ["hello", "robot"],
        ]
        
        # Pick a template if appropriate length
        for template in command_templates:
            if len(template) <= expected_words:
                word_list = template
                break
                
        if not word_list:
            # Generate random words
            word_list = np.random.choice(self.vocabulary, expected_words, replace=True).tolist()
            
        # Add timing and confidence
        time_per_word = duration / len(word_list)
        current_time = 0.0
        
        for word in word_list:
            words.append({
                'word': word,
                'start_time': current_time,
                'end_time': current_time + time_per_word,
                'confidence': np.random.uniform(0.85, 0.99)
            })
            current_time += time_per_word
            
        return {
            'text': ' '.join(word_list),
            'confidence': np.mean([w['confidence'] for w in words]),
            'words': words
        }


class AudioCapture:
    """Helper class for audio capture"""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        if AUDIO_AVAILABLE:
            self.pa = pyaudio.PyAudio()
            self.stream = None
            
    def start(self):
        """Start audio capture"""
        if not AUDIO_AVAILABLE:
            return
            
        self.is_recording = True
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()
        
    def stop(self):
        """Stop audio capture"""
        self.is_recording = False
        
        if AUDIO_AVAILABLE and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
        
    def get_audio_chunk(self, duration: float) -> Optional[np.ndarray]:
        """Get audio chunk of specified duration"""
        if not AUDIO_AVAILABLE:
            # Return simulated audio
            samples = int(self.sample_rate * duration)
            # Generate speech-like pattern
            t = np.linspace(0, duration, samples)
            audio = np.sin(2 * np.pi * 200 * t) * np.exp(-t)  # Decaying tone
            audio += 0.1 * np.random.randn(samples)  # Add noise
            return audio
            
        # Collect real audio
        chunks = []
        samples_needed = int(self.sample_rate * duration)
        samples_collected = 0
        
        timeout = time.time() + duration + 1.0  # Add buffer
        
        while samples_collected < samples_needed and time.time() < timeout:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                chunks.append(chunk)
                samples_collected += len(chunk) // 2  # 16-bit samples
            except queue.Empty:
                continue
                
        if not chunks:
            return None
            
        # Convert to numpy array
        audio_bytes = b''.join(chunks)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Trim to exact duration
        if len(audio) > samples_needed:
            audio = audio[:samples_needed]
            
        return audio.astype(np.float32) / 32768.0  # Normalize


def main():
    """Main function demonstrating elastic speech recognition"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ElasticROS Speech Recognition Example')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Audio chunk duration in seconds')
    parser.add_argument('--continuous', action='store_true',
                       help='Continuous recognition mode')
    parser.add_argument('--simulate-cpu', type=float, default=None,
                       help='Simulate CPU usage (%)')
    args = parser.parse_args()
    
    # Initialize ElasticNode
    elastic_node = ElasticNode(args.config)
    
    # Create and register nodes
    press_node = SpeechRecognitionPressNode("speech_press")
    release_node = SpeechRecognitionReleaseNode(
        "speech_release",
        config={'instance_type': 't2.small', 'use_gpu': False}
    )
    
    elastic_node.register_node_pair("speech_recognition", press_node, release_node)
    
    # Initialize audio capture
    audio_capture = AudioCapture()
    
    # Simulate CPU constraint if requested
    cpu_load_thread = None
    if args.simulate_cpu:
        def cpu_load():
            """Generate CPU load"""
            while True:
                # Busy loop to consume CPU
                start = time.time()
                while time.time() - start < args.simulate_cpu / 100:
                    _ = sum(i*i for i in range(1000))
                time.sleep(1 - args.simulate_cpu / 100)
                
        cpu_load_thread = threading.Thread(target=cpu_load, daemon=True)
        cpu_load_thread.start()
        print(f"Simulating {args.simulate_cpu}% CPU usage")
    
    print(f"Starting speech recognition (chunk duration: {args.duration}s)")
    print("Speak into microphone (or using simulated audio)...")
    
    if args.continuous:
        print("Press Ctrl+C to stop")
        
        audio_capture.start()
        
        try:
            while True:
                # Get audio chunk
                audio = audio_capture.get_audio_chunk(args.duration)
                
                if audio is None:
                    time.sleep(0.1)
                    continue
                    
                # Process with ElasticROS
                start_time = time.time()
                result = elastic_node.elastic_execute("speech_recognition", audio)
                total_time = time.time() - start_time
                
                # Display results
                if result.get('text'):
                    print(f"\n[{time.strftime('%H:%M:%S')}] Recognized: \"{result['text']}\"")
                    print(f"  Confidence: {result.get('confidence', 0):.2%}")
                    print(f"  Total time: {total_time*1000:.1f}ms")
                    print(f"  Processing: {result.get('preprocessing_level', 'unknown')}")
                    
                # Show word timings if available
                if 'words' in result and result['words']:
                    print("  Words:")
                    for word_info in result['words']:
                        print(f"    {word_info['word']}: "
                              f"{word_info['start_time']:.2f}-{word_info['end_time']:.2f}s "
                              f"({word_info['confidence']:.2%})")
                              
        except KeyboardInterrupt:
            print("\nStopping...")
            
        finally:
            audio_capture.stop()
            
    else:
        # Single recognition
        print("\nProcessing multiple audio chunks to demonstrate adaptation...")
        
        for i in range(10):
            # Get or generate audio
            audio = audio_capture.get_audio_chunk(args.duration)
            
            if audio is None:
                print(f"Iteration {i+1}: No audio captured")
                continue
                
            # Process
            start_time = time.time()
            result = elastic_node.elastic_execute("speech_recognition", audio)
            total_time = time.time() - start_time
            
            print(f"\nIteration {i+1}:")
            print(f"  Recognized: \"{result.get('text', '')}\"")
            print(f"  Confidence: {result.get('confidence', 0):.2%}")
            print(f"  Total time: {total_time*1000:.1f}ms")
            print(f"  Cloud processing: {result.get('processing_time', 0)*1000:.1f}ms")
            print(f"  Preprocessing: {result.get('preprocessing_level', 'unknown')}")
            
            # Simulate CPU spike
            if i == 5 and args.simulate_cpu:
                new_cpu = min(90, args.simulate_cpu * 1.5)
                print(f"\n!!! CPU usage increased to {new_cpu}% !!!\n")
                # In practice, would actually increase load
                
    # Final statistics
    stats = elastic_node.get_statistics()
    print("\n=== Final Statistics ===")
    print(f"Average execution time: {stats['average_execution_time']*1000:.1f}ms")
    print(f"Action distribution: {stats['action_distribution']}")
    print(f"Total executions: {stats['total_executions']}")
    
    # Cleanup
    elastic_node.shutdown()
    

if __name__ == '__main__':
    main()