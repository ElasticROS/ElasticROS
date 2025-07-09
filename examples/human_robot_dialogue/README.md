# ElasticROS Human-Robot Dialogue Example

This example demonstrates how ElasticROS optimizes speech recognition and dialogue systems by dynamically adapting to CPU usage and system resources.

## Overview

The human-robot dialogue example implements:
- **Press Node**: Local audio preprocessing (VAD, feature extraction)
- **Release Node**: Cloud-based speech recognition using deep learning
- **Elastic Node**: Adaptive decision making based on CPU usage and latency requirements

## Features

- **Voice Activity Detection (VAD)**: Efficient local filtering of non-speech audio
- **Adaptive Feature Extraction**: Dynamic preprocessing based on available resources
- **Cloud ASR Integration**: Scalable speech recognition in the cloud
- **Real-time Processing**: Low-latency dialogue capabilities

## Requirements

- Python 3.8+
- PyAudio (optional, for real microphone input)
- ElasticROS core installed
- (Optional) Microphone for live audio

## Installation

```bash
# Install audio dependencies
sudo apt-get install -y portaudio19-dev
pip install pyaudio

# If PyAudio installation fails, the example will use simulated audio
```

## Quick Start

### 1. Basic Test

Run with simulated audio:
```bash
python speech_recognition_node.py
```

### 2. Continuous Recognition

Process audio continuously:
```bash
python speech_recognition_node.py --continuous
```

### 3. Custom Duration

Set audio chunk duration:
```bash
python speech_recognition_node.py --duration 3.0
```

### 4. Simulate CPU Load

Test adaptation to high CPU usage:
```bash
# Simulate 70% CPU usage
python speech_recognition_node.py --simulate-cpu 70 --continuous
```

## Configuration

Edit `config.yaml` to customize:

```yaml
speech_recognition:
  sample_rate: 16000
  chunk_duration: 2.0
  
  vad:
    energy_threshold: 0.01
    silence_frames: 20
    
  features:
    frame_size: 512
    mel_bins: 128
    mfcc_coeffs: 13

thresholds:
  max_latency: 300  # ms - for interactive dialogue
  max_cpu_usage: 60  # Offload when CPU > 60%
```

## Expected Behavior

### 1. **Low CPU Usage (<30%)**:
   - Full cloud processing for best accuracy
   - Minimal local computation
   - Lower latency due to fast cloud inference

### 2. **Medium CPU Usage (30-60%)**:
   - Local VAD and basic preprocessing
   - Cloud handles main recognition
   - Balanced resource utilization

### 3. **High CPU Usage (>60%)**:
   - Maximum local preprocessing
   - Feature extraction done locally
   - Only essential data sent to cloud

## Audio Pipeline

```
Microphone/Audio → VAD → Feature Extraction → Cloud ASR → Text Output
                    ↑                    ↑
                    └── Press Node ──────┴──── Release Node
```

## Performance Metrics

The example tracks:
- **Recognition Latency**: End-to-end processing time
- **Cloud Processing Time**: ASR inference duration
- **Preprocessing Level**: Where computation was performed
- **VAD Efficiency**: Percentage of audio containing speech

## Usage Examples

### Simple Command Recognition

```python
# Configure for commands
config = {
    'optimization_metric': 'latency',
    'thresholds': {
        'max_latency': 200  # Fast response for commands
    }
}
```

### Conversational Dialogue

```python
# Configure for conversation
config = {
    'optimization_metric': 'accuracy',
    'speech_recognition': {
        'chunk_duration': 5.0,  # Longer utterances
        'vad': {
            'energy_threshold': 0.005  # More sensitive
        }
    }
}
```

### Noisy Environment

```python
# Configure for noise
config = {
    'speech_recognition': {
        'vad': {
            'energy_threshold': 0.02,  # Higher threshold
            'silence_frames': 30  # More silence required
        }
    }
}
```

## Troubleshooting

### 1. **No audio input detected**:
   - Check microphone permissions
   - Verify PyAudio installation: `python -c "import pyaudio"`
   - Try with simulated audio first

### 2. **High latency**:
   - Check network connection to cloud
   - Reduce chunk duration
   - Enable more local preprocessing

### 3. **Poor recognition accuracy**:
   - Ensure good audio quality (16kHz sampling)
   - Check microphone placement
   - Adjust VAD thresholds

### 4. **CPU simulation not working**:
   - The simulation creates actual CPU load
   - Monitor with `htop` or system monitor
   - Adjust simulation percentage as needed

## Advanced Usage

### Custom Audio Processing

```python
class MyAudioPressNode(PressNode):
    def _process(self, audio, compute_ratio):
        if compute_ratio > 0.5:
            # Heavy preprocessing
            features = extract_mfcc(audio)
            features = apply_noise_reduction(features)
            return {'features': features}
        else:
            # Light preprocessing
            return {'audio': normalize_audio(audio)}
```

### Integration with TTS

```python
# Add text-to-speech for full dialogue
class DialogueSystem:
    def __init__(self):
        self.elastic = ElasticNode()
        self.tts = TTSEngine()
        
    def process_turn(self, audio):
        # Speech recognition
        text = self.elastic.elastic_execute("speech", audio)
        
        # Generate response
        response = generate_response(text)
        
        # Text to speech
        audio_response = self.tts.synthesize(response)
        return audio_response
```

### Multi-language Support

```python
# Configure for multiple languages
config = {
    'speech_recognition': {
        'languages': ['en', 'es', 'fr'],
        'model': 'multilingual_whisper'
    }
}
```

## Benchmarking

Compare ElasticROS with fixed strategies:

```bash
# Run performance comparison
cd ../benchmarks
python speech_benchmark.py --iterations 100
```

This will show:
- Adaptation to CPU load changes
- Latency distribution
- Resource utilization patterns

## Integration with ROS

For ROS integration:

```bash
# Launch with ROS
roslaunch elasticros_ros speech_recognition.launch

# Subscribe to transcription topic
rostopic echo /elasticros/speech/transcription
```

## Future Enhancements

- [ ] Real-time speech translation
- [ ] Multi-speaker identification
- [ ] Emotion recognition
- [ ] Context-aware dialogue management
- [ ] Edge-optimized models

## Contributing

We welcome contributions! Ideas for improvement:
- Add support for more audio formats
- Implement custom VAD algorithms
- Create language-specific optimizations
- Add streaming recognition support