# Voice Agent Images

Documentation images for the Voice Agent README.

## Required Images

Place the following image files in this directory:

| Filename | Description | Used In |
|----------|-------------|---------|
| `stt_only_audio_file_upload.png` | STT-Only mode audio file upload screenshot | README.md, examples/voice_agent/README.md |
| `logo_white.png` | Logo (white, for dark theme) | Web client |
| `logo_black.png` | Logo (black, for light theme) | Web client |

## Screenshot Guidelines

For `stt_only_audio_file_upload.png`:
- Capture the STT-Only mode with a file uploaded
- Show the waveform visualization
- Include the conversation area with transcription results
- Recommended width: 800-1200px
- Use PNG format for best quality

## Logo Files

Copy logo files from GPU server:
```bash
# If logos are on GPU server
cp /path/to/logo_*.png ./

# Copy to web client public directory
cp logo_*.png ../client/public/images/
```

## File Structure

```
examples/voice_agent/images/
├── README.md                           # This file
├── stt_only_audio_file_upload.png      # Screenshot for README
├── logo_white.png               # Logo for dark theme
└── logo_black.png               # Logo for light theme
```
