Camera modules expose a shared interface for stream URLs, snapshot capture, and brand-specific controls.

Usage example:

```python
from src.camera import get_camera

front1 = get_camera("front_1")
print(front1.get_stream_url())

front1.pan_left(20)
front1.stop()

left1 = get_camera("left_1")
left1.zoom_in(100)
```

Capabilities by brand:

- `HIKVISION`: stream, snapshot, pan, tilt, zoom, PTZ continuous move
- `AXIS`: stream, snapshot, zoom only
