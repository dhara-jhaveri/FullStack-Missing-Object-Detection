import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({ 
    object_present: true, 
    fps: 0 
  });
  const [roi, setRoi] = useState(null); // ROI: {x1, y1, x2, y2}
  const [drawing, setDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);

  const startStream = () => {
    if (wsRef.current) return;
    setStreaming(true);
    setError(null);
    wsRef.current = new WebSocket('ws://localhost:8000/ws/stream');
    wsRef.current.onopen = () => {
      // Send ROI if already selected
      if (roi) {
        wsRef.current.send(JSON.stringify({ type: 'roi', roi }));
      }
    };
    wsRef.current.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === "frame") {
          const byteArray = new Uint8Array(
            message.data.match(/.{1,2}/g).map(byte => parseInt(byte, 16))
          );
          const blob = new Blob([byteArray], { type: 'image/jpeg' });
          const url = URL.createObjectURL(blob);
          if (videoRef.current) {
            if (videoRef.current.src) {
              URL.revokeObjectURL(videoRef.current.src);
            }
            videoRef.current.src = url;
          }
          setStats(message.stats);
        }
      } catch (err) {
        console.error('Error processing message:', err);
      }
    };
    wsRef.current.onerror = (error) => {
      setError('Failed to connect to video stream');
      stopStream();
    };
    wsRef.current.onclose = () => {
      stopStream();
    };
  };

  const stopStream = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStreaming(false);
    if (videoRef.current) {
      URL.revokeObjectURL(videoRef.current.src);
      videoRef.current.src = '';
    }
  };

  useEffect(() => {
    return () => {
      stopStream();
    };
  }, []);

  // Canvas overlay for ROI selection
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !roi) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.strokeRect(
      roi.x1,
      roi.y1,
      roi.x2 - roi.x1,
      roi.y2 - roi.y1
    );
  }, [roi, streaming]);

  const handleCanvasMouseDown = (e) => {
    if (!streaming) return;
    const rect = e.target.getBoundingClientRect();
    setStartPoint({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
    setDrawing(true);
  };

  const handleCanvasMouseMove = (e) => {
    if (!drawing) return;
    const rect = e.target.getBoundingClientRect();
    const x2 = e.clientX - rect.left;
    const y2 = e.clientY - rect.top;
    setRoi({
      x1: startPoint.x,
      y1: startPoint.y,
      x2,
      y2
    });
  };

  const handleCanvasMouseUp = () => {
    setDrawing(false);
    if (roi && wsRef.current && wsRef.current.readyState === 1) {
      wsRef.current.send(JSON.stringify({ type: 'roi', roi }));
    }
  };

  return (
    <div className="App">
      <h1>Real-Time Object Monitoring</h1>
      <div className="controls">
        <button 
          onClick={streaming ? stopStream : startStream}
          className={streaming ? 'stop-btn' : 'start-btn'}
        >
          {streaming ? 'Stop Stream' : 'Start Stream'}
        </button>
      </div>
      {error && <div className="error-message">{error}</div>}
      <div className="video-container" style={{ position: 'relative', width: 960, height: 600 }}>
        <img 
          ref={videoRef} 
          alt="Live Stream with Detection" 
          style={{ display: streaming ? 'block' : 'none', width: 960, height: 600 }} 
        />
        <canvas
          ref={canvasRef}
          width={960}
          height={600}
          style={{
            position: 'absolute',
            left: 0,
            top: 0,
            pointerEvents: streaming ? 'auto' : 'none',
            zIndex: 2
          }}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleCanvasMouseMove}
          onMouseUp={handleCanvasMouseUp}
        />
        {!streaming && !error && (
          <div className="placeholder">
            Stream offline. Click "Start Stream" to begin.
          </div>
        )}
      </div>
      {streaming && (
        <div className="stats-panel">
          <h3>Detection Statistics</h3>
          <div className="stat-item">
            <span className="stat-label">Object Present:</span>
            <span className="stat-value">{stats.object_present ? 'Yes' : 'No'}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">FPS:</span>
            <span className="stat-value">{stats.fps.toFixed(1)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;