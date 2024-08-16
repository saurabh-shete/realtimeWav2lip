import React, { useState, useEffect, useRef } from "react";
import io from "socket.io-client";
import Camera from "./Camera";

const ThirdPage = () => {
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [audioBuffer, setAudioBuffer] = useState([]);
  const [socket, setSocket] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const videoCanvasRef = useRef(null);

  useEffect(() => {
    const newSocket = io("http://127.0.0.1:5000/video", {
      transports: ["websocket", "polling"],
    });

    setSocket(newSocket);

    newSocket.on("connect", () => {
      console.log("Connected to the /video namespace");
    });

    newSocket.on("video_frame", (data) => {
      const canvas = videoCanvasRef.current;
      if (canvas) {
        const context = canvas.getContext("2d");
        const image = new Image();
        image.onload = () => {
          context.clearRect(0, 0, canvas.width, canvas.height);
          context.drawImage(image, 0, 0, canvas.width, canvas.height);
        };
        image.src = `data:image/jpeg;base64,${data.frame}`;
      }
    });

    return () => newSocket.close();
  }, []);

  const startRecording = () => {
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        const newMediaRecorder = new MediaRecorder(stream, {
          mimeType: "audio/webm",
        });
        setMediaRecorder(newMediaRecorder);
        setIsRecording(true);
        newMediaRecorder.start(1000);

        newMediaRecorder.addEventListener("dataavailable", (event) => {
          const audioChunk = event.data;
          setAudioBuffer((prevBuffer) => {
            const updatedBuffer = [...prevBuffer, audioChunk];

            if (updatedBuffer.length >= 2) {
              const audioBlob = new Blob(updatedBuffer, { type: "audio/webm" });
              convertAndSendAudioData(audioBlob);
              return [];
            }

            return updatedBuffer;
          });
        });

        newMediaRecorder.addEventListener("stop", () => {
          stream.getTracks().forEach((track) => track.stop());
        });
      })
      .catch((error) => {
        console.error("Error accessing audio devices:", error);
      });
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  const convertAndSendAudioData = (audioBlob) => {
    if (socket) {
      const reader = new FileReader();
      reader.readAsArrayBuffer(audioBlob);

      reader.onloadend = () => {
        const audioArrayBuffer = reader.result;
        socket.emit("audio_data", audioArrayBuffer);
        setIsLoading(true);
        console.log("Sent audio data");
      };
    }
  };

  return (
    <div className="flex flex-col items-center bg-gray-100 p-8 h-screen">
      <h1 className="text-4xl font-bold mb-6 text-center text-gray-800">
        Real Time Lip Sync
      </h1>
      <div className="border-2 border-green-500 p-24 rounded-lg shadow-lg w-full max-w-6xl" style={{ boxShadow: '0 0 10px #00ff00' }}>
        <div className="flex justify-between items-center">
          <div className="flex-1 p-2 border-2 border-green-500 rounded-lg h-[400px] w-64 mr-4" style={{ boxShadow: '0 0 10px #00ff00' }}>
            <h2 className="text-xl font-semibold mb-4 text-center">Video Input</h2>
            <Camera />
          </div>
          <div className="border-2 border-green-500 rounded-lg flex items-center justify-center mx-4" style={{ boxShadow: '0 0 10px #00ff00' }}>
            {!isRecording ? (
              <button
                onClick={startRecording}
                className="bg-green-500 text-white font-semibold py-2 px-4 rounded-lg hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-400"
              >
                Virtual Mode
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="bg-red-500 text-white font-semibold py-2 px-4 rounded-lg hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-400"
              >
                Stop Virtual Mode
              </button>
            )}
          </div>
          <div className="flex-1 p-2 border-2 border-green-500 rounded-lg h-[400px] w-64 ml-4" style={{ boxShadow: '0 0 10px #00ff00' }}>
            <h2 className="text-xl font-semibold mb-4 text-center">Lip Sync Output</h2>
            <canvas
              ref={videoCanvasRef}
              className="border h-[300px] w-full border-gray-300"
            ></canvas>
          </div>
        </div>
        {isLoading && (
          <div className="mt-4 text-center">
            <p>Processing...</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ThirdPage;
