import React, { useRef, useEffect, useState } from "react";

const Camera = () => {
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);

  useEffect(() => {
    const getVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setStream(stream);
      } catch (err) {
        console.error("Error accessing the camera: ", err);
      }
    };

    getVideo();

    return () => {
      stopVideoStream();
    };
  }, []);

  const stopVideoStream = () => {
    if (stream) {
      // Stop all video and audio tracks
      console.log("Stopped")
      stream.getTracks().forEach((track) => {
        track.stop();
      });
      videoRef.current.srcObject = null;
    }
  };

  return (
    <div>
      <video className="border border-gray-300 w-auto" ref={videoRef}  />
    </div>
  );
};

export default Camera;
