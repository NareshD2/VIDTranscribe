import React, { useState } from "react";
import axios from "axios";
import {
  Box,
  Card,
  Typography,
  Button,
  Select,
  MenuItem,
  CircularProgress,
  LinearProgress,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import { motion, AnimatePresence } from "framer-motion";

const LANGUAGES = [
  "English", "Spanish", "French", "German", "Hindi",
  "Tamil", "Telugu", "Chinese", "Japanese", "Korean"
];

function toCode(lang) {
  const map = {
    English: "en", Spanish: "es", French: "fr", German: "de",
    Hindi: "hi", Tamil: "ta", Telugu: "te", Chinese: "zh-CN",
    Japanese: "ja", Korean: "ko",
  };
  return map[lang] || lang.toLowerCase().slice(0, 2);
}

export default function VideoTranslatePanel({
  previewUrl,
  onFileSelected,
  sourceLang,
  targetLang,
  setSourceLang,
  setTargetLang,
}) {
  const [file, setFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [outputUrl, setOutputUrl] = useState(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState([]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    if (!selectedFile.type.startsWith("video/")) {
      alert("Please upload a valid video file.");
      return;
    }
    setFile(selectedFile);
    onFileSelected(selectedFile);
    setOutputUrl(null);
  };

  const handleTranslateClick = async () => {
    if (!previewUrl || !file) return alert("Please upload a video first.");

    setIsProcessing(true);
    setOutputUrl(null);
    setProgress(0);
    setStatusMsg("Starting translation...");

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("src_lang", toCode(sourceLang));
      formData.append("target_lang", toCode(targetLang));
//https://consultatively-brushed-christian.ngrok-free.dev
      const res = await axios.post("http://127.0.0.1:5000/translate", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const downloadPath = res.data.download_path;
      setStatusMsg(res.data.message || "Processing started...");

      const evtSource = new EventSource("http://127.0.0.1:5000/progress");

      evtSource.onopen = () => {
        setConnected(true);
      };

      evtSource.onmessage = async (e) => {
        try {
          const data = JSON.parse(e.data);

          if (data.status === "processing") {
            setEvents((prev) => [data, ...prev].slice(0, 50));
            if (data.total > 0) {
              setProgress(Math.min(100, Math.round((data.processed / data.total) * 100)));
            }
            if (data.message) setStatusMsg(data.message);
          }

          if (data.status === "done") {
            evtSource.close();
            setProgress(100);
            setStatusMsg("Translation complete 🎉");
            setIsProcessing(false); // ✅ stop processing before showing video

            const downloadRes = await axios.get(
              `http://127.0.0.1:5000/download?path=${encodeURIComponent(downloadPath)}`,
              { responseType: "blob" }
            );

            const videoUrl = URL.createObjectURL(downloadRes.data);
            setOutputUrl(videoUrl); // ✅ show download button now
          }
        } catch (err) {
          console.error("SSE parse error:", err);
        }
      };

      evtSource.onerror = (err) => {
        console.error("SSE connection error:", err);
        evtSource.close();
        setStatusMsg("Connection lost. Please retry.");
      };
    } catch (err) {
      console.error(err);
      alert("Translation failed. Please try again.");
      setIsProcessing(false);
    }
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", mt: 4 }}>
      <Typography
  variant="h5"
  fontWeight="bold"
  sx={{ mb: 2, color: "white" }}
>
  Video Translation Panel
</Typography>

<Typography sx={{ mb: 1, color: "white" }}>
  SSE Status: {connected ? "🟢 Connected" : "🔴 Disconnected"}
</Typography>


      <Box sx={{ display: "flex", gap: 4, alignItems: "flex-start", justifyContent: "center", flexWrap: "wrap" }}>
        {/* Upload Panel */}
        <motion.div
          animate={isProcessing ? "processing" : outputUrl ? "output" : "idle"}
          variants={{
            idle: { scale: 1, x: 0 },
            processing: { scale: 0.95, x: -40 },
            output: { scale: 0.9, x: -80 },
          }}
        >
          {/* --- Upload Card --- */}
          <Card sx={{
            backgroundColor: "#1E1E1E",
            color: "#fff",
            borderRadius: "20px",
            boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
            p: 3,
            width: 350,
          }}>
            <Typography variant="h6" fontWeight="bold" gutterBottom>
              Upload Video
            </Typography>

            <Box
              sx={{
                border: previewUrl ? "none" : "2px dashed rgba(255,255,255,0.3)",
                borderRadius: "16px",
                aspectRatio: "16/9",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                backgroundColor: previewUrl ? "#000" : "rgba(255,255,255,0.05)",
                cursor: "pointer",
                mb: 3,
              }}
              onClick={() => document.getElementById("video-input").click()}
            >
              {!previewUrl ? (
                <Box textAlign="center">
                  <CloudUploadIcon sx={{ fontSize: 60, color: "rgba(255,255,255,0.7)" }} />
                  <Typography variant="body1" sx={{ opacity: 0.7 }}>
                    Click or Drag & Drop
                  </Typography>
                </Box>
              ) : (
                <video src={previewUrl} controls style={{ width: "100%", height: "100%", borderRadius: "16px" }} />
              )}
            </Box>
            <input id="video-input" type="file" accept="video/*" style={{ display: "none" }} onChange={handleFileChange} />

            <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", justifyContent: "center" }}>
              <Select
                value={sourceLang}
                onChange={(e) => setSourceLang(e.target.value)}
                size="small"
                sx={{ backgroundColor: "#2a2a2a", color: "#fff", borderRadius: "8px", minWidth: 130 }}
              >
                {LANGUAGES.map((l) => <MenuItem key={l} value={l}>{l}</MenuItem>)}
              </Select>
              <Select
                value={targetLang}
                onChange={(e) => setTargetLang(e.target.value)}
                size="small"
                sx={{ backgroundColor: "#2a2a2a", color: "#fff", borderRadius: "8px", minWidth: 130 }}
              >
                {LANGUAGES.map((l) => <MenuItem key={l} value={l}>{l}</MenuItem>)}
              </Select>
              <Button
                variant="contained"
                onClick={handleTranslateClick}
                disabled={isProcessing}
                sx={{
                  backgroundColor: "#2D89FF",
                  borderRadius: "12px",
                  px: 3,
                  "&:hover": { backgroundColor: "#176BEF" },
                }}
              >
                {isProcessing ? <CircularProgress size={20} color="inherit" /> : "Translate"}
              </Button>
            </Box>
          </Card>
        </motion.div>

        {/* Output Panel */}
        <AnimatePresence>
          {(isProcessing || outputUrl) && (
            <motion.div
              key={isProcessing ? "processing" : "done"}
              initial={{ opacity: 0, x: 80 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 80 }}
            >
              <Card sx={{
                backgroundColor: "#1E1E1E",
                color: "#fff",
                borderRadius: "20px",
                boxShadow: "0 4px 25px rgba(0,0,0,0.5)",
                p: 3,
                width: 420,
                minHeight: "300px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
              }}>
                <Typography variant="h6" fontWeight="bold" gutterBottom>
                  Output Video
                </Typography>

                {isProcessing && (
                  <>
                    <Typography variant="body1" sx={{ mt: 2 }}>{statusMsg}</Typography>
                    <LinearProgress
                      variant="determinate"
                      value={progress}
                      sx={{
                        width: "80%",
                        mt: 3,
                        height: 10,
                        borderRadius: 5,
                        backgroundColor: "#333",
                        "& .MuiLinearProgress-bar": { backgroundColor: "#2D89FF" },
                      }}
                    />
                    <Typography variant="body2" sx={{ mt: 1 }}>{progress}% completed</Typography>
                  </>
                )}

                {outputUrl && !isProcessing && (
                  <>
                    <Box sx={{
                      width: "100%",
                      aspectRatio: "16/9",
                      mb: 2,
                      borderRadius: "16px",
                      overflow: "hidden",
                      backgroundColor: "#000",
                    }}>
                      <video src={outputUrl} controls style={{ width: "100%", height: "100%" }} />
                    </Box>
                    <Box sx={{ display: "flex", gap: 2 }}>
                      <Button
                        variant="contained"
                        sx={{ backgroundColor: "#2D89FF", "&:hover": { backgroundColor: "#176BEF" } }}
                        onClick={() => {
                          const a = document.createElement("a");
                          a.href = outputUrl;
                          a.download = "translated_video.mp4";
                          a.click();
                        }}
                      >
                        Download
                      </Button>
                      <Button
                        variant="outlined"
                        sx={{ borderColor: "#FF6B6B", color: "#FF6B6B" }}
                        onClick={() => setOutputUrl(null)}
                      >
                        Delete
                      </Button>
                    </Box>
                  </>
                )}
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      </Box>

      {/* Event log */}
      <Box sx={{ mt: 5, maxWidth: 700 }}>
        <Typography variant="h6">Incoming events (most recent first)</Typography>
        <ul>
          {events.map((ev, idx) => (
            <li key={idx} style={{ marginBottom: 8 }}>
              <pre style={{ margin: 0 }}>{JSON.stringify(ev, null, 2)}</pre>
            </li>
          ))}
        </ul>
      </Box>
    </Box>
  );
}
