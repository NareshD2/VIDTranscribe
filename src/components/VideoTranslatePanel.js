import React, { useState } from "react";
import { Box, Card, Typography, Button, Select, MenuItem, CircularProgress } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import { motion, AnimatePresence } from "framer-motion";

const LANGUAGES = [
  "English", "Spanish", "French", "German", "Hindi", "Tamil",
  "Telugu", "Chinese", "Japanese", "Korean"
];

export default function VideoTranslatePanel({
  previewUrl,
  onFileSelected,
  sourceLang,
  targetLang,
  setSourceLang,
  setTargetLang,
}) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [dummyOutput, setDummyOutput] = useState(null);

  function handleFileChange(e) {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.type.startsWith("video/")) {
      alert("Please upload a valid video file.");
      return;
    }
    onFileSelected(file);
    setDummyOutput(null);
  }

  function handleTranslateClick() {
    if (!previewUrl) return alert("Upload a video first.");

    // Simulate processing
    setIsProcessing(true);
    setDummyOutput(null);

    setTimeout(() => {
      setDummyOutput(previewUrl); // simulate output
      setIsProcessing(false);
    }, 1500);
  }

  const cardVariants = {
    idle: { scale: 1, x: 0, opacity: 1 },
    processing: { scale: 0.85, x: -100, opacity: 0.95 }
  };

  const outputVariants = {
    hidden: { x: 300, opacity: 0 },
    visible: { x: 0, opacity: 1 }
  };

  return (
    <Box
      sx={{
        display: "flex",
        gap: 4,
        justifyContent: "center",
        alignItems: "flex-start",
        flexWrap: "wrap",
        mt: 4
      }}
    >
      {/* Upload Card */}
      <motion.div
        animate={isProcessing || dummyOutput ? "processing" : "idle"}
        variants={cardVariants}
        transition={{ type: "spring", stiffness: 300, damping: 25 }}
      >
        <Card
          sx={{
            backgroundColor: "#1E1E1E",
            color: "#fff",
            borderRadius: "20px",
            boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
            p: 3,
            width: 350,
          }}
        >
          <Typography variant="h6" fontWeight="bold" gutterBottom>
            Upload Video
          </Typography>

          {/* Upload Area */}
          <Box
            sx={{
              position: "relative",
              border: previewUrl ? "none" : "2px dashed rgba(255,255,255,0.3)",
              borderRadius: "16px",
              aspectRatio: "16/9",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              backgroundColor: previewUrl ? "#000" : "rgba(255,255,255,0.05)",
              cursor: "pointer",
              overflow: "hidden",
              mb: 3,
            }}
            onClick={() => document.getElementById("video-input").click()}
          >
            {!previewUrl && (
              <Box sx={{ textAlign: "center" }}>
                <CloudUploadIcon sx={{ fontSize: 60, color: "rgba(255,255,255,0.7)" }} />
                <Typography variant="body1" sx={{ opacity: 0.7 }}>
                  Click or Drag & Drop
                </Typography>
              </Box>
            )}
            {previewUrl && (
              <video
                src={previewUrl}
                controls
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  borderRadius: "16px",
                }}
              />
            )}
          </Box>
          <input
            id="video-input"
            type="file"
            accept="video/*"
            style={{ display: "none" }}
            onChange={handleFileChange}
          />

          {/* Language selectors + Translate */}
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
                textTransform: "none",
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

      {/* Output Card */}
      <AnimatePresence>
        {(isProcessing || dummyOutput) && (
          <motion.div
            initial="hidden"
            animate="visible"
            exit="hidden"
            variants={outputVariants}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            style={{ marginTop: 10 }} // output card slightly down
          >
            <Card
              sx={{
                backgroundColor: "#1E1E1E",
                color: "#fff",
                borderRadius: "20px",
                boxShadow: "0 4px 25px rgba(0,0,0,0.5)",
                p: 3,
                width: 420, // bigger than input card
                minHeight: "300px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                Output Video
              </Typography>

              {isProcessing && <CircularProgress size={50} sx={{ mt: 4 }} />}
              {dummyOutput && !isProcessing && (
                <>
                  <Box
                    sx={{
                      width: "100%",
                      aspectRatio: "16/9",
                      mb: 2,
                      borderRadius: "16px",
                      overflow: "hidden",
                      backgroundColor: "#000",
                    }}
                  >
                    <video
                      src={dummyOutput}
                      controls
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "cover",
                      }}
                    />
                  </Box>
                  <Box sx={{ display: "flex", gap: 2 }}>
                    <Button
                      variant="contained"
                      sx={{ backgroundColor: "#2D89FF", "&:hover": { backgroundColor: "#176BEF" } }}
                      onClick={() => {
                        if (!dummyOutput) return;
                        const a = document.createElement("a");
                        a.href = dummyOutput;
                        a.download = "translated_video.mp4";
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                      }}
                    >
                      Download
                    </Button>
                    <Button
                      variant="outlined"
                      sx={{ borderColor: "#FF6B6B", color: "#FF6B6B" }}
                      onClick={() => setDummyOutput(null)}
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
  );
}
