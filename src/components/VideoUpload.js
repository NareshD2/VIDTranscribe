import React from "react";
import { Box, Typography, Button, Select, MenuItem, CircularProgress, Card } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import PlayCircleOutlineIcon from "@mui/icons-material/PlayCircleOutline";

const LANGUAGES = [
  "English", "Spanish", "French", "German", "Hindi", "Tamil", "Telugu", "Chinese", "Japanese", "Korean"
];

export default function VideoUpload({
  previewUrl,
  onFileSelected,
  sourceLang,
  targetLang,
  setSourceLang,
  setTargetLang,
  onTranslate,
  isProcessing
}) {

  function handleFileChange(e) {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.type.startsWith("video/")) {
      alert("Please upload a valid video file.");
      return;
    }
    onFileSelected(file);
  }

  return (
    <Card
      sx={{
        backgroundColor: "#1E1E1E",
        color: "#fff",
        borderRadius: "20px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
        p: 3,
        maxWidth: "700px",
        mx: "auto",
        mb: 4,
      }}
    >
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        Upload & Translate Video
      </Typography>

      {/* Upload Box */}
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
              Click or Drag & Drop to Upload
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
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2, justifyContent: "center" }}>
        <Box>
          <Typography variant="body2" sx={{ opacity: 0.8 }}>
            Source Language
          </Typography>
          <Select
            value={sourceLang}
            onChange={(e) => setSourceLang(e.target.value)}
            size="small"
            sx={{
              backgroundColor: "#2a2a2a",
              color: "#fff",
              borderRadius: "8px",
              minWidth: 140,
            }}
          >
            {LANGUAGES.map((l) => (
              <MenuItem key={l} value={l}>{l}</MenuItem>
            ))}
          </Select>
        </Box>

        <Box>
          <Typography variant="body2" sx={{ opacity: 0.8 }}>
            Target Language
          </Typography>
          <Select
            value={targetLang}
            onChange={(e) => setTargetLang(e.target.value)}
            size="small"
            sx={{
              backgroundColor: "#2a2a2a",
              color: "#fff",
              borderRadius: "8px",
              minWidth: 140,
            }}
          >
            {LANGUAGES.map((l) => (
              <MenuItem key={l} value={l}>{l}</MenuItem>
            ))}
          </Select>
        </Box>

        <Button
          variant="contained"
          onClick={onTranslate}
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
  );
}
