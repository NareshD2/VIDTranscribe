import React from "react";
import { Typography, Box } from "@mui/material";

export default function AboutScreen() {
  return (
    <Box sx={{ color: "#fff" }}>
      <Typography variant="h5" gutterBottom>
        About This App
      </Typography>
      <Typography>
        This is a demo UI for a video translation software. Upload videos, select source and target languages, and translate video text in real-time.
      </Typography>
    </Box>
  );
}
