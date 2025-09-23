import React from "react";
import { Box, Typography, Button, Divider } from "@mui/material";

export default function VideoOutput({ translatedUrl, onDownload, onDelete }) {
  return (
    <Box>
      <Divider sx={{ my: 3 }} />
      <Typography variant="h6" gutterBottom>Translated Output</Typography>

      {!translatedUrl && <Typography>No translated video yet.</Typography>}

      {translatedUrl && (
        <Box>
          <video src={translatedUrl} controls style={{ width: "100%", borderRadius: 8, marginBottom: "1rem" }} />
          <Box sx={{ display: "flex", gap: 2 }}>
            <Button variant="contained" color="success" onClick={onDownload}>Download</Button>
            <Button variant="outlined" color="error" onClick={onDelete}>Delete</Button>
          </Box>
        </Box>
      )}
    </Box>
  );
}
