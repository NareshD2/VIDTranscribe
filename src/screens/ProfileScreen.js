import React from "react";
import { Typography, Box, Card } from "@mui/material";

export default function ProfileScreen() {
  // Dynamically import all video files from "public/videos"
  const videos = require.context("../../public/videos", false, /\.(mp4|webm|ogg)$/);
  const videoList = videos.keys().map((key, index) => ({
    id: index + 1,
    src: `/videos/${key.replace("./", "")}`,
  }));

  return (
    <Box sx={{ color: "#fff", p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Profile
      </Typography>
      <Typography gutterBottom>
        Authentication & profile management are on hold for now.
      </Typography>

      <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
        Your Uploaded Videos
      </Typography>

      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
        {videoList.map((video) => (
          <Card
            key={video.id}
            sx={{
              width: 320,
              bgcolor: "#1e1e1e",
              borderRadius: 2,
              boxShadow: 3,
              overflow: "hidden",
            }}
          >
            <video
              width="100%"
              height="200"
              controls
              src={video.src}
              style={{ display: "block" }}
            />
          </Card>
        ))}
      </Box>
    </Box>
  );
}
