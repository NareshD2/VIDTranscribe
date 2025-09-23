import React from "react";
import { Typography, Box } from "@mui/material";

export default function ProfileScreen() {
  return (
    <Box sx={{ color: "#fff" }}>
      <Typography variant="h5" gutterBottom>
        Profile
      </Typography>
      <Typography>
        Authentication & profile management are on hold for now.
      </Typography>
    </Box>
  );
}
