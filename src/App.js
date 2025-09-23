import React, { useState } from "react";
import { Box,Typography } from "@mui/material";
import Sidebar from "./components/Sidebar";
import HomeScreen from "./screens/HomeScreen";
import AboutScreen from "./screens/AboutScreen";
import ProfileScreen from "./screens/ProfileScreen";

export default function App() {
  const [activeNav, setActiveNav] = useState("Home");

  const renderScreen = () => {
    switch (activeNav) {
      case "Home":
        return <HomeScreen />;
      case "About":
        return <AboutScreen />;
      case "Profile":
        return <ProfileScreen />;
      default:
        return <HomeScreen />;
    }
  };

  return (

    
    <Box sx={{ display: "flex", bgcolor: "#121212", minHeight: "100vh" }}>
      {/* Left Sidebar */}
      <Sidebar active={activeNav} onNav={setActiveNav} />

      {/* Main Content */}
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
  {/* App Title */}
  <Box sx={{ mb: 4 }}>
    <Typography variant="h4" sx={{ color: "#fff", fontWeight: "bold" }}>
      VidTranscribe
    </Typography>
  </Box>

  {/* Screen content */}
  {renderScreen()}
</Box>
    </Box>
  );
}