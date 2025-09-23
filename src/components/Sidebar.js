import React from "react";
import { Drawer, List, ListItemButton, ListItemIcon, Tooltip } from "@mui/material";
import HomeIcon from "@mui/icons-material/Home";
import InfoIcon from "@mui/icons-material/Info";
import AccountCircleIcon from "@mui/icons-material/AccountCircle";

export default function Sidebar({ active, onNav }) {
  const navItems = [
    { key: "Home", label: "Home", icon: <HomeIcon /> },
    { key: "About", label: "About", icon: <InfoIcon /> },
    { key: "Profile", label: "Profile", icon: <AccountCircleIcon /> },
  ];

  return (
    <Drawer
      variant="permanent"
      anchor="left"
      sx={{
        width: 80,
        flexShrink: 0,
        [`& .MuiDrawer-paper`]: {
          width: 80,
          borderTopRightRadius: 30,
          borderBottomRightRadius: 30,
          backgroundColor: "#1E1E1E",
          color: "#fff",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        },
      }}
    >
      <List
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 3,
        }}
      >
        {navItems.map((item) => (
          <Tooltip title={item.label} placement="right" key={item.key}>
            <ListItemButton
              onClick={() => onNav(item.key)}
              sx={{
                borderRadius: "50%",
                width: 50,
                height: 50,
                justifyContent: "center",
                backgroundColor: active === item.key ? "#2D89FF" : "transparent",
                "&:hover": {
                  backgroundColor: active === item.key ? "#2D89FF" : "rgba(255,255,255,0.1)",
                },
              }}
            >
              <ListItemIcon sx={{ color: "#fff", minWidth: 0, justifyContent: "center" }}>
                {item.icon}
              </ListItemIcon>
            </ListItemButton>
          </Tooltip>
        ))}
      </List>
    </Drawer>
  );
}
