import React, { useState } from "react";
import VideoTranslatePanel from "../components/VideoTranslatePanel";

export default function HomeScreen() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [sourceLang, setSourceLang] = useState("English");
  const [targetLang, setTargetLang] = useState("Spanish");

  const handleFileSelected = (f) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  };

  return (
    <VideoTranslatePanel
      previewUrl={previewUrl}
      onFileSelected={handleFileSelected}
      sourceLang={sourceLang}
      targetLang={targetLang}
      setSourceLang={setSourceLang}
      setTargetLang={setTargetLang}
    />
  );
}
