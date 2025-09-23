import React, { useState, useRef } from "react";
import VideoTranslatePanel from "../components/VideoTranslatePanel";

export default function HomeScreen() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [sourceLang, setSourceLang] = useState("English");
  const [targetLang, setTargetLang] = useState("Spanish");
  const [isProcessing, setIsProcessing] = useState(false);
  const [translatedUrl, setTranslatedUrl] = useState(null);
  const translatedBlobRef = useRef(null);

  const handleFileSelected = (f) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    if (translatedUrl) {
      URL.revokeObjectURL(translatedUrl);
      setTranslatedUrl(null);
    }
  };

  const handleTranslate = async () => {
    if (!file) return alert("Upload a video first.");
    setIsProcessing(true);

    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("sourceLang", sourceLang);
      fd.append("targetLang", targetLang);

      const resp = await fetch("/api/translate", { method: "POST", body: fd });
      if (!resp.ok) throw new Error("Server error: " + resp.status);
      const blob = await resp.blob();
      if (translatedUrl) URL.revokeObjectURL(translatedUrl);
      const tUrl = URL.createObjectURL(blob);
      translatedBlobRef.current = blob;
      setTranslatedUrl(tUrl);
    } catch (err) {
      console.error(err);
      alert("Translation failed: " + err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (!translatedBlobRef.current) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(translatedBlobRef.current);
    a.download = `translated_${file ? file.name.replace(/\.[^.]+$/, "") : "video"}.mp4`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  const handleDelete = () => {
    if (translatedUrl) URL.revokeObjectURL(translatedUrl);
    translatedBlobRef.current = null;
    setTranslatedUrl(null);
  };

  return (
    <VideoTranslatePanel
      previewUrl={previewUrl}
      onFileSelected={handleFileSelected}
      sourceLang={sourceLang}
      targetLang={targetLang}
      setSourceLang={setSourceLang}
      setTargetLang={setTargetLang}
      onTranslate={handleTranslate}
      isProcessing={isProcessing}
      translatedUrl={translatedUrl}
      onDownload={handleDownload}
      onDelete={handleDelete}
    />
  );
}
