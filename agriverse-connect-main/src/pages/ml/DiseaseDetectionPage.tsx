import { useMemo, useRef, useState } from "react";
import { Cropper } from "react-cropper";
import type { ReactCropperElement } from "react-cropper";
import "cropperjs/dist/cropper.css";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";

import {
  mlAdvice,
  mlPredict,
  mlForwardToGovtOfficer,
  MlPredictionResponse,
} from "@/api/ml";

import { Loader2 } from "lucide-react";
import { useNavigate } from "react-router-dom";

type StatusVariant = "default" | "success" | "error" | "loading";

const crops = [
  { value: "rice", label: "Rice" },
  { value: "jute", label: "Jute" },
  { value: "potato", label: "Potato" },
  { value: "tomato", label: "Tomato" },
];

function pct(x?: number) {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(2)}%`;
}

export default function DiseaseDetectionPage() {
  const navigate = useNavigate();
  const cropperRef = useRef<ReactCropperElement>(null);

  const [crop, setCrop] = useState(crops[0].value);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");

  const [croppedBlob, setCroppedBlob] = useState<Blob | null>(null);
  const [croppedUrl, setCroppedUrl] = useState("");

  const [prediction, setPrediction] =
    useState<MlPredictionResponse | null>(null);

  const [advice, setAdvice] = useState("—");
  const [adviceOpen, setAdviceOpen] = useState(false);
  const [adviceLoading, setAdviceLoading] = useState(false);
  const [adviceReady, setAdviceReady] = useState(false);
  const [adviceStatus, setAdviceStatus] = useState("");

  const [statusMsg, setStatusMsg] = useState("");
  const [statusVariant, setStatusVariant] =
    useState<StatusVariant>("default");

  const [busyPredict, setBusyPredict] = useState(false);
  const [busyForward, setBusyForward] = useState(false);

  const canCrop = !!previewUrl && !busyPredict && !busyForward;
  const canPredict = !!croppedBlob && !busyPredict && !busyForward;

  const cropLabel = useMemo(
    () => crops.find((c) => c.value === crop)?.label ?? crop,
    [crop]
  );

  const diseaseLabel = prediction?.prediction ?? "—";
  const top5 = prediction?.topk?.slice(0, 5) ?? [];

  function resetForNewImage() {
    setCroppedBlob(null);
    if (croppedUrl) URL.revokeObjectURL(croppedUrl);
    setCroppedUrl("");

    setPrediction(null);
    setAdvice("—");
    setAdviceOpen(false);
    setAdviceReady(false);
    setAdviceStatus("");
  }

  function onPickFile(f: File | null) {
    setFile(f);
    resetForNewImage();

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(f ? URL.createObjectURL(f) : "");

    setStatusMsg(
      f ? "Image loaded. Adjust crop and click “Crop Leaf”." : ""
    );
    setStatusVariant("default");
  }

  async function doCrop() {
    const cropper = cropperRef.current?.cropper;
    if (!cropper) return;

    setStatusMsg("Cropping…");
    setStatusVariant("loading");

    const canvas = cropper.getCroppedCanvas({ width: 224, height: 224 });

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setStatusMsg("Crop failed.");
          setStatusVariant("error");
          return;
        }

        setCroppedBlob(blob);
        if (croppedUrl) URL.revokeObjectURL(croppedUrl);
        setCroppedUrl(URL.createObjectURL(blob));

        setStatusMsg("Cropped ✅ Now click “Predict”.");
        setStatusVariant("success");
      },
      "image/jpeg",
      0.95
    );
  }

  async function doPredict() {
    if (!croppedBlob) return;

    setBusyPredict(true);
    setStatusMsg(`Predicting (${crop})…`);
    setStatusVariant("loading");

    setAdvice("—");
    setAdviceOpen(false);
    setAdviceReady(false);

    try {
      const res = await mlPredict(crop, croppedBlob);
      if (res?.error) throw new Error(res.error);

      setPrediction(res);
      setAdviceOpen(true);
      setStatusMsg("Prediction done ✅");
      setStatusVariant("success");
    } catch (e: any) {
      setStatusMsg(e.message || "Prediction failed");
      setStatusVariant("error");
    } finally {
      setBusyPredict(false);
    }
  }

  async function doAdvice() {
    if (!prediction?.prediction) return;

    setAdviceLoading(true);
    setAdvice("লোড হচ্ছে...");
    setAdviceStatus("");

    try {
      const res = await mlAdvice(cropLabel, prediction.prediction);
      setAdvice(res.answer ?? "কোনো উত্তর পাওয়া যায়নি");
      setAdviceReady(true);
      setAdviceStatus("সম্পন্ন ✅");
    } catch {
      setAdvice("AI advice failed.");
      setAdviceStatus("ব্যর্থ ❌");
    } finally {
      setAdviceLoading(false);
    }
  }

  async function doForward() {
    if (!croppedBlob || !prediction?.prediction || !adviceReady) return;

    setBusyForward(true);
    setStatusMsg("Forwarding to govt officer…");
    setStatusVariant("loading");

    try {
      const res = await mlForwardToGovtOfficer(
        cropLabel,
        prediction.prediction,
        advice,
        croppedBlob
      );

      const id = (res as any)?.id;
      setStatusMsg("Forwarded ✅");
      setStatusVariant("success");

      if (id) navigate(`/requests/${id}`);
    } catch (e: any) {
      setStatusMsg(e.message || "Forward failed");
      setStatusVariant("error");
    } finally {
      setBusyForward(false);
    }
  }

  return (
    <div className="min-h-screen bg-background px-4 py-6">
      <div className="mx-auto max-w-5xl space-y-6">
        <header className="flex items-center gap-3">
          <div className="h-11 w-11 rounded-2xl bg-secondary border flex items-center justify-center">
            🌿
          </div>
          <div>
            <h1 className="text-2xl font-semibold">Plant Disease Detection</h1>
            <p className="text-sm text-muted-foreground">
              Upload → Crop → Predict → AI → Forward
            </p>
          </div>
        </header>

        <Card>
          <CardContent className="space-y-5 p-6">
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <Label>Select Crop</Label>
                <select
                value={crop}
                onChange={(e) => setCrop(e.target.value)}
                className="w-full h-10 rounded-md border border-border bg-background px-3 text-sm
                            text-foreground focus:outline-none focus:ring-2 focus:ring-ring/30
                            disabled:opacity-70"
                >
                {crops.map((c) => (
                    <option
                    key={c.value}
                    value={c.value}
                    className="bg-green text-white"
                    >
                    {c.label}
                    </option>
                ))}
                </select>

              </div>

              <div>
                <Label>Upload Leaf Image</Label>
                <Input
                  type="file"
                  accept="image/*"
                  onChange={(e) => onPickFile(e.target.files?.[0] ?? null)}
                />
              </div>
            </div>

            {previewUrl && (
              <Cropper
                src={previewUrl}
                ref={cropperRef}
                style={{ height: 420, width: "100%" }}
                viewMode={1}
                autoCropArea={0.8}
                background={false}
              />
            )}

            <div className="flex gap-2 flex-wrap">
              <Button onClick={doCrop} disabled={!canCrop}>
                Crop Leaf
              </Button>

              <Button onClick={doPredict} disabled={!canPredict}>
                {busyPredict && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Predict
              </Button>

              <Button
                variant="outline"
                onClick={doAdvice}
                disabled={!prediction || adviceLoading}
              >
                {adviceLoading && (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                )}
                পরামর্শ দেখুন
              </Button>

              <Button
                variant="outline"
                onClick={doForward}
                disabled={!adviceReady || busyForward}
              >
                {busyForward && (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                )}
                Forward it to a govt officer
              </Button>
            </div>

            {statusMsg && (
              <div className="text-sm text-muted-foreground">{statusMsg}</div>
            )}

            <Separator />

            <div className="space-y-3">
              <div>
                <strong>Prediction:</strong> {diseaseLabel}
              </div>
              <div>
                <strong>Confidence:</strong> {pct(prediction?.confidence)}
              </div>

              {top5.length > 0 && (
                <ul className="space-y-2">
                  {top5.map((t) => (
                    <li key={t.label} className="flex justify-between">
                      <span>{t.label}</span>
                      <span>{pct(t.score)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {adviceOpen && (
              <pre className="border rounded-md p-3 bg-muted">
                {advice}
              </pre>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
