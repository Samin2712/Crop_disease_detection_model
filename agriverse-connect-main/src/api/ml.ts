// src/api/ml.ts
import apiClient from "@/api/client";

export type MlTopK = { label: string; score: number };
export type MlPredictionResponse = {
  crop?: string;
  model?: string;
  prediction?: string;
  confidence?: number;
  topk?: MlTopK[];
  error?: string;
};

export type PredictAndCreateResponse = {
  prediction?: MlPredictionResponse;
  advice?: string | null;

  // backend returns a request object (UserRequestResponse)
  request?: { id: number } & Record<string, any>;
};

export async function mlPredict(crop: string, image: Blob) {
  const form = new FormData();
  form.append("crop", crop);
  form.append("image", image, "leaf.jpg");

  const res = await apiClient.post<MlPredictionResponse>("/api/ml/predict", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export async function mlPredictAndCreateRequest(
  crop: string,
  image: Blob,
  state?: string,
  district?: string
) {
  const form = new FormData();
  form.append("crop", crop);
  form.append("image", image, "leaf.jpg");
  if (state) form.append("state", state);
  if (district) form.append("district", district);

  const res = await apiClient.post<PredictAndCreateResponse>("/api/ml/predict-and-create", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
  
}

export async function mlAdvice(cropName: string, diseaseName: string) {
  const res = await apiClient.post<{ answer: string | null }>(
    "/api/ml/advice",
    {
      crop_name: cropName,
      disease_name: diseaseName,
    }
  );
  return res.data;
}

export async function mlForwardToGovtOfficer(
  crop: string,
  diseaseName: string,
  advice: string,
  image: Blob,
  state?: string,
  district?: string
) {
  const form = new FormData();
  form.append("crop", crop);
  form.append("diseaseName", diseaseName);
  form.append("advice", advice);
  form.append("image", image, "leaf.jpg");
  if (state) form.append("state", state);
  if (district) form.append("district", district);

  const res = await apiClient.post("/api/ml/forward", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data; // UserRequestResponse
}

