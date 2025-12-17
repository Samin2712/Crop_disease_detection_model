const API_URL = "http://localhost:5000/predict";

const cropSelect = document.getElementById("cropSelect");

const fileInput = document.getElementById("file");
const imgPreview = document.getElementById("imgPreview");
const croppedPreview = document.getElementById("croppedPreview");

const cropBtn = document.getElementById("cropBtn");
const predictBtn = document.getElementById("predictBtn");

const statusEl = document.getElementById("status");
const predEl = document.getElementById("pred");
const confEl = document.getElementById("conf");

// optional topk
const topkWrap = document.getElementById("topkWrap");
const topkList = document.getElementById("topkList");

let cropper = null;
let croppedBlob = null;

function resetTopk() {
  if (!topkWrap || !topkList) return;
  topkList.innerHTML = "";
  topkWrap.style.display = "none";
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  predEl.textContent = "—";
  confEl.textContent = "—";
  statusEl.textContent = "";
  croppedBlob = null;
  resetTopk();

  croppedPreview.style.display = "none";
  croppedPreview.src = "";

  if (!file) return;

  const url = URL.createObjectURL(file);
  imgPreview.src = url;
  imgPreview.style.display = "block";

  // reset cropper
  if (cropper) cropper.destroy();

  cropper = new Cropper(imgPreview, {
    viewMode: 1,
    autoCropArea: 0.8,
    responsive: true,
    background: false
  });

  cropBtn.disabled = false;
  predictBtn.disabled = true;
});

cropBtn.addEventListener("click", async () => {
  if (!cropper) return;

  statusEl.textContent = "Cropping...";
  predEl.textContent = "—";
  confEl.textContent = "—";
  resetTopk();

  const canvas = cropper.getCroppedCanvas({
    width: 224,
    height: 224
  });

  canvas.toBlob((blob) => {
    croppedBlob = blob;

    const previewUrl = URL.createObjectURL(blob);
    croppedPreview.src = previewUrl;
    croppedPreview.style.display = "block";

    statusEl.textContent = "Cropped ✅ Now click Predict.";
    predictBtn.disabled = false;
  }, "image/jpeg", 0.95);
});

predictBtn.addEventListener("click", async () => {
  if (!croppedBlob) {
    statusEl.textContent = "Please crop the leaf first.";
    return;
  }

  const crop = cropSelect.value; // ✅ NEW

  predictBtn.disabled = true;
  cropBtn.disabled = true;
  statusEl.textContent = `Uploading cropped leaf & predicting (${crop})...`;
  resetTopk();

  try {
    const formData = new FormData();
    formData.append("crop", crop);                 // ✅ NEW
    formData.append("image", croppedBlob, "leaf.jpg");

    const res = await fetch(API_URL, {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (!res.ok || data?.error) throw new Error(data?.error || "Prediction failed");

    predEl.textContent = data.prediction ?? "Unknown";
    confEl.textContent = (data.confidence == null)
      ? "N/A"
      : (data.confidence * 100).toFixed(2) + "%";

    // Optional: show topk list if backend returns it
    if (data.topk && Array.isArray(data.topk) && data.topk.length > 0 && topkWrap && topkList) {
      topkList.innerHTML = "";
      data.topk.slice(0, 5).forEach((t) => {
        const li = document.createElement("li");
        const pct = (typeof t.score === "number") ? (t.score * 100).toFixed(2) + "%" : "N/A";
        li.textContent = `${t.label} — ${pct}`;
        topkList.appendChild(li);
      });
      topkWrap.style.display = "block";
    }

    statusEl.textContent = "Done ✅";
  } catch (e) {
    statusEl.textContent = "Error: " + e.message;
  } finally {
    predictBtn.disabled = false;
    cropBtn.disabled = false;
  }
});
