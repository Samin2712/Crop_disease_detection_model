// AgriVerse-style JS: small helpers, consistent state updates, clearer status variants.
const API_URL = "http://localhost:5000/predict";

// n8n webhook (kept hidden from UI)
const N8N_WEBHOOK_URL =
  "https://n8n.srv915514.hstgr.cloud/webhook/a2b31e68-13cf-4edc-a106-987699018fdd";

const els = {
  cropSelect: document.getElementById("cropSelect"),
  fileInput: document.getElementById("file"),
  imgPreview: document.getElementById("imgPreview"),
  croppedPreview: document.getElementById("croppedPreview"),
  cropBtn: document.getElementById("cropBtn"),
  predictBtn: document.getElementById("predictBtn"),
  status: document.getElementById("status"),
  pred: document.getElementById("pred"),
  conf: document.getElementById("conf"),
  topkWrap: document.getElementById("topkWrap"),
  topkList: document.getElementById("topkList"),

  // n8n advice
  adviceSection: document.getElementById("adviceSection"),
  adviceBtn: document.getElementById("adviceBtn"),
  adviceOut: document.getElementById("adviceOut"),
  adviceStatus: document.getElementById("adviceStatus"),
  adviceCrop: document.getElementById("adviceCrop"),
  adviceDisease: document.getElementById("adviceDisease"),
};

let cropper = null;
let croppedBlob = null;
let lastPrediction = null;

function setAdviceStatus(message) {
  if (!els.adviceStatus) return;
  els.adviceStatus.textContent = message || "";
}

function resetAdvice() {
  lastPrediction = null;
  if (els.adviceSection) els.adviceSection.style.display = "none";
  if (els.adviceBtn) els.adviceBtn.disabled = true;
  if (els.adviceOut) els.adviceOut.textContent = "—";
  setAdviceStatus("");
  if (els.adviceCrop) els.adviceCrop.textContent = "—";
  if (els.adviceDisease) els.adviceDisease.textContent = "—";
}

function setStatus(message, variant = "default") {
  els.status.textContent = message || "";
  els.status.dataset.variant = variant; // default | success | error | loading
}

function setResult(prediction, confidence) {
  els.pred.textContent = prediction ?? "—";
  if (confidence == null || Number.isNaN(confidence)) {
    els.conf.textContent = "—";
  } else {
    els.conf.textContent = (confidence * 100).toFixed(2) + "%";
  }
}

function resetTopk() {
  if (!els.topkWrap || !els.topkList) return;
  els.topkList.innerHTML = "";
  els.topkWrap.style.display = "none";
}

function setButtons({ canCrop, canPredict, busy }) {
  // busy disables both
  if (busy) {
    els.cropBtn.disabled = true;
    els.predictBtn.disabled = true;
    return;
  }
  els.cropBtn.disabled = !canCrop;
  els.predictBtn.disabled = !canPredict;
}

function destroyCropper() {
  if (cropper) {
    cropper.destroy();
    cropper = null;
  }
}

function clearCroppedPreview() {
  els.croppedPreview.style.display = "none";
  els.croppedPreview.src = "";
  croppedBlob = null;
}

function clearPreview() {
  els.imgPreview.style.display = "none";
  els.imgPreview.src = "";
}

els.fileInput.addEventListener("change", () => {
  const file = els.fileInput.files?.[0];

  setResult("—", null);
  setStatus("");
  resetTopk();
  resetAdvice();
  clearCroppedPreview();
  destroyCropper();

  if (!file) {
    clearPreview();
    setButtons({ canCrop: false, canPredict: false, busy: false });
    return;
  }

  const url = URL.createObjectURL(file);
  els.imgPreview.src = url;
  els.imgPreview.style.display = "block";

  cropper = new Cropper(els.imgPreview, {
    viewMode: 1,
    autoCropArea: 0.8,
    responsive: true,
    background: false,
  });

  setButtons({ canCrop: true, canPredict: false, busy: false });
  setStatus("Image loaded. Adjust crop and click “Crop Leaf”.", "default");
});

els.cropBtn.addEventListener("click", async () => {
  if (!cropper) return;

  setStatus("Cropping…", "loading");
  setResult("—", null);
  resetTopk();
  resetAdvice();
  setButtons({ canCrop: true, canPredict: false, busy: true });

  const canvas = cropper.getCroppedCanvas({ width: 224, height: 224 });

  canvas.toBlob(
    (blob) => {
      croppedBlob = blob;

      const previewUrl = URL.createObjectURL(blob);
      els.croppedPreview.src = previewUrl;
      els.croppedPreview.style.display = "block";

      setStatus("Cropped ✅ Now click “Predict”.", "success");
      setButtons({ canCrop: true, canPredict: true, busy: false });
    },
    "image/jpeg",
    0.95
  );
});

els.predictBtn.addEventListener("click", async () => {
  if (!croppedBlob) {
    setStatus("Please crop the leaf first.", "error");
    return;
  }

  const crop = els.cropSelect.value;

  setButtons({ canCrop: false, canPredict: false, busy: true });
  setStatus(`Uploading & predicting (${crop})…`, "loading");
  resetTopk();
  resetAdvice();

  try {
    const formData = new FormData();
    formData.append("crop", crop);
    formData.append("image", croppedBlob, "leaf.jpg");

    const res = await fetch(API_URL, { method: "POST", body: formData });

    let data = null;
    try {
      data = await res.json();
    } catch {
      // non-json error
    }

    if (!res.ok || data?.error) {
      throw new Error(data?.error || `Request failed (HTTP ${res.status})`);
    }

    setResult(data.prediction ?? "Unknown", data.confidence);

    // Show n8n advice panel using selected crop + detected disease
    lastPrediction = {
      crop_value: crop,
      crop_label:
        els.cropSelect?.options?.[els.cropSelect.selectedIndex]?.text || crop,
      disease_label: data.prediction ?? "Unknown",
    };

    if (els.adviceSection) els.adviceSection.style.display = "block";
    if (els.adviceCrop) els.adviceCrop.textContent = lastPrediction.crop_label;
    if (els.adviceDisease) els.adviceDisease.textContent = lastPrediction.disease_label;
    if (els.adviceBtn) els.adviceBtn.disabled = false;
    if (els.adviceOut) els.adviceOut.textContent = "—";
    setAdviceStatus("");

    if (Array.isArray(data.topk) && data.topk.length && els.topkWrap && els.topkList) {
      els.topkList.innerHTML = "";
      data.topk.slice(0, 5).forEach((t) => {
        const li = document.createElement("li");
        const pct =
          typeof t.score === "number" ? (t.score * 100).toFixed(2) + "%" : "N/A";
        li.innerHTML = `<span class="topk-item">${t.label}</span><span class="topk-pct">${pct}</span>`;
        els.topkList.appendChild(li);
      });
      els.topkWrap.style.display = "block";
    }

    setStatus("Done ✅", "success");
  } catch (e) {
    setStatus(`Error: ${e.message}`, "error");
  } finally {
    setButtons({ canCrop: !!cropper, canPredict: !!croppedBlob, busy: false });
  }
});

// n8n advice request
if (els.adviceBtn) {
  els.adviceBtn.addEventListener("click", async () => {
    if (!lastPrediction) {
      setAdviceStatus("আগে Predict করুন।");
      return;
    }

    els.adviceBtn.disabled = true;
    if (els.adviceOut) els.adviceOut.textContent = "লোড হচ্ছে...";
    setAdviceStatus("");

    try {
      const payload = {
        crop_name: lastPrediction.crop_label,
        disease_name: lastPrediction.disease_label,
      };

      const res = await fetch(N8N_WEBHOOK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const contentType = res.headers.get("content-type") || "";
      const data = contentType.includes("application/json")
        ? await res.json()
        : { raw: await res.text() };

      if (!res.ok) {
        if (els.adviceOut) els.adviceOut.textContent = JSON.stringify(data, null, 2);
        setAdviceStatus(`HTTP ${res.status}`);
        return;
      }

      if (els.adviceOut) {
        els.adviceOut.textContent = data.answer ?? data.raw ?? "কোনো উত্তর পাওয়া যায়নি";
      }
      setAdviceStatus("সম্পন্ন ✅");
    } catch (err) {
      if (els.adviceOut) {
        els.adviceOut.textContent =
          "রিকোয়েস্ট ব্যর্থ হয়েছে। সম্ভবত CORS সমস্যা।\n\n" + err;
      }
      setAdviceStatus("ব্যর্থ ❌");
    } finally {
      els.adviceBtn.disabled = false;
    }
  });
}
