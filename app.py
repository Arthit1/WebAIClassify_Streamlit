from __future__ import annotations

import csv
import re
import zipfile
from collections import Counter, defaultdict
from io import BytesIO, StringIO
from pathlib import Path, PurePosixPath

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_efficientnet_b0.pth"

IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp"]
DEFAULT_IMAGE_SIZE = 320
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 16

st.set_page_config(
    page_title="Image Classification - 7 Types",
    page_icon="🖼️",
    layout="wide",
)


# ============================================================
# MODEL
# ============================================================
def _torch_load_checkpoint(model_path: Path, device: torch.device):
    """Load checkpoint while staying compatible with multiple PyTorch versions."""
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _torch_load_checkpoint(Path(model_path), device)

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint ไม่มี key 'model_state_dict'")

    classes = list(checkpoint.get("classes", []))
    model_config = checkpoint.get("model_config", {}) or {}

    state_dict = checkpoint["model_state_dict"]
    classifier_key = "classifier.1.weight"
    if classifier_key not in state_dict:
        raise ValueError(f"Checkpoint ไม่มี key '{classifier_key}'")

    num_outputs = state_dict[classifier_key].shape[0]

    if not classes:
        classes = [f"type_{i}" for i in range(num_outputs)]
    elif len(classes) < num_outputs:
        classes.extend(f"unused_{i}" for i in range(len(classes), num_outputs))

    model_name = model_config.get("name", "efficientnet_b0")
    if model_name != "efficientnet_b0":
        raise ValueError(
            f"เว็บเวอร์ชันนี้รองรับ efficientnet_b0 แต่ checkpoint ระบุ '{model_name}'"
        )

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_outputs)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    image_size = int(model_config.get("image_size", DEFAULT_IMAGE_SIZE))
    mean = model_config.get("normalize_mean", DEFAULT_MEAN)
    std = model_config.get("normalize_std", DEFAULT_STD)

    image_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return model, device, classes, image_transform, model_config


# ============================================================
# HELPERS
# ============================================================
def safe_basename(uploaded_name: str) -> str:
    """Return a safe filename only; never trust uploaded paths for filesystem writes."""
    normalized = uploaded_name.replace("\\", "/")
    name = PurePosixPath(normalized).name
    name = re.sub(r"[^A-Za-z0-9._()\- ]+", "_", name).strip()
    return name or "image.jpg"


def unique_name(filename: str, used_names: set[str]) -> str:
    """Prevent duplicate filenames inside the same predicted class folder."""
    path = Path(filename)
    stem = path.stem
    suffix = path.suffix
    candidate = filename
    number = 2

    while candidate.lower() in used_names:
        candidate = f"{stem}_{number}{suffix}"
        number += 1

    used_names.add(candidate.lower())
    return candidate


def make_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def make_excel_bytes(df: pd.DataFrame, classes: list[str]) -> bytes:
    output = BytesIO()
    counts = df["predicted"].value_counts().reindex(classes, fill_value=0)
    summary_df = pd.DataFrame(
        {
            "class": classes,
            "image_count": [int(counts.get(class_name, 0)) for class_name in classes],
        }
    )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        results_ws = writer.book["Results"]
        summary_ws = writer.book["Summary"]

        for ws in (results_ws, summary_ws):
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for column_cells in ws.columns:
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                ws.column_dimensions[column_cells[0].column_letter].width = min(
                    max(max_length + 2, 12), 60
                )

    return output.getvalue()


def build_zip_bytes(
    results_df: pd.DataFrame,
    image_records: list[dict],
    classes: list[str],
    csv_bytes: bytes,
    excel_bytes: bytes,
) -> bytes:
    output = BytesIO()
    used_names_by_class: dict[str, set[str]] = defaultdict(set)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Create all class folders, even when a class has zero images.
        for class_name in classes:
            zf.writestr(f"{class_name}/", b"")

        for record in image_records:
            predicted = record["predicted"]
            output_name = unique_name(
                safe_basename(record["original_name"]),
                used_names_by_class[predicted],
            )
            zf.writestr(f"{predicted}/{output_name}", record["bytes"])

        zf.writestr("prediction_results.csv", csv_bytes)
        zf.writestr("prediction_results.xlsx", excel_bytes)

    return output.getvalue()


def predict_uploaded_images(
    uploaded_files,
    model,
    device,
    classes,
    image_transform,
    batch_size: int = BATCH_SIZE,
):
    valid_records = []
    error_records = []

    progress = st.progress(0, text="กำลังตรวจสอบไฟล์รูปภาพ...")
    total_files = len(uploaded_files)

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        original_name = uploaded_file.name
        raw_bytes = uploaded_file.getvalue()

        try:
            with Image.open(BytesIO(raw_bytes)) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                tensor = image_transform(img)

            valid_records.append(
                {
                    "original_name": original_name,
                    "bytes": raw_bytes,
                    "tensor": tensor,
                }
            )
        except Exception as exc:
            error_records.append(
                {
                    "image": original_name,
                    "error": str(exc),
                }
            )

        progress.progress(
            int(index / max(total_files, 1) * 25),
            text=f"กำลังอ่านรูป {index}/{total_files}",
        )

    if not valid_records:
        progress.empty()
        return pd.DataFrame(), [], error_records

    result_rows = []
    image_records = []
    number_of_batches = (len(valid_records) + batch_size - 1) // batch_size

    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(valid_records), batch_size), start=1):
            batch_records = valid_records[start : start + batch_size]
            images = torch.stack([record["tensor"] for record in batch_records]).to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted_indexes = torch.max(probabilities, dim=1)

            confidences = confidences.detach().cpu().tolist()
            predicted_indexes = predicted_indexes.detach().cpu().tolist()

            for record, class_index, confidence in zip(
                batch_records, predicted_indexes, confidences
            ):
                predicted = classes[class_index]
                confidence_percent = round(float(confidence) * 100, 2)

                result_rows.append(
                    {
                        "image": record["original_name"],
                        "predicted": predicted,
                        "confidence": confidence_percent,
                    }
                )

                image_records.append(
                    {
                        "original_name": record["original_name"],
                        "predicted": predicted,
                        "confidence": confidence_percent,
                        "bytes": record["bytes"],
                    }
                )

            progress_value = 25 + int(batch_index / max(number_of_batches, 1) * 75)
            progress.progress(
                min(progress_value, 100),
                text=f"กำลัง Classification ชุดที่ {batch_index}/{number_of_batches}",
            )

    progress.progress(100, text="Classification เสร็จเรียบร้อย")
    return pd.DataFrame(result_rows), image_records, error_records


# ============================================================
# UI
# ============================================================
st.title("🖼️ Image Classification - 7 Types")
st.caption(
    "อัปโหลดรูปหลายไฟล์หรือทั้งโฟลเดอร์ → Classification → ดาวน์โหลด ZIP แยกตาม Class พร้อม CSV/Excel"
)

if not MODEL_PATH.exists():
    st.error(
        "ไม่พบไฟล์โมเดล `best_efficientnet_b0.pth` กรุณาวางไฟล์โมเดลไว้โฟลเดอร์เดียวกับ `app.py`"
    )
    st.stop()

try:
    model, device, classes, image_transform, model_config = load_model(str(MODEL_PATH))
except Exception as exc:
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("Model")
    st.write(f"**Device:** `{device}`")
    st.write(f"**Classes:** {len(classes)}")
    st.write(", ".join(f"`{class_name}`" for class_name in classes))
    st.write(f"**Image size:** {model_config.get('image_size', DEFAULT_IMAGE_SIZE)} × {model_config.get('image_size', DEFAULT_IMAGE_SIZE)}")
    st.caption("ชื่อ Class และ preprocessing อ่านจาก checkpoint โดยตรง")

upload_mode = st.radio(
    "รูปแบบการอัปโหลด",
    ["อัปโหลดหลายรูป", "อัปโหลดทั้งโฟลเดอร์"],
    horizontal=True,
)

if upload_mode == "อัปโหลดหลายรูป":
    uploaded_files = st.file_uploader(
        "เลือกรูปภาพ",
        type=IMAGE_EXTENSIONS,
        accept_multiple_files=True,
        help="รองรับ JPG, JPEG, PNG, WEBP และ BMP",
    )
else:
    uploaded_files = st.file_uploader(
        "เลือกโฟลเดอร์รูปภาพ",
        type=IMAGE_EXTENSIONS,
        accept_multiple_files="directory",
        help="ระบบจะรับรูปในโฟลเดอร์และ subfolder ทั้งหมด",
    )

if uploaded_files:
    st.info(f"พบไฟล์ที่เลือกทั้งหมด **{len(uploaded_files):,} รูป**")

    col1, col2 = st.columns([1, 3])
    with col1:
        process_clicked = st.button(
            "▶ เริ่ม Classification",
            type="primary",
            use_container_width=True,
        )
    with col2:
        st.caption(
            "เมื่อกดเริ่ม ระบบจะประมวลผลเป็น batch แล้วสร้าง ZIP + CSV + Excel ให้ดาวน์โหลด"
        )

    if process_clicked:
        results_df, image_records, errors = predict_uploaded_images(
            uploaded_files=uploaded_files,
            model=model,
            device=device,
            classes=classes,
            image_transform=image_transform,
        )

        if results_df.empty:
            st.error("ไม่พบรูปภาพที่สามารถประมวลผลได้")
            if errors:
                st.dataframe(pd.DataFrame(errors), use_container_width=True)
        else:
            csv_bytes = make_csv_bytes(results_df)
            excel_bytes = make_excel_bytes(results_df, classes)
            zip_bytes = build_zip_bytes(
                results_df,
                image_records,
                classes,
                csv_bytes,
                excel_bytes,
            )

            st.session_state["classification_results"] = {
                "df": results_df,
                "errors": errors,
                "csv": csv_bytes,
                "excel": excel_bytes,
                "zip": zip_bytes,
            }

if "classification_results" in st.session_state:
    result = st.session_state["classification_results"]
    results_df = result["df"]

    st.divider()
    st.subheader("ผลการ Classification")

    total = len(results_df)
    avg_confidence = results_df["confidence"].mean() if total else 0
    class_counts = results_df["predicted"].value_counts().reindex(classes, fill_value=0)

    metric_cols = st.columns(3)
    metric_cols[0].metric("จำนวนรูปทั้งหมด", f"{total:,}")
    metric_cols[1].metric("Average confidence", f"{avg_confidence:.2f}%")
    metric_cols[2].metric("จำนวน Class ที่พบ", f"{int((class_counts > 0).sum())}/{len(classes)}")

    st.markdown("#### จำนวนรูปแยกตาม Class")
    count_df = pd.DataFrame(
        {
            "class": classes,
            "image_count": [int(class_counts[class_name]) for class_name in classes],
        }
    )
    st.dataframe(count_df, use_container_width=True, hide_index=True)

    st.markdown("#### รายละเอียดผลลัพธ์")
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "image": st.column_config.TextColumn("Image"),
            "predicted": st.column_config.TextColumn("Predicted"),
            "confidence": st.column_config.NumberColumn(
                "Confidence (%)", format="%.2f%%"
            ),
        },
    )

    st.markdown("#### ดาวน์โหลดผลลัพธ์")
    download_cols = st.columns(3)
    download_cols[0].download_button(
        "⬇ ดาวน์โหลด ZIP",
        data=result["zip"],
        file_name="classified_images.zip",
        mime="application/zip",
        use_container_width=True,
        type="primary",
    )
    download_cols[1].download_button(
        "⬇ ดาวน์โหลด CSV",
        data=result["csv"],
        file_name="prediction_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_cols[2].download_button(
        "⬇ ดาวน์โหลด Excel",
        data=result["excel"],
        file_name="prediction_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    if result["errors"]:
        with st.expander(f"ไฟล์ที่อ่านไม่ได้ ({len(result['errors'])} ไฟล์)"):
            st.dataframe(
                pd.DataFrame(result["errors"]),
                use_container_width=True,
                hide_index=True,
            )
