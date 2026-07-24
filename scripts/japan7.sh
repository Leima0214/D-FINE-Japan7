#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${DFINE_CONFIG:-${ROOT_DIR}/configs/dfine/custom/dfine_hgnetv2_s_japan7.yml}"
DATA_ROOT="${JAPAN7_COCO_ROOT:-/COCO/Japan7-COCO}"
WEIGHTS="${DFINE_S_WEIGHTS:-${ROOT_DIR}/weights/dfine_s_coco.pth}"
OUTPUT_ROOT="${JAPAN7_OUTPUT_ROOT:-${ROOT_DIR}/outputs/japan7}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "${PYTHON_BIN}" == "python" && -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"
EPOCHS=""
CHECKPOINT=""
COMMAND="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --val-batch-size) VAL_BATCH_SIZE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAPAN7_COCO_ROOT="${DATA_ROOT}"

data_updates=(
  "train_dataloader.dataset.img_folder=${DATA_ROOT}/images/train"
  "train_dataloader.dataset.ann_file=${DATA_ROOT}/annotations/instances_train.json"
  "val_dataloader.dataset.img_folder=${DATA_ROOT}/images/val"
  "val_dataloader.dataset.ann_file=${DATA_ROOT}/annotations/instances_val.json"
  "train_dataloader.total_batch_size=${BATCH_SIZE}"
  "val_dataloader.total_batch_size=${VAL_BATCH_SIZE}"
  "train_dataloader.num_workers=${WORKERS}"
  "val_dataloader.num_workers=${WORKERS}"
)

download() {
  local url="https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth"
  mkdir -p "$(dirname "${WEIGHTS}")"
  if [[ ! -s "${WEIGHTS}" ]]; then
    curl --fail --location "${url}" --output "${WEIGHTS}.part"
    mv "${WEIGHTS}.part" "${WEIGHTS}"
  fi
  [[ "$(stat --format=%s "${WEIGHTS}")" -gt 1000000 ]]
  sha256sum "${WEIGHTS}"
  echo "weights=${WEIGHTS}"
}

preflight() {
  command -v nvidia-smi >/dev/null
  nvidia-smi
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/check_japan7_coco.py" --root "${DATA_ROOT}"
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/preflight_japan7.py" \
    --config "${CONFIG}" \
    --root "${DATA_ROOT}" \
    --checkpoint "${WEIGHTS}" \
    --device "${DEVICE}" \
    --workers "${WORKERS}"
  mkdir -p "${OUTPUT_ROOT}/preflight"
  git -C "${ROOT_DIR}" rev-parse HEAD > "${OUTPUT_ROOT}/preflight/git_commit.txt"
  nvidia-smi > "${OUTPUT_ROOT}/preflight/nvidia-smi.txt"
  "${PYTHON_BIN}" -m pip freeze > "${OUTPUT_ROOT}/preflight/pip-freeze.txt"
}

run_training() {
  local mode="$1"
  local run_name="$2"
  local output_dir="${OUTPUT_ROOT}/${run_name}"
  local init_args=("-t" "${WEIGHTS}")
  mkdir -p "${output_dir}"

  if [[ -n "${CHECKPOINT}" ]]; then
    init_args=("-r" "${CHECKPOINT}")
  fi

  local command=(
    "${PYTHON_BIN}" "${ROOT_DIR}/train.py"
    -c "${CONFIG}"
    "${init_args[@]}"
    -d "${DEVICE}"
    --use-amp
    --seed "${SEED}"
    --output-dir "${output_dir}"
    -u "epochs=${EPOCHS}" "${data_updates[@]}"
  )
  git -C "${ROOT_DIR}" rev-parse HEAD > "${output_dir}/git_commit.txt"
  printf '%q ' "${command[@]}" > "${output_dir}/command.txt"
  printf '\n' >> "${output_dir}/command.txt"
  nvidia-smi > "${output_dir}/nvidia-smi-before.txt"
  "${PYTHON_BIN}" -m pip freeze > "${output_dir}/pip-freeze.txt"
  "${command[@]}" 2>&1 | tee "${output_dir}/${mode}.log"
  nvidia-smi > "${output_dir}/nvidia-smi-after.txt"
  echo "output=${output_dir}"
}

case "${COMMAND}" in
  download)
    download
    ;;
  preflight)
    [[ -s "${WEIGHTS}" ]] || { echo "missing weights: ${WEIGHTS}" >&2; exit 1; }
    preflight
    ;;
  smoke)
    [[ -s "${WEIGHTS}" ]] || { echo "missing weights: ${WEIGHTS}" >&2; exit 1; }
    preflight
    EPOCHS=1
    data_updates+=("train_dataloader.collate_fn.base_size_repeat=null")
    run_name="dfine_s_japan7_smoke_e1_img640_b${BATCH_SIZE}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
    run_training smoke "${run_name}"
    log="${OUTPUT_ROOT}/${run_name}/smoke.log"
    [[ -s "${OUTPUT_ROOT}/${run_name}/last.pth" ]]
    if grep -Eiq '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)|out of memory|CUDA error' "${log}"; then
      echo "smoke log contains a numerical, OOM, or CUDA error" >&2
      exit 1
    fi
    echo "SMOKE PASS"
    ;;
  test)
    [[ -n "${CHECKPOINT}" ]] || { echo "--checkpoint is required" >&2; exit 2; }
    output_dir="${OUTPUT_ROOT}/test_$(date +%Y%m%d_%H%M%S)"
    "${PYTHON_BIN}" "${ROOT_DIR}/train.py" \
      -c "${CONFIG}" -r "${CHECKPOINT}" -d "${DEVICE}" --test-only \
      --output-dir "${output_dir}" -u "${data_updates[@]}"
    ;;
  resume)
    [[ -n "${CHECKPOINT}" ]] || { echo "--checkpoint is required" >&2; exit 2; }
    EPOCHS=1
    run_training resume "resume_check_$(date +%Y%m%d_%H%M%S)"
    echo "RESUME LOAD PASS"
    ;;
  formal)
    [[ -n "${EPOCHS}" ]] || { echo "--epochs is required for formal training" >&2; exit 2; }
    run_training formal "dfine_s_japan7_formal_e${EPOCHS}_img640_b${BATCH_SIZE}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
    ;;
  *)
    echo "usage: $0 {download|preflight|smoke|test|resume|formal} [options]" >&2
    exit 2
    ;;
esac
