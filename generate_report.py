"""Generate outputs/report.md as a draft (HW3 spec §8 section structure).

Reads outputs/tables/*.csv and references outputs/figs/*.png to build the
full section scaffold (abstract, Step 1-3, conclusion, references,
contribution table). Sections that require hand-written prose are marked
with TODO in the output.

The contribution table is loaded from team.md if it exists; otherwise a
TODO placeholder is written. team.md format:
    | id | name | responsibility |
    |---|---|---|
    | 12345 | Alice | Step 1, README |

The output markdown stays in Traditional Chinese to match the HW report
deliverable; only this file's Python docstrings/comments are in English.
"""

from __future__ import annotations

import csv
from pathlib import Path

TABLES = Path("outputs/tables")
FIGS_REL = "figs"
OUT = Path("outputs/report.md")
TEAM = Path("team.md")

STR_COLS = {"condition", "method", "param", "subject", "idx", "attacker_ckpt"}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def md_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_(尚無資料 — 請先跑對應的 step runner)_\n"
    header = "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n"
    body = ""
    for r in rows:
        cells = []
        for c in cols:
            if c in STR_COLS:
                cells.append(str(r.get(c, "")))
            else:
                try:
                    cells.append(f"{float(r[c]):.4f}")
                except (ValueError, KeyError):
                    cells.append(str(r.get(c, "")))
        body += "| " + " | ".join(cells) + " |\n"
    return header + body


def aggregate_step1(rows: list[dict]) -> list[dict]:
    agg: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        key = (r["method"], r["param"])
        agg.setdefault(key, []).append(float(r["ssim"]))
    return [
        {"method": k[0], "param": k[1], "avg_ssim": sum(v) / len(v)}
        for k, v in sorted(agg.items())
    ]


def step3_retrain_with_baseline(step2_rows: list[dict], retrain_rows: list[dict]) -> list[dict]:
    """Merge NP-Pix(b=16) and NP-Blur(k=99) baselines from step2 into the step3 retrain table.

    Produces the full Fan Table 1 layout: NP-Pix b=16, DP-Pix x eps grid,
    NP-Blur k=99, DP-Blur x eps grid.
    """
    by_cond = {r["condition"]: r for r in step2_rows}
    out: list[dict] = []
    pix_baseline = by_cond.get("pixel_b16")
    blur_baseline = by_cond.get("blur_k99")

    if pix_baseline:
        out.append(
            {
                "method": "NP-Pix b=16",
                "eps": "-",
                "top1": pix_baseline["top1"],
                "top5": pix_baseline["top5"],
            }
        )
    for r in [r for r in retrain_rows if r["method"] == "dp_pix"]:
        out.append({"method": "DP-Pix b=16", "eps": r["eps"], "top1": r["top1"], "top5": r["top5"]})

    if blur_baseline:
        out.append(
            {
                "method": "NP-Blur k=99",
                "eps": "-",
                "top1": blur_baseline["top1"],
                "top5": blur_baseline["top5"],
            }
        )
    for r in [r for r in retrain_rows if r["method"] == "dp_blur"]:
        out.append({"method": "DP-Blur k=99", "eps": r["eps"], "top1": r["top1"], "top5": r["top5"]})
    return out


def load_team_table() -> str:
    """Read the contribution table from team.md, or return a TODO placeholder."""
    if TEAM.exists():
        return TEAM.read_text(encoding="utf-8").strip() + "\n"
    return (
        "| 學號 | 姓名 | 負責 |\n"
        "| --- | --- | --- |\n"
        "| TODO | TODO | TODO（請編輯根目錄的 team.md 後重跑 generate_report.py）|\n"
    )


def main() -> None:
    step1_rows = read_csv(TABLES / "step1_quality.csv")
    step2_rows = read_csv(TABLES / "step2_results.csv")
    step3_util = read_csv(TABLES / "step3_utility.csv")
    step3_retrain = read_csv(TABLES / "step3_retrain_attack.csv")
    step3_noret = read_csv(TABLES / "step3_no_retrain_attack.csv")
    step3_m = read_csv(TABLES / "step3_m_sweep.csv")
    # Optional ResNet18 results (only if --with-resnet was used)
    step2_resnet = read_csv(TABLES / "step2_results_resnet18.csv")
    step3_retrain_resnet = read_csv(TABLES / "step3_retrain_attack_resnet18.csv")
    step3_noret_resnet = read_csv(TABLES / "step3_no_retrain_attack_resnet18.csv")
    step1_avg = aggregate_step1(step1_rows)
    fan_table_1 = step3_retrain_with_baseline(step2_rows, step3_retrain)
    fan_table_1_resnet = step3_retrain_with_baseline(step2_resnet, step3_retrain_resnet) if step2_resnet else []

    # Pre-build ResNet section (f-strings forbid backslashes in expressions on Py 3.11)
    if step2_resnet:
        resnet_section = (
            "### Step 2 主結果（ResNet18）\n\n"
            + md_table(step2_resnet, ["condition", "top1", "top5", "baseline_top1", "baseline_top5"])
            + "\n### Fan Table 1（ResNet18 重訓）\n\n"
            + md_table(fan_table_1_resnet, ["method", "eps", "top1", "top5"])
            + "\n### no-retrain（ResNet18）\n\n"
            + md_table(step3_noret_resnet, ["method", "eps", "top1", "top5", "attacker_ckpt"])
            + "\nTODO（手寫）：比較 ResNet18 vs CNN_ATT 在同一 (method, ε) 上的 top1 差距。"
            + "若 ResNet18 在 NP 上略高但在 DP 上沒有顯著優勢 → 證明 DP 對更強攻擊者也有效。\n"
        )
    else:
        resnet_section = "_(尚未跑 ResNet18 對照；用 `python run_all.py --with-resnet` 啟動)_\n"

    md = f"""# HW3：Face De-Identification & Its Attacks and Defenses

## 0. 摘要

本作業實作並評估三項任務：(1) 用高斯模糊（NP-Blur）與像素化（NP-Pix）對 AT&T 人臉資料庫做去識別化；
(2) 訓練 CNN_ATT（McPherson 2016 Appendix A.3）對去識別化影像進行身份再辨識攻擊；
(3) 採用差分隱私 DP-Pix 與 DP-Blur（Fan 2019）防禦上述攻擊，並以重訓 (retrain) 與不重訓 (no-retrain) 兩種情境量化效益。

資料集為 AT&T (ORL)，共 40 個身份 × 10 張 92×112 灰階影像，依 McPherson §5.5 切 8/2 train/test。
所有實驗以 `seed=42` 確保可重現。

TODO（手寫）：補一段「主要發現」摘要，引用下面數字。預期論文落點：
McPherson Table 2 AT&T original ≈ 95% top-1；Fan Table 1 DP-Pix ε=0.1 ≈ 3.75%、DP-Blur ε=0.1 ≈ 1.25%。

### 組員分工

{load_team_table()}

---

## 1. Step 1：去識別化（NP-Pix / NP-Blur）

### 1.1 方法

- **NP-Pix（pixelization）**：將影像切成 b×b 區塊取平均後填回；以 `numpy.reshape + mean` 計算精確區塊平均（避免 `cv2.INTER_AREA` 對 factor>2 的整數縮放不等於 block average 的 quirk），最近鄰上採樣回原尺寸。
- **NP-Blur（Gaussian blur）**：`cv2.GaussianBlur(img, (k,k), sigmaX=0)`，σ 自動由 k 推算。

### 1.2 參數掃描

- Pixel block size `b ∈ {{2, 4, 8, 16}}`
- Gaussian kernel `k ∈ {{15, 45, 99}}`

### 1.3 視覺對比圖

![NP-Pix 對比](figs/step1_pixel_grid.png)

![NP-Blur 對比](figs/step1_blur_grid.png)

### 1.4 效用統計（平均 SSIM vs 原圖）

{md_table(step1_avg, ['method', 'param', 'avg_ssim'])}

### 1.5 觀察

TODO（手寫）：b 越大、k 越大 → 視覺品質越低，可從上方表格看出趨勢。
k=99 已大於影像高 112，輸出近乎均勻灰；b=16 把 92×112 縮成 ~6×7 區塊。

---

## 2. Step 2：CNN 攻擊（McPherson 2016）

### 2.1 威脅模型

攻擊者擁有與防禦者**同一種去識別化方法**處理過的 train set + 真實 label（身份 1-40），
目標是對同方法去識別化的 test set 預測身份。每個 (方法, 參數) 組合**獨立**訓一個攻擊模型（McPherson §5.5「no mixing」）。

### 2.2 CNN_ATT 架構（McPherson Appendix A.3）

```
Input (B, 1, 112, 92)                                         # AT&T 灰階 92×112
Conv(1→32,   3×3, pad=1) → LeakyReLU(0.01) → MaxPool(2×2)     → (B, 32, 56, 46)
Conv(32→64,  3×3, pad=1) → LeakyReLU(0.01) → MaxPool(2×2)     → (B, 64, 28, 23)
Conv(64→128, 3×3, pad=1) → LeakyReLU(0.01) → MaxPool(3×3)     → (B, 128, 9, 7)
Flatten (128 × 9 × 7 = 8064) → Linear(8064→1024) → LeakyReLU(0.01)
Dropout(0.5) → Linear(1024→40) → LogSoftmax
Loss: NLLLoss
```

### 2.3 訓練超參（McPherson §5.5）

| 超參 | 值 |
| --- | --- |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Weight decay | 5e-4 |
| LR decay | 1e-7 per step (LambdaLR `1/(1+1e-7·step)`) |
| Epochs | 100 |
| Batch size | 32 |
| Random seed | 42 |

### 2.4 結果表（McPherson Table 2 對應）

| Baseline | Top-1 | Top-5 |
| --- | --- | --- |
| Random (1/40) | 0.0250 | 0.1250 |

{md_table(step2_rows, ['condition', 'top1', 'top5', 'baseline_top1', 'baseline_top5'])}

### 2.5 訓練曲線

![Pixel 訓練曲線](figs/step2_curves_pixel.png)

![Blur 訓練曲線](figs/step2_curves_blur.png)

### 2.6 與 McPherson 對比討論

TODO（手寫）：
- 比對 `original` top-1 ≈ ? % 與 McPherson Table 2 AT&T original = 95.00%。
- 比對 `pixel_b16` top-1 與 Table 2 16×16 = 96.25%（同論文觀察：CNN 對 16×16 pixelization 幾乎無視）。
- 注意：本實作的 NP-Blur(k=99) 與 McPherson 用的 YouTube 自動模糊不同 — 後者在 AT&T 達 top-1 = 57.75%，
  我們的 k=99 結果可能不同（k=99 是 Fan 2019 採用的 Gaussian kernel）。

---

## 3. Step 3：DP 防禦（Fan 2019）

### 3.1 差分隱私概念

定義（Fan §3.2 ε-Image DP）：對任意兩張 m-Neighborhood 鄰居影像 I₁ 與 I₂（至多 m 個 pixel 不同）以及任何輸出 I_out：

$$\\Pr[\\mathcal{{A}}(I_1) = \\tilde{{I}}] \\le e^{{\\varepsilon}} \\cdot \\Pr[\\mathcal{{A}}(I_2) = \\tilde{{I}}]$$

ε 越小 → 隱私越強 → 雜訊越大 → 效用越低。

### 3.2 DP-Pix 演算法（Fan §3.3）

```
def dp_pix(I, b=16, m=16, eps=0.5):
    1. small = block_average(I, b)              # b×b 區塊平均
    2. scale = 255 · m / (b² · eps)             # Laplace scale（sensitivity）
    3. noisy = small + Laplace(0, scale)        # 加雜訊
    4. return upsample_nearest(noisy, b) clipped to [0,255]
```

預設參數 m=16（最多 16 個 pixel 可變）、b=16（區塊邊長）、eps=0.5。

### 3.3 DP-Blur 演算法（Fan §3.3）

```
def dp_blur(I, b0=4, m=16, eps=0.5, k=99):
    1. small = block_average(I, b0)             # 用小 cell b0=4
    2. noisy = small + Laplace(0, 255·m/(b0²·eps))
    3. up = upsample_nearest(noisy, b0)         # 回到 ORIGINAL size（92×112）
    4. return GaussianBlur(clip(up), k=99)
```

### 3.4 效用 vs 隱私參數（Fan Fig 3-4 對應）

ε 掃描範圍：`{{0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0}}`（log 軸）。

![MSE DP-Pix](figs/step3_mse_pix.png) ![SSIM DP-Pix](figs/step3_ssim_pix.png)

![MSE DP-Blur](figs/step3_mse_blur.png) ![SSIM DP-Blur](figs/step3_ssim_blur.png)

{md_table(step3_util, ['eps', 'mse_dp_pix', 'ssim_dp_pix', 'mse_dp_blur', 'ssim_dp_blur'])}

### 3.5 攻擊準確率（重訓，完整 ε 掃描，Fan Table 1 對應）

含 NP baseline 對照（NP-Pix b=16 / NP-Blur k=99 數字取自 Step 2）。
ε 掃描範圍從 Fan Table 1 的原始 3 點（0.1, 0.5, 1）擴充到完整 7 點（0.1, 0.3, 0.5, 0.7, 1, 3, 5）：

{md_table(fan_table_1, ['method', 'eps', 'top1', 'top5'])}

### 3.6 攻擊準確率（不重訓，bonus）

直接以 Step 2 NP-Pix(b=16) / NP-Blur(k=99) 攻擊模型評估 DP 化的 test set：

{md_table(step3_noret, ['method', 'eps', 'top1', 'top5', 'attacker_ckpt'])}

### 3.7 m-Neighborhood 掃描（ε=0.5 固定）

m 是 Fan §3.2 m-Neighborhood 參數（允許變動的 pixel 數）。sensitivity ∝ m，
所以 m 越大 → Laplace 雜訊越強 → 隱私越強但效用越差。

![MSE DP-Pix vs m](figs/step3_m_mse_pix.png) ![SSIM DP-Pix vs m](figs/step3_m_ssim_pix.png)

![MSE DP-Blur vs m](figs/step3_m_mse_blur.png) ![SSIM DP-Blur vs m](figs/step3_m_ssim_blur.png)

{md_table(step3_m, ['m', 'eps', 'mse_dp_pix', 'ssim_dp_pix', 'mse_dp_blur', 'ssim_dp_blur'])}

TODO（手寫）：補一段觀察 — m=64 vs m=8 的 SSIM 差距、與 Fan §3.2 m-Neighborhood 定義的關聯。

### 3.8 視覺對比

5 欄定性對比，預設參數 (ε=0.5, m=16, b=16, b₀=4, k=99)：

![Qualitative 5 欄對比](figs/step3_qualitative.png)

### 3.9 隱私-效用 tradeoff 討論

TODO（手寫）：
- 引用 §3.4 表格：ε 從 0.1 → 5 時 SSIM 上升、MSE 下降（DP 線逼近 NP baseline）。
- 引用 §3.5 表格：DP-Pix ε=0.1 預期 ≈ 3.75%（Fan）、ε=1 預期 ≈ 77.5%；DP-Blur ε=0.1 預期 ≈ 1.25%。
- 觀察：DP-Blur 在所有 ε 下攻擊準確率都更低（更安全），但視覺品質也較差（k=99 已是模糊）。
- 補充：no-retrain 場景準確率應 ≤ retrain（攻擊者沒看過 DP 化的 train set，無法適應雜訊）。

---

## 3a. 更強攻擊者對照（ResNet18，bonus）

CNN_ATT (~9M params, McPherson Appendix A.3) 是論文採用的「足夠強」攻擊者。
我們額外用 ResNet18 (~11M params, ImageNet 影像分類經典架構) 重做 Step 2 + Step 3 retrain
作為「更強攻擊者」對照，論點：**即使換更強模型，DP 在 ε ≤ 1 仍維持有效防禦**。

需注意：ResNet18 在 320 張訓練影像上**幾乎必然 overfit**（11M params vs 320 sample），
但這本身也是值得記錄的觀察 — 大模型 + 小資料 ≠ 更強攻擊。

{resnet_section}

---

## 4. 結論

TODO（手寫，套用實際數字）：
1. 傳統去識別化（NP-Pix、NP-Blur）對 CNN 攻擊幾乎無防禦：original top-1 ≈ 95%，b=16 仍 ≈ 96.25%。
2. 差分隱私 DP-Pix / DP-Blur 在 ε ≤ 1 可將重訓攻擊準確率打到接近隨機（2.5%）。
3. DP-Blur 比 DP-Pix 更安全（同 ε 下攻擊準確率更低），代價是視覺品質稍差。
4. 不重訓場景下 DP 防禦效果尤其顯著（攻擊者完全沒見過雜訊分布）。

---

## 5. References

1. R. McPherson, R. Shokri, V. Shmatikov. "Defeating Image Obfuscation with Deep Learning." arXiv:1609.00408, 2016.
2. L. Fan. "Differential Privacy for Image Publication." TPDP 2019.
3. L. Fan. "Image Pixelization with Differential Privacy." DBSec 2018, Springer LNCS 10980.
4. S. Hill, Z. Zhou, L. Saul, H. Shacham. "On the (In)effectiveness of Mosaicing and Blurring as Tools for Document Redaction." PETS 2016.
5. F. S. Samaria, A. C. Harter. "Parameterisation of a stochastic model for human face identification." 1994.（AT&T database）
6. Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli. "Image quality assessment: from error visibility to structural similarity." IEEE TIP 2004.（SSIM）

---

*本報告由 `generate_report.py` 自動產生 markdown 骨架；TODO 段落由組員手動補完。最後以 `pandoc outputs/report.md -o report/Report.docx` 轉換成繳交格式。*
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(md, encoding="utf-8")
    print(f"[report] wrote {OUT} ({len(md)} chars)")


if __name__ == "__main__":
    main()
