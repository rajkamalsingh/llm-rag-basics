from __future__ import annotations
"""
Minimal Starter: Scale-Space Blob Detection (LoG / DoG vs Downsample)
-------------------------------------------------
Implements the five TODOs from the minimal starter:
 - build_scale_space
 - nms3d_and_threshold
 - (inside nms) across-scale peak check
 - peaks_to_circles

Run examples:
  python blob_starter.py --input asset
  python blob_starter.py --input asset/coins.png --method downsample --levels 12 --k 1.2
"""
import time
from sklearn.datasets import make_blobs
#from __future__ import annotations
import argparse, os, glob
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy import ndimage as ndi
from skimage.transform import resize


# -------------------------
# tiny I/O helpers (provided)
# -------------------------

def load_grayscale_float(path: str) -> np.ndarray:
    """Return grayscale float32 in [0,1]. Keep it simple on purpose."""
    import imageio.v3 as iio
    img = iio.imread(path)
    img = img.astype(np.float32)
    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[..., :3]
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        img = 0.2989*r + 0.5870*g + 0.1140*b
    vmax = img.max()
    if vmax > 1.0:
        img = img / (255.0 if vmax <= 255.0 else vmax)
    return np.clip(img, 0.0, 1.0).astype(np.float32)

def save_overlay(path_out: str, image: np.ndarray, circles: List[Tuple[float,float,float]]) -> None:
    """Draw (x,y,r) circles over image and save."""
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    for (x, y, r) in circles:
        ax.add_patch(Circle((x, y), r, edgecolor="lime", facecolor="none", linewidth=1.5))
    ax.set_title(f"{len(circles)} detections"); ax.axis("off")
    fig.tight_layout(); fig.savefig(path_out, dpi=150); plt.close(fig)


# -------------------------
# YOUR WORK STARTS HERE
# -------------------------

def build_scale_space(
    I: np.ndarray, sigma0: float, k: float, levels: int, method: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a LoG scale-space S with shape (levels, H, W) and an array of sigmas (levels,).

    method == 'LoG' or 'DoG':
        Keep image size fixed. For level i:
           sigma_i = sigma0 * k**i
           LoG:   L = gaussian_laplace(I, sigma_i);   R = (sigma_i^2) * L;   S[i] = (R)^2
           DoG:   R = G(I, c*sigma_i) - G(I, sigma_i)  (e.g., c = sqrt(2));  S[i] = (R)^2
           (Pick ONE of LoG or DoG to implement.)

    method == 'downsample':
        For level i:
           - Run LoG with fixed sigma0 on a downsampled image (by 1/k at each level)
           - Upsample response back to (H, W) and store to S[i]
           - Effective sigma is sigma0 * k**i  (in original coordinates)
    """
    H, W = I.shape
    S = np.empty((levels, H, W), dtype=np.float32)
    sigmas = np.empty((levels,), dtype=np.float32)

    if method == "LoG":
        for i in range(levels):
            # TODO-1: Generate scale space using Laplacian-of-Gaussian
            # This should match the behavior of the "downsample" alternative implementation below,
            sigma_i = sigma0 * (k ** i)
            L = ndi.gaussian_laplace(I, sigma=float(sigma_i))
            R = (sigma_i ** 2) * L
            S[i] = (R * R).astype(np.float32)
            sigmas[i] = float(sigma_i)
    elif method == "DoG":
        c = np.sqrt(2)
        for i in range(levels):
            # TODO-1: Generate scale space using DoG
            # This should match the behavior of the "downsample" implementation below,
            sigma_i = sigma0 * (k ** i)
            G1 = ndi.gaussian_filter(I, sigma=sigma_i)
            G2 = ndi.gaussian_filter(I, sigma=c * sigma_i)
            R = G2 - G1
            S[i] = (R * R).astype(np.float32)
            sigmas[i] = float(sigma_i)
    elif method == "downsample":
        I_curr = I.copy()
        for i in range(levels):
            # LoG with fixed sigma0 on the current (downsampled) image
            L = ndi.gaussian_laplace(I_curr, sigma=float(sigma0))
            R = (sigma0 * sigma0) * L
            R2 = (R * R).astype(np.float32)

            # Upsample to base size so S aligns across levels
            R_up = resize(R2, (H, W), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
            S[i] = R_up

            # Effective sigma mapped to original coordinates
            sigmas[i] = float(sigma0 * (k ** i))

            # Prepare next level image (downsample by 1/k)
            newH = max(1, int(round(I_curr.shape[0] / k)))
            newW = max(1, int(round(I_curr.shape[1] / k)))
            if newH == I_curr.shape[0] and newW == I_curr.shape[1]:
                # too small to shrink further: repeat last response if needed
                for j in range(i+1, levels):
                    S[j] = S[i]
                    sigmas[j] = sigmas[i] * (k ** (j - i))
                break
            I_curr = resize(I_curr, (newH, newW), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)

    else:
        raise ValueError("method must be 'LoG', 'DoG' or 'downsample'")

    return S, sigmas


def nms3d_and_threshold(
    S: np.ndarray, thresh_fraction: float
) -> List[Tuple[int,int,int,float]]:
    """
    3D non-maximum suppression + threshold.

    Keep a voxel if:
      - It is a 2D local maximum within a 3x3 neighborhood in its level,
      - It is >= its neighbors at level-1 and level+1 at the same (y,x) (edges replicated),
      - It is >= (thresh_fraction * S.max()).

    Return a list of peaks: (level, y, x, score), sorted by score descending.
    """
    if S.size == 0:
        return []
    L, H, W = S.shape

    # 2D local maxima per level
    S_xymax = np.empty_like(S)
    for i in range(L):
        # TODO-2: complete this loop to get S_xymax[i]
        local_max = ndi.maximum_filter(S[i], size=3, mode='nearest')
        S_xymax[i] = local_max

    # Neighbor comparisons across scale (replicate ends)
    S_prev = np.vstack([S[0:1], S[:-1]])    # (L,H,W)
    S_next = np.vstack([S[1:],  S[-1:]])    # (L,H,W)

    # TODO-3: detect peaks
    is_peak = (S == S_xymax) & (S >= S_prev) & (S >= S_next)

    # Global threshold
    thr = float(thresh_fraction * S.max())
    is_peak &= (S >= thr)

    ls, ys, xs = np.where(is_peak)
    if ls.size == 0:
        return []
    scores = S[ls, ys, xs]
    order = np.argsort(scores)[::-1]  # sort by score descending

    peaks = [(int(ls[i]), int(ys[i]), int(xs[i]), float(scores[i])) for i in order]
    return peaks


def peaks_to_circles(
    peaks: List[Tuple[int,int,int,float]], sigmas: np.ndarray
) -> List[Tuple[float,float,float]]:
    """
    Convert (level, y, x, score) to circles (x, y, r) where r = sqrt(2) * sigma[level].
    """
    circles = []
    for (level, y, x, _score) in peaks:
        sigma = float(sigmas[level])
        # TODO-4: compute r
        r = np.sqrt(2) * sigma
        circles.append((float(x), float(y), float(r)))
    return circles

# TODO-5: Implement edge filtering using Hessian matrix analysis
def edge_filter_hessian(I: np.ndarray, thresh: float = 10.0) -> np.ndarray:
    """
    Remove edge-like responses using Hessian ratio test.
    Returns mask of valid (non-edge) pixels.
    """
    Ixx = ndi.sobel(ndi.sobel(I, axis=0), axis=0)
    Iyy = ndi.sobel(ndi.sobel(I, axis=1), axis=1)
    Ixy = ndi.sobel(ndi.sobel(I, axis=0), axis=1)

    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy**2

    # Avoid division issues
    eps = 1e-6
    ratio = (trace ** 2) / (det + eps)

    return ratio < thresh

# ==== VISUALIZATION ====

def plot_scale_space(S, sigmas):
    for i in range(0, len(sigmas), max(1, len(sigmas)//4)):
        plt.imshow(S[i], cmap='hot')
        plt.title(f"Scale {i}, sigma={sigmas[i]:.2f}")
        plt.colorbar()
        plt.show()


def plot_scale_response(S, sigmas, points):
    for (x,y) in points:
        vals = [S[i,y,x] for i in range(len(sigmas))]
        plt.plot(sigmas, vals, label=f"{x,y}")
    plt.xlabel("Sigma")
    plt.ylabel("Response")
    plt.legend()
    plt.show()

# ==== SYNTHETIC DATA ====

def generate_blob_image(n_blobs=10, size=256, std=5):
    centers, _ = make_blobs(n_samples=n_blobs, centers=n_blobs,
                           cluster_std=std)
    img = np.zeros((size,size))

    for (x,y) in centers:
        img[int(y)%size, int(x)%size] = 1

    img = ndi.gaussian_filter(img, sigma=std)
    return img, centers


# ==== EVALUATION ====

def match_detections(detected, gt, eps=10):
    matched = 0
    for (gx,gy) in gt:
        for (dx,dy,_) in detected:
            if np.hypot(gx-dx, gy-dy) < eps:
                matched += 1
                break

    precision = matched / max(len(detected),1)
    recall = matched / max(len(gt),1)
    f1 = 2*precision*recall/(precision+recall+1e-6)

    return precision, recall, f1


# ==== EXPERIMENT RUNNER ====

def run_experiments():
    results = []

    for n in [5,10,20]:
        for std in [3,5,8]:

            img, gt = generate_blob_image(n, std=std)

            for method in ["LoG", "downsample"]:
                start = time.time()

                S, sigmas = build_scale_space(img, 2.0, 1.2, 10, method)
                peaks = nms3d_and_threshold(S, 0.03)

                # Apply Hessian filter
                mask = edge_filter_hessian(img)
                peaks = [(l,y,x,s) for (l,y,x,s) in peaks if mask[y,x]]

                circles = peaks_to_circles(peaks, sigmas)

                runtime = time.time() - start

                p,r,f1 = match_detections(circles, gt)

                results.append([method,n,std,p,r,f1,runtime])

    return results
# ==== PR CURVE ====

def plot_pr_curve(img, gt):
    thresholds = np.linspace(0.01, 0.1, 10)
    precisions, recalls = [], []

    S, sigmas = build_scale_space(img, 2.0, 1.2, 10, "LoG")

    for t in thresholds:
        peaks = nms3d_and_threshold(S, t)
        circles = peaks_to_circles(peaks, sigmas)

        p,r,f1 = match_detections(circles, gt)
        precisions.append(p)
        recalls.append(r)

    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.show()
# -------------------------
# CLI / runner (provided)
# -------------------------

def process_one(
    in_path: str, out_dir: str, method: str, sigma0: float, k: float, levels: int, thresh: float
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    I = load_grayscale_float(in_path)
    S, sigmas = build_scale_space(I, sigma0=sigma0, k=k, levels=levels, method=method)
    plot_scale_space(S, sigmas)
    peaks = nms3d_and_threshold(S, thresh_fraction=thresh)
    mask = edge_filter_hessian(I)
    peaks = [(l, y, x, s) for (l, y, x, s) in peaks if mask[y, x]]
    circles = peaks_to_circles(peaks, sigmas)
    #plot_pr_curve(I, peaks)
    fname = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(out_dir, f"{fname}_overlay_{method}.png")
    save_overlay(out_path, I, circles)
    plot_scale_response(S, sigmas,[(124,128), (255,248), (414,214),(372,43),(28,274)])
    results = run_experiments()
    print(results)
    img, gt = generate_blob_image(10)
    plot_pr_curve(img,gt)
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Minimal Blob Detector (starter)")
    ap.add_argument("--input", type=str, default="asset",
                    help="Image file or folder (default: asset)")
    ap.add_argument("--output", type=str, default="faq-finetune-chatbot/src/results",
                    help="Output folder for overlays")
    ap.add_argument("--method", type=str, default="downsample", choices=["downsample","LoG", "DoG"])
    ap.add_argument("--sigma0", type=float, default=2.0)
    ap.add_argument("--k", type=float, default=1.2)
    ap.add_argument("--levels", type=int, default=12)
    ap.add_argument("--thresh", type=float, default=0.03)
    args = ap.parse_args()

    # resolve inputs
    if os.path.isdir(args.input):
        imgs = []
        for ext in ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp","*.webp"):
            imgs += glob.glob(os.path.join(args.input, ext))
        imgs = sorted(imgs)
        if not imgs:
            print(f"No images found in: {args.input}"); return
    else:
        imgs = [args.input]

    for p in imgs:
        out = process_one(
            p, args.output, args.method, args.sigma0, args.k, args.levels, args.thresh
        )
        print(f"[OK] {p} -> {out}")

if __name__ == "__main__":
    main()
