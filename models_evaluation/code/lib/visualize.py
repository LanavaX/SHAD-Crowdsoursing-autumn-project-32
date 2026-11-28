"""Contact sheets with GT/pred boxes drawn on images — saved as PNG, never read into context."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from PIL import Image


def _draw(ax, boxes, color, label, W, H):
    # boxes are normalized 0..1; image axes are in pixels — scale up or the
    # rectangles collapse into invisible dots in the corner.
    for b in boxes:
        ax.add_patch(Rectangle((b["left"] * W, b["top"] * H), b["width"] * W, b["height"] * H,
                               fill=False, edgecolor=color, linewidth=2.5))
    ax.plot([], [], color=color, label=label)


def contact_sheet(entries, out_path, cols=4, title=""):
    """entries: [(image_path, gt_boxes, pred_boxes, caption)] — saves one PNG sheet."""
    entries = entries[:16]  # ponytail: one sheet is enough for eyeballing
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = (len(entries) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax, (img_path, gt, pred, cap) in zip(axes.flat, entries):
        im = Image.open(img_path)  # keep open: imshow reads pixels lazily at draw time
        W, H = im.size
        ax.imshow(im)
        _draw(ax, gt, "lime", "GT", W, H)
        _draw(ax, pred, "red", "pred", W, H)
        ax.set_title(cap, fontsize=7)
        ax.axis("off")
    for ax in axes.flat[len(entries):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
