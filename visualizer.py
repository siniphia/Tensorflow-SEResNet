import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


# x, y : center coords
# w, h : width and height of the bounding box
# c : class
def plot_bounding_box(image, label, reg_pred, cls_lt_pred, cls_rt_pred, numbering, save_path, save_step, save=True):
    # groundtruth boxes
    gt_lx, gt_ly, gt_lw, gt_lh = _center_to_topleft(label[2], label[3], label[4], label[5])
    gt_rx, gt_ry, gt_rw, gt_rh = _center_to_topleft(label[6], label[7], label[8], label[9])

    # inferenced boxes
    lx, ly, lw, lh = _center_to_topleft(reg_pred[0], reg_pred[1], reg_pred[2], reg_pred[3])
    rx, ry, rw, rh = _center_to_topleft(reg_pred[4], reg_pred[5], reg_pred[6], reg_pred[7])

    # image & title
    fig, ax = plt.subplots(1)
    plt.title('RT_MAXIL : %s, LT_MAXIL : %s' % (str(label[1]), str(label[0])))
    ax.imshow(image)

    # draw boxes
    gt_ltbb = patches.Rectangle((gt_lx, gt_ly), gt_lw, gt_lh, linewidth=1, edgecolor='black', facecolor=None, fill=False)
    ax.add_patch(gt_ltbb)
    gt_rtbb = patches.Rectangle((gt_rx, gt_ry), gt_rw, gt_rh, linewidth=1, edgecolor='black', facecolor=None, fill=False)
    ax.add_patch(gt_rtbb)
    ltbb = patches.Rectangle((lx, ly), lw, lh, linewidth=1, edgecolor='r', facecolor=None, fill=False)
    ax.add_patch(ltbb)
    rtbb = patches.Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='r', facecolor=None, fill=False)
    ax.add_patch(rtbb)

    # add comments
    # ax.annotate('GT LT MAXIL', (gt_lx, gt_ly), color='black', weight='bold', fontsize=10)
    # ax.annotate('GT RT MAXIL', (gt_rx, gt_ry), color='black', weight='bold', fontsize=10)
    ax.annotate('LT MAXIL : ' + str(cls_lt_pred[0]), (lx, ly), color='black', weight='bold', fontsize=10)
    ax.annotate('RT MAXIL : ' + str(cls_rt_pred[0]), (rx, ry), color='black', weight='bold', fontsize=10)

    # correctness
    if cls_lt_pred[0] == label[0]:
        lt_correct = 'True'
    else:
        lt_correct = 'False'

    if cls_rt_pred[0] == label[1]:
        rt_correct = 'True'
    else:
        rt_correct = 'False'

    if not os.path.exists(os.path.join(save_path, save_step)):
        os.mkdir(os.path.join(save_path, save_step))

    if save:
        plt.savefig(os.path.join(save_path, save_step, str(numbering) + '_' + rt_correct + '_' + lt_correct + '.png'))

    plt.show()


def draw_bounding_box(img, lx, ly, lw, lh, rx, ry, rw, rh, x, y, xl, yl, c=1):
    # inferenced boxes
    lx, ly, lw, lh = _center_to_topleft(lx, ly, lw, lh)
    rx, ry, rw, rh = _center_to_topleft(rx, ry, rw, rh)

    fig, ax = plt.subplots(1)
    plt.title('LT_MAXIL : %s, RT_MAXIL : %s' % (str(xl), str(yl)))
    ax.imshow(img)
    ltbb = patches.Rectangle((lx, ly), lw, lh, linewidth=1, edgecolor='r', facecolor=None, fill=False)
    ax.add_patch(ltbb)
    rtbb = patches.Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='r', facecolor=None, fill=False)
    ax.add_patch(rtbb)
    ax.annotate('LT MAXIL : ' + str(x), (lx, ly), color='black', weight='bold', fontsize=10)
    ax.annotate('RT MAXIL : ' + str(y), (rx, ry), color='black', weight='bold', fontsize=10)
    plt.show()


# convert bb center coord to bottom-left coord
def _center_to_topleft(x, y, w, h):
    return x - int(w / 2), y - int(h / 2), w, h
