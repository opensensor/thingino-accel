/*
 * Mars YOLO Detection Test - Load YOLOv5 .mars model, run on real images
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mars.h"
#include "mars_runtime.h"
#include "nna.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"

#define YOLO_NUM_CLASSES  80
#define YOLO_CONF_THRESH  0.25f
#define YOLO_NMS_THRESH   0.45f

static const char* CLASS_NAMES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

typedef struct { float x, y, w, h, conf; int cls; } det_t;

/* Load image to INT8 with letterbox resize */
static int8_t* load_image(const char *path, int tw, int th, int nhwc, int *ow, int *oh) {
    int ch;
    unsigned char *img = stbi_load(path, ow, oh, &ch, 3);
    if (!img) { fprintf(stderr, "Failed to load: %s\n", path); return NULL; }
    printf("Loaded: %dx%d\n", *ow, *oh);

    float scale = fminf((float)tw / *ow, (float)th / *oh);
    int nw = (int)(*ow * scale), nh = (int)(*oh * scale);
    int px = (tw - nw) / 2, py = (th - nh) / 2;
    printf("Letterbox: %dx%d scale=%.3f pad=%d,%d\n", nw, nh, scale, px, py);

    unsigned char *rsz = malloc(nw * nh * 3);
    stbir_resize_uint8(img, *ow, *oh, 0, rsz, nw, nh, 0, 3);
    stbi_image_free(img);

    int8_t *out = malloc(tw * th * 3);
    memset(out, -17, tw * th * 3);  /* Gray letterbox */

    for (int y = 0; y < nh; y++) {
        for (int x = 0; x < nw; x++) {
            int si = (y * nw + x) * 3;
            int dy = y + py, dx = x + px;
            if (nhwc) {
                int di = (dy * tw + dx) * 3;
                out[di+0] = (int8_t)(rsz[si+0] - 128);
                out[di+1] = (int8_t)(rsz[si+1] - 128);
                out[di+2] = (int8_t)(rsz[si+2] - 128);
            } else {
                int ps = tw * th;
                out[0*ps + dy*tw + dx] = (int8_t)(rsz[si+0] - 128);
                out[1*ps + dy*tw + dx] = (int8_t)(rsz[si+1] - 128);
                out[2*ps + dy*tw + dx] = (int8_t)(rsz[si+2] - 128);
            }
        }
    }
    free(rsz);
    return out;
}

/* Parse YOLO [1, 25200, 85] output */
static int parse_output(const int8_t *data, int npred, float scale, det_t *dets, int maxd) {
    int cnt = 0;
    for (int i = 0; i < npred && cnt < maxd; i++) {
        const int8_t *p = data + i * 85;
        float obj = 1.0f / (1.0f + expf(-(float)p[4] * scale));
        if (obj < YOLO_CONF_THRESH) continue;

        int best_c = 0; float best_s = -1e9f;
        for (int c = 0; c < 80; c++) {
            float s = (float)p[5+c] * scale;
            if (s > best_s) { best_s = s; best_c = c; }
        }
        float conf = obj / (1.0f + expf(-best_s));
        if (conf < YOLO_CONF_THRESH) continue;

        dets[cnt].x = (float)p[0] * scale;
        dets[cnt].y = (float)p[1] * scale;
        dets[cnt].w = (float)p[2] * scale;
        dets[cnt].h = (float)p[3] * scale;
        dets[cnt].conf = conf;
        dets[cnt].cls = best_c;
        cnt++;
    }
    return cnt;
}

/* Simple NMS */
static int nms(det_t *d, int n, float thresh) {
    for (int i = 0; i < n-1; i++)
        for (int j = i+1; j < n; j++)
            if (d[j].conf > d[i].conf) { det_t t = d[i]; d[i] = d[j]; d[j] = t; }

    int *sup = calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        if (sup[i]) continue;
        for (int j = i+1; j < n; j++) {
            if (sup[j] || d[i].cls != d[j].cls) continue;
            float x1 = fmaxf(d[i].x - d[i].w/2, d[j].x - d[j].w/2);
            float y1 = fmaxf(d[i].y - d[i].h/2, d[j].y - d[j].h/2);
            float x2 = fminf(d[i].x + d[i].w/2, d[j].x + d[j].w/2);
            float y2 = fminf(d[i].y + d[i].h/2, d[j].y + d[j].h/2);
            float inter = fmaxf(0,x2-x1) * fmaxf(0,y2-y1);
            float iou = inter / (d[i].w*d[i].h + d[j].w*d[j].h - inter + 1e-6f);
            if (iou > thresh) sup[j] = 1;
        }
    }
    int out = 0;
    for (int i = 0; i < n; i++) if (!sup[i]) d[out++] = d[i];
    free(sup);
    return out;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.mars> [image.jpg]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *image_path = (argc > 2) ? argv[2] : NULL;

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Mars YOLO Detection Test                                ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    printf("[1] Initializing NNA...\n");
    if (nna_init() != NNA_SUCCESS) { fprintf(stderr, "NNA init failed\n"); return 1; }

    printf("[2] Loading model: %s\n", model_path);
    mars_model_t *model = NULL;
    if (mars_load_file(model_path, &model) != MARS_OK) {
        fprintf(stderr, "Failed to load model\n"); nna_deinit(); return 1;
    }
    mars_print_summary(model);

    mars_runtime_tensor_t *input = mars_get_input(model, 0);
    if (!input) { fprintf(stderr, "No input tensor\n"); mars_free(model); nna_deinit(); return 1; }

    int nhwc = (input->desc.format == MARS_FORMAT_NHWC);
    int in_h = nhwc ? input->desc.shape[1] : input->desc.shape[2];
    int in_w = nhwc ? input->desc.shape[2] : input->desc.shape[3];
    printf("Input: %dx%d format=%s\n", in_w, in_h, nhwc ? "NHWC" : "NCHW");

    int ow = in_w, oh = in_h;
    if (image_path) {
        printf("[3] Loading image: %s\n", image_path);
        int8_t *img = load_image(image_path, in_w, in_h, nhwc, &ow, &oh);
        if (!img) { mars_free(model); nna_deinit(); return 1; }
        memcpy(input->vaddr, img, in_w * in_h * 3);
        free(img);
    } else {
        printf("[3] Using test pattern\n");
        memset(input->vaddr, 0, input->alloc_size);
    }

    printf("[4] Running inference...\n");
    if (mars_run(model) != MARS_OK) {
        fprintf(stderr, "Inference failed\n"); mars_free(model); nna_deinit(); return 1;
    }
    printf("    Done!\n");

    printf("[5] Parsing detections...\n");
    mars_runtime_tensor_t *output = mars_get_output(model, 0);
    if (output) {
        printf("    Output: [%d, %d, %d] scale=%.6f\n",
               output->desc.shape[0], output->desc.shape[1], output->desc.shape[2],
               output->desc.scale);

        det_t dets[1000];
        int nd = parse_output((int8_t*)output->vaddr, output->desc.shape[1],
                              output->desc.scale, dets, 1000);
        printf("    Raw detections: %d\n", nd);
        nd = nms(dets, nd, YOLO_NMS_THRESH);

        printf("\n╔══════════════════════════════════════════════════════════╗\n");
        printf("║  Detection Results                                       ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        if (nd > 0) {
            printf("Found %d detections:\n\n", nd);
            for (int i = 0; i < nd && i < 20; i++) {
                const char *name = (dets[i].cls < YOLO_NUM_CLASSES) ? CLASS_NAMES[dets[i].cls] : "?";
                printf("  [%2d] %-12s %5.1f%%  @ (%.0f,%.0f) %.0fx%.0f\n",
                       i+1, name, dets[i].conf*100, dets[i].x, dets[i].y, dets[i].w, dets[i].h);
            }
        } else {
            printf("No detections above threshold %.2f\n", YOLO_CONF_THRESH);
        }
    }

    printf("\n[6] Cleanup...\n");
    mars_free(model);
    nna_deinit();
    printf("    Done!\n\n");
    return 0;
}
