/**
 * Mars YOLOv5 Detection Test
 *
 * Loads a JPEG/PNG/PPM image, runs YOLOv5 inference, and outputs detections
 */

#define _POSIX_C_SOURCE 200809L
#define _BSD_SOURCE
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <sys/stat.h>
#include "mars.h"
#include "mars_runtime.h"
#include "nna.h"
#include "nna_memory.h"

/* Daemon mode globals */
static volatile int g_running = 1;
static void signal_handler(int sig) { (void)sig; g_running = 0; }

/* stb_image for JPEG/PNG loading */
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM  /* We have our own PPM loader */
#include "stb_image.h"

/* stb_image_write for PNG/BMP output */
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* libjpeg-turbo for proper JPEG output */
#include <jpeglib.h>

/* External cache flush for MIPS */
extern void nna_cache_flush(void *ptr, size_t size);

// YOLO parameters (defaults, can be overridden by CLI args)
static float g_conf_threshold = 0.25f;  // Detection confidence threshold (default 25%)
static float g_nms_threshold = 0.45f;   // NMS IoU threshold
#define CONF_THRESHOLD g_conf_threshold
#define NMS_THRESHOLD g_nms_threshold
#define DEFAULT_NUM_CLASSES 80
#define DEFAULT_INPUT_SIZE 640
#define NUM_ANCHORS 3

// Class names for TinyDet (3 classes) - must match training order!
// Training uses: CLASS_NAMES = ['person', 'cat', 'dog']
static const char* tinydet_classes[] = {"person", "cat", "dog"};

// Class names for Security model (4 classes)
// Training uses: CLASS_NAMES = ['person', 'vehicle', 'cat', 'dog']
static const char* security_classes[] = {"person", "vehicle", "cat", "dog"};

// Class names for COCO (80 classes)
static const char* coco_classes[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

// YOLOv5n anchors (per scale)
static const float anchors[3][3][2] = {
    // 80x80 (stride 8)
    {{10, 13}, {16, 30}, {33, 23}},
    // 40x40 (stride 16)
    {{30, 61}, {62, 45}, {59, 119}},
    // 20x20 (stride 32)
    {{116, 90}, {156, 198}, {373, 326}}
};

static const int strides[3] = {8, 16, 32};
static const int grid_sizes[3] = {80, 40, 20};

typedef struct {
    float x1, y1, x2, y2;  // Bounding box
    float confidence;
    int class_id;
} Detection;

// Sigmoid function
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Compute IoU (Intersection over Union) between two boxes
static float compute_iou(const Detection* a, const Detection* b) {
    float x1 = fmaxf(a->x1, b->x1);
    float y1 = fmaxf(a->y1, b->y1);
    float x2 = fminf(a->x2, b->x2);
    float y2 = fminf(a->y2, b->y2);

    float inter_w = fmaxf(0.0f, x2 - x1);
    float inter_h = fmaxf(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;

    float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0) return 0.0f;
    return inter_area / union_area;
}

// Max detections we support (to avoid dynamic allocation)
#define MAX_DETECTIONS 100

// Non-Maximum Suppression - returns new count after suppression
static int apply_nms(Detection* detections, int count, float nms_threshold) {
    if (count <= 1) return count;
    if (count > MAX_DETECTIONS) count = MAX_DETECTIONS;

    // Sort by confidence (simple bubble sort for small arrays)
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (detections[j].confidence > detections[i].confidence) {
                Detection tmp = detections[i];
                detections[i] = detections[j];
                detections[j] = tmp;
            }
        }
    }

    // Mark suppressed detections
    static uint8_t suppressed[MAX_DETECTIONS];
    memset(suppressed, 0, count);

    // Use separate result array to avoid corruption during compaction
    static Detection result[MAX_DETECTIONS];
    int kept = 0;

    for (int i = 0; i < count; i++) {
        if (suppressed[i]) continue;

        // Keep this detection in result array
        result[kept] = detections[i];
        kept++;

        // Suppress overlapping detections of same class
        for (int j = i + 1; j < count; j++) {
            if (suppressed[j]) continue;
            if (detections[j].class_id != detections[i].class_id) continue;

            float iou = compute_iou(&detections[i], &detections[j]);
            if (iou > nms_threshold) {
                suppressed[j] = 1;
            }
        }
    }

    // Copy kept detections back to original array
    memcpy(detections, result, kept * sizeof(Detection));

    return kept;
}

// Load original image at full resolution (caller must free returned pointer)
static uint8_t* load_original_image(const char* path, int* out_w, int* out_h) {
    int channels;
    unsigned char* img = stbi_load(path, out_w, out_h, &channels, 3);
    return img;  // Returns NULL on failure, caller must free with stbi_image_free or free()
}

// Resize image using bilinear interpolation - matches PIL Image.BILINEAR exactly
// PIL uses half-pixel center convention: src = (dst + 0.5) * scale - 0.5
static void resize_image(const uint8_t* src, int src_w, int src_h,
                         uint8_t* dst, int dst_w, int dst_h) {
    float x_scale = (float)src_w / dst_w;
    float y_scale = (float)src_h / dst_h;

    for (int y = 0; y < dst_h; y++) {
        // PIL coordinate mapping: src_y = (y + 0.5) * scale - 0.5
        float src_yf = (y + 0.5f) * y_scale - 0.5f;
        if (src_yf < 0) src_yf = 0;
        if (src_yf > src_h - 1) src_yf = src_h - 1;

        int y0 = (int)src_yf;
        int y1 = y0 + 1;
        if (y1 >= src_h) y1 = src_h - 1;
        float y_frac = src_yf - y0;

        for (int x = 0; x < dst_w; x++) {
            // PIL coordinate mapping: src_x = (x + 0.5) * scale - 0.5
            float src_xf = (x + 0.5f) * x_scale - 0.5f;
            if (src_xf < 0) src_xf = 0;
            if (src_xf > src_w - 1) src_xf = src_w - 1;

            int x0 = (int)src_xf;
            int x1 = x0 + 1;
            if (x1 >= src_w) x1 = src_w - 1;
            float x_frac = src_xf - x0;

            // Get 4 neighboring pixels
            int idx00 = (y0 * src_w + x0) * 3;
            int idx01 = (y0 * src_w + x1) * 3;
            int idx10 = (y1 * src_w + x0) * 3;
            int idx11 = (y1 * src_w + x1) * 3;
            int dst_idx = (y * dst_w + x) * 3;

            // Bilinear interpolation for each channel
            for (int c = 0; c < 3; c++) {
                float top = src[idx00 + c] * (1 - x_frac) + src[idx01 + c] * x_frac;
                float bot = src[idx10 + c] * (1 - x_frac) + src[idx11 + c] * x_frac;
                float val = top * (1 - y_frac) + bot * y_frac;
                dst[dst_idx + c] = (uint8_t)(val + 0.5f);  // Round to nearest
            }
        }
    }
}

// Load image using stb_image (supports JPEG, PNG, BMP) or PPM fallback
static int load_test_image(const char* path, uint8_t* rgb_data, int target_w, int target_h) {
    int w, h, channels;

    // Try stb_image first (handles JPEG, PNG, BMP)
    printf("  Attempting to load: %s\n", path);
    unsigned char* img = stbi_load(path, &w, &h, &channels, 3);  // Force RGB

    if (img) {
        printf("Loaded image: %dx%d (%d channels -> RGB)\n", w, h, channels);

        if (w == target_w && h == target_h) {
            // Direct copy
            memcpy(rgb_data, img, w * h * 3);
        } else {
            // Resize using nearest-neighbor
            printf("Resizing from %dx%d to %dx%d\n", w, h, target_w, target_h);
            resize_image(img, w, h, rgb_data, target_w, target_h);
        }

        stbi_image_free(img);
        return 0;
    }

    // stb_image failed - try PPM fallback
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Cannot open image: %s\n", path);
        printf("stb_image error: %s\n", stbi_failure_reason());
        return -1;
    }

    // Check if it's a PPM file
    char header[3];
    if (fread(header, 1, 2, f) != 2) {
        fclose(f);
        return -1;
    }
    header[2] = '\0';

    if (strcmp(header, "P6") == 0) {
        // PPM format
        int ppm_w = 0, ppm_h = 0, maxval = 0;
        int c;

        // Skip whitespace after P6
        while ((c = fgetc(f)) != EOF && (c == ' ' || c == '\t' || c == '\n' || c == '\r'));

        // Skip comments
        while (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n');
            c = fgetc(f);
        }

        // Read width
        ungetc(c, f);
        if (fscanf(f, "%d", &ppm_w) != 1) { fclose(f); return -1; }

        // Skip whitespace
        while ((c = fgetc(f)) != EOF && (c == ' ' || c == '\t' || c == '\n' || c == '\r'));

        // Skip comments
        while (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n');
            c = fgetc(f);
        }

        // Read height
        ungetc(c, f);
        if (fscanf(f, "%d", &ppm_h) != 1) { fclose(f); return -1; }

        // Skip whitespace
        while ((c = fgetc(f)) != EOF && (c == ' ' || c == '\t' || c == '\n' || c == '\r'));

        // Skip comments
        while (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n');
            c = fgetc(f);
        }

        // Read maxval
        ungetc(c, f);
        if (fscanf(f, "%d", &maxval) != 1) { fclose(f); return -1; }

        // Skip single whitespace character before binary data
        fgetc(f);

        printf("Loading PPM image: %dx%d (maxval=%d)\n", ppm_w, ppm_h, maxval);

        if (ppm_w == target_w && ppm_h == target_h) {
            size_t bytes_read = fread(rgb_data, 1, ppm_w * ppm_h * 3, f);
            fclose(f);
            return (bytes_read == (size_t)(ppm_w * ppm_h * 3)) ? 0 : -1;
        }

        // Resize line-by-line
        printf("Resizing from %dx%d to %dx%d\n", ppm_w, ppm_h, target_w, target_h);
        long data_start = ftell(f);
        uint8_t src_line[1920 * 3];
        if (ppm_w > 1920) {
            printf("Image too wide: %d > 1920\n", ppm_w);
            fclose(f);
            return -1;
        }

        int last_src_y = -1;
        for (int y = 0; y < target_h; y++) {
            int src_y = y * ppm_h / target_h;
            if (src_y != last_src_y) {
                fseek(f, data_start + (long)src_y * ppm_w * 3, SEEK_SET);
                if (fread(src_line, 1, ppm_w * 3, f) != (size_t)(ppm_w * 3)) {
                    fclose(f);
                    return -1;
                }
                last_src_y = src_y;
            }
            for (int x = 0; x < target_w; x++) {
                int src_x = x * ppm_w / target_w;
                int src_idx = src_x * 3;
                int dst_idx = (y * target_w + x) * 3;
                rgb_data[dst_idx + 0] = src_line[src_idx + 0];
                rgb_data[dst_idx + 1] = src_line[src_idx + 1];
                rgb_data[dst_idx + 2] = src_line[src_idx + 2];
            }
        }
        fclose(f);
        return 0;
    }

    fclose(f);
    printf("Unsupported image format: %s\n", path);
    return -1;
}

// Preprocess RGB image to INT8 NHWC format
// Uses symmetric quantization: int8 = float / scale (zero_point = 0)
static void preprocess_image_int8(const uint8_t* rgb, int8_t* output, int w, int h, float scale) {
    for (int i = 0; i < w * h * 3; i++) {
        float val = rgb[i] / 255.0f;  // Normalize to [0, 1]
        int32_t quantized = (int32_t)(val / scale);
        // Clamp to int8 range
        if (quantized > 127) quantized = 127;
        if (quantized < -128) quantized = -128;
        output[i] = (int8_t)quantized;
    }
}

// ImageNet normalization constants (must match training)
static const float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};  // RGB
static const float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};   // RGB

// Preprocess RGB image to Float32 NHWC format
// Normalizes with ImageNet mean/std (same as training)
static void preprocess_image_float32_nhwc(const uint8_t* rgb, float* output, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 3;
            for (int c = 0; c < 3; c++) {
                float val = rgb[idx + c] / 255.0f;  // Normalize to [0, 1]
                output[idx + c] = (val - IMAGENET_MEAN[c]) / IMAGENET_STD[c];  // ImageNet normalize
            }
        }
    }
}

// Preprocess RGB image to Float32 NCHW format
// Normalizes with ImageNet mean/std, rearranges from HWC to CHW
static void preprocess_image_float32_nchw(const uint8_t* rgb, float* output, int w, int h) {
    int plane_size = w * h;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int src_idx = (y * w + x) * 3;  // HWC index
            int dst_idx = y * w + x;        // Spatial index within plane
            for (int c = 0; c < 3; c++) {
                float val = rgb[src_idx + c] / 255.0f;  // Normalize to [0, 1]
                output[c * plane_size + dst_idx] = (val - IMAGENET_MEAN[c]) / IMAGENET_STD[c];  // ImageNet normalize
            }
        }
    }
}

// Preprocess RGB image to INT8 NCHW format
static void preprocess_image_int8_nchw(const uint8_t* rgb, int8_t* output, int w, int h, float scale) {
    int plane_size = w * h;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int src_idx = (y * w + x) * 3;  // HWC index
            int dst_idx = y * w + x;        // Spatial index within plane
            for (int c = 0; c < 3; c++) {
                float val = rgb[src_idx + c] / 255.0f;  // Normalize to [0, 1]
                int32_t quantized = (int32_t)(val / scale);
                if (quantized > 127) quantized = 127;
                if (quantized < -128) quantized = -128;
                output[c * plane_size + dst_idx] = (int8_t)quantized;
            }
        }
    }
}

// Decode a single YOLOv5 detection head
// Input: NHWC format [1, H, W, 255] where 255 = 3 anchors * 85 values
// 85 = 5 (x, y, w, h, obj_conf) + 80 (class scores)
static int decode_head(const int8_t* data, float scale, int scale_idx,
                       Detection* detections, int max_detections, int current_count) {
    int num_dets = current_count;
    int grid_h = grid_sizes[scale_idx];
    int grid_w = grid_sizes[scale_idx];
    int stride = strides[scale_idx];

    // Debug first position
    printf("\n[Scale %d] Grid=%dx%d, stride=%d, scale=%.6f\n",
           scale_idx, grid_h, grid_w, stride, scale);

    // Sample key positions - ONNX reference found high objectness at these positions:
    // Scale 2 (20x20): [10,13], [10,14], [10,15], [15,8] - class 0 (person)
    int check_positions[4][2];
    if (scale_idx == 0) {  // 80x80, stride 8
        check_positions[0][0] = 39; check_positions[0][1] = 64;  // ONNX max obj
        check_positions[1][0] = 40; check_positions[1][1] = 40;  // Center
        check_positions[2][0] = 40; check_positions[2][1] = 52;  // ~[10,13]*4
        check_positions[3][0] = 0; check_positions[3][1] = 0;    // Corner
    } else if (scale_idx == 1) {  // 40x40, stride 16
        check_positions[0][0] = 29; check_positions[0][1] = 16;  // ONNX max obj
        check_positions[1][0] = 20; check_positions[1][1] = 26;  // ~[10,13]*2
        check_positions[2][0] = 20; check_positions[2][1] = 28;  // ~[10,14]*2
        check_positions[3][0] = 0; check_positions[3][1] = 0;    // Corner
    } else {  // 20x20, stride 32
        check_positions[0][0] = 15; check_positions[0][1] = 8;   // ONNX max obj (0.420) raw=-3
        check_positions[1][0] = 14; check_positions[1][1] = 8;   // ONNX raw=-18
        check_positions[2][0] = 10; check_positions[2][1] = 14;  // ONNX high obj (0.368)
        check_positions[3][0] = 0; check_positions[3][1] = 0;    // Corner
    }

    printf("  Checking key positions (y,x) - compare with ONNX reference:\n");
    for (int p = 0; p < 4; p++) {
        int y = check_positions[p][0];
        int x = check_positions[p][1];
        int offset = (y * grid_w + x) * 255;  // NHWC offset
        // Print raw int8 values for all 3 anchors' objectness
        printf("    pos[%d,%d]: obj_raw=[%d,%d,%d] -> conf=[%.3f,%.3f,%.3f]\n",
               y, x, data[offset+4], data[offset+89], data[offset+174],
               sigmoid(data[offset+4] * scale),
               sigmoid(data[offset+89] * scale),
               sigmoid(data[offset+174] * scale));
        // For scale 2, also print first 16 bytes and class scores
        if (scale_idx == 2 && p < 3) {
            printf("      -> first 16: [%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]\n",
                   data[offset+0], data[offset+1], data[offset+2], data[offset+3],
                   data[offset+4], data[offset+5], data[offset+6], data[offset+7],
                   data[offset+8], data[offset+9], data[offset+10], data[offset+11],
                   data[offset+12], data[offset+13], data[offset+14], data[offset+15]);
            int class0_raw = data[offset + 5];  // First class score for anchor 0
            float class0_conf = sigmoid(class0_raw * scale);
            float combined = sigmoid(data[offset+4] * scale) * class0_conf;
            printf("      -> class0_raw=%d, class0_conf=%.3f, combined=%.3f\n",
                   class0_raw, class0_conf, combined);
        }
    }

    // Find max objectness in this head for debugging
    float max_obj_conf = 0.0f;
    int max_obj_y = 0, max_obj_x = 0, max_obj_a = 0;
    int8_t max_obj_raw = 0;

    for (int y = 0; y < grid_h; y++) {
        for (int x = 0; x < grid_w; x++) {
            for (int a = 0; a < NUM_ANCHORS; a++) {
                // Offset in NHWC: (y * W + x) * 255 + a * 85
                int offset = (y * grid_w + x) * 255 + a * 85;
                const int8_t* box = data + offset;

                // Get raw values
                int8_t raw_bx = box[0];
                int8_t raw_by = box[1];
                int8_t raw_bw = box[2];
                int8_t raw_bh = box[3];
                int8_t raw_obj = box[4];

                // Dequantize box values
                float bx = raw_bx * scale;
                float by = raw_by * scale;
                float bw = raw_bw * scale;
                float bh = raw_bh * scale;
                float obj = raw_obj * scale;

                // Apply sigmoid to objectness
                float obj_conf = sigmoid(obj);

                // Track max objectness for debug
                if (obj_conf > max_obj_conf) {
                    max_obj_conf = obj_conf;
                    max_obj_y = y;
                    max_obj_x = x;
                    max_obj_a = a;
                    max_obj_raw = raw_obj;
                }

                if (obj_conf < CONF_THRESHOLD) continue;
                if (num_dets >= max_detections) continue;

                // Find best class from NHWC layout (80 classes for COCO/YOLOv5)
                int best_class = 0;
                float best_score = -1e9f;
                for (int c = 0; c < DEFAULT_NUM_CLASSES; c++) {
                    int8_t raw_score = box[5 + c];  // NHWC: classes at offset 5-84
                    float score = raw_score * scale;
                    if (score > best_score) {
                        best_score = score;
                        best_class = c;
                    }
                }

                // Filter: only person (0), cat (15), dog (16)
                if (best_class != 0 && best_class != 15 && best_class != 16) continue;

                float class_conf = sigmoid(best_score);
                float final_conf = obj_conf * class_conf;
                if (final_conf < CONF_THRESHOLD) continue;

                // Decode box coordinates
                // YOLOv5: xy = (sigmoid(xy) * 2 - 0.5 + grid) * stride
                //         wh = (sigmoid(wh) * 2)^2 * anchor
                float cx = (sigmoid(bx) * 2.0f - 0.5f + x) * stride;
                float cy = (sigmoid(by) * 2.0f - 0.5f + y) * stride;
                float w = powf(sigmoid(bw) * 2.0f, 2) * anchors[scale_idx][a][0];
                float h = powf(sigmoid(bh) * 2.0f, 2) * anchors[scale_idx][a][1];

                Detection* det = &detections[num_dets++];
                det->x1 = cx - w / 2;
                det->y1 = cy - h / 2;
                det->x2 = cx + w / 2;
                det->y2 = cy + h / 2;
                det->confidence = final_conf;
                det->class_id = best_class;
            }
        }
    }

    printf("  Max objectness: %.3f at pos[%d,%d,a%d] (raw=%d)\n",
           max_obj_conf, max_obj_y, max_obj_x, max_obj_a, max_obj_raw);

    return num_dets;
}

// Draw a horizontal line on RGB image
static void draw_hline(uint8_t* rgb, int w, int h, int x1, int x2, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (y < 0 || y >= h) return;
    if (x1 < 0) x1 = 0;
    if (x2 >= w) x2 = w - 1;
    for (int x = x1; x <= x2; x++) {
        int idx = (y * w + x) * 3;
        rgb[idx] = r;
        rgb[idx + 1] = g;
        rgb[idx + 2] = b;
    }
}

// Draw a vertical line on RGB image
static void draw_vline(uint8_t* rgb, int w, int h, int x, int y1, int y2, uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || x >= w) return;
    if (y1 < 0) y1 = 0;
    if (y2 >= h) y2 = h - 1;
    for (int y = y1; y <= y2; y++) {
        int idx = (y * w + x) * 3;
        rgb[idx] = r;
        rgb[idx + 1] = g;
        rgb[idx + 2] = b;
    }
}

// Draw a rectangle (bounding box) with thickness
static void draw_rect(uint8_t* rgb, int w, int h, int x1, int y1, int x2, int y2,
                      uint8_t r, uint8_t g, uint8_t b, int thickness) {
    for (int t = 0; t < thickness; t++) {
        draw_hline(rgb, w, h, x1, x2, y1 + t, r, g, b);  // Top
        draw_hline(rgb, w, h, x1, x2, y2 - t, r, g, b);  // Bottom
        draw_vline(rgb, w, h, x1 + t, y1, y2, r, g, b);  // Left
        draw_vline(rgb, w, h, x2 - t, y1, y2, r, g, b);  // Right
    }
}

// Color palette for different classes (RGB)
static const uint8_t class_colors[][3] = {
    {255, 0, 0},     // 0: person - red
    {0, 255, 0},     // 1: bicycle - green
    {0, 0, 255},     // 2: car - blue
    {255, 255, 0},   // 3: motorcycle - yellow
    {255, 0, 255},   // 4: airplane - magenta
    {0, 255, 255},   // 5: bus - cyan
    {255, 128, 0},   // 6: train - orange
    {128, 0, 255},   // 7: truck - purple
    {0, 255, 128},   // 8: boat - spring green
    {255, 128, 128}, // 9+: default - light red
};

// Draw detections on image
// Note: Detection coords are normalized [0,1], need to scale to image size
static void draw_detections(uint8_t* rgb, int w, int h, Detection* dets, int num_dets) {
    for (int i = 0; i < num_dets; i++) {
        Detection* d = &dets[i];
        // Scale normalized coords to pixel coords
        int x1 = (int)(d->x1 * w), y1 = (int)(d->y1 * h);
        int x2 = (int)(d->x2 * w), y2 = (int)(d->y2 * h);

        // Clamp to image bounds
        if (x1 < 0) x1 = 0; if (x1 >= w) x1 = w - 1;
        if (x2 < 0) x2 = 0; if (x2 >= w) x2 = w - 1;
        if (y1 < 0) y1 = 0; if (y1 >= h) y1 = h - 1;
        if (y2 < 0) y2 = 0; if (y2 >= h) y2 = h - 1;

        // Get color for this class
        int color_idx = d->class_id < 10 ? d->class_id : 9;
        uint8_t r = class_colors[color_idx][0];
        uint8_t g = class_colors[color_idx][1];
        uint8_t b = class_colors[color_idx][2];

        // Draw bounding box with thickness 3
        draw_rect(rgb, w, h, x1, y1, x2, y2, r, g, b, 3);
    }
}

// Get file extension (lowercase)
static const char* get_extension(const char* path) {
    const char* dot = strrchr(path, '.');
    if (!dot || dot == path) return "";
    return dot + 1;
}

// Save JPEG using libjpeg-turbo (produces proper, social-media-compatible JPEGs)
static int save_jpeg(const char* path, const uint8_t* rgb, int w, int h, int quality) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE* outfile;
    JSAMPROW row_pointer[1];

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    if ((outfile = fopen(path, "wb")) == NULL) {
        printf("Error: Cannot open %s for writing\n", path);
        return -1;
    }
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = w;
    cinfo.image_height = h;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    // Use optimized Huffman tables (like ImageMagick does) - this is KEY for compatibility
    cinfo.optimize_coding = TRUE;

    // Use 4:4:4 subsampling for maximum quality/compatibility
    cinfo.comp_info[0].h_samp_factor = 1;
    cinfo.comp_info[0].v_samp_factor = 1;
    cinfo.comp_info[1].h_samp_factor = 1;
    cinfo.comp_info[1].v_samp_factor = 1;
    cinfo.comp_info[2].h_samp_factor = 1;
    cinfo.comp_info[2].v_samp_factor = 1;

    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = (JSAMPROW)&rgb[cinfo.next_scanline * w * 3];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);

    return 0;
}

// Save image - auto-detect format from extension (jpg, png, bmp, or ppm)
static int save_image(const char* path, const uint8_t* rgb, int w, int h) {
    const char* ext = get_extension(path);
    int result = 0;

    if (strcasecmp(ext, "jpg") == 0 || strcasecmp(ext, "jpeg") == 0) {
        // JPEG with quality 90 using libjpeg-turbo
        result = save_jpeg(path, rgb, w, h, 90);
        if (result == 0) printf("Saved JPEG: %s\n", path);
        return result;
    } else if (strcasecmp(ext, "png") == 0) {
        result = stbi_write_png(path, w, h, 3, rgb, w * 3);
        if (result) printf("Saved PNG: %s\n", path);
    } else if (strcasecmp(ext, "bmp") == 0) {
        result = stbi_write_bmp(path, w, h, 3, rgb);
        if (result) printf("Saved BMP: %s\n", path);
    } else {
        // Default to PPM
        FILE* f = fopen(path, "wb");
        if (!f) {
            printf("Error: Cannot open %s for writing\n", path);
            return -1;
        }
        fprintf(f, "P6\n%d %d\n255\n", w, h);
        fwrite(rgb, 1, w * h * 3, f);
        fclose(f);
        printf("Saved PPM: %s\n", path);
        return 0;
    }

    if (!result) {
        printf("Error: Failed to save %s\n", path);
        return -1;
    }
    return 0;
}

// Decode TinyDet output (single head, anchor-free)
// Output format: [1, H, W, 5+num_classes] = [1, 12, 20, 8] for 320x192 input with 3 classes
// Channels: x_off, y_off, w, h, objectness, class0, class1, class2
// Box format: x_off/y_off are sigmoid offsets within cell, w/h are sigmoid-normalized to image
static int decode_tinydet_output(mars_model_t* model, Detection* detections, int max_detections, int input_w, int input_h) {
    int num_dets = 0;

    mars_runtime_tensor_t* output = mars_get_output(model, 0);
    if (!output || !output->vaddr) {
        printf("Error: Failed to get output tensor\n");
        return 0;
    }

    // Debug: print raw shape array
    printf("  Output shape raw: [%d, %d, %d, %d] ndims=%u format=%d\n",
           output->desc.shape[0], output->desc.shape[1],
           output->desc.shape[2], output->desc.shape[3],
           output->desc.ndims, output->desc.format);

    // Determine format from shape:
    // TinyDet output has channels = 5 + num_classes (8 for 3 classes, up to ~20 for more)
    // Grid size is typically 10, 20, 40, 80
    // NCHW: [1, C, H, W] - small C in position 1, larger H/W in positions 2,3
    // NHWC: [1, H, W, C] - small C in position 3, larger H/W in positions 1,2
    // Heuristic: channels dimension is the smallest non-batch dimension
    int s1 = output->desc.shape[1];
    int s2 = output->desc.shape[2];
    int s3 = output->desc.shape[3];
    int is_nchw = (s1 < s2 && s1 < s3);  // position 1 has smallest value = NCHW

    int grid_h, grid_w, channels;
    if (is_nchw) {
        // NCHW: [N, C, H, W]
        channels = s1;
        grid_h = s2;
        grid_w = s3;
        printf("  Detected NCHW format\n");
    } else {
        // NHWC: [N, H, W, C]
        grid_h = s1;
        grid_w = s2;
        channels = s3;
        printf("  Detected NHWC format\n");
    }

    int num_classes = channels - 5;  // channels = 5 + num_classes (box4 + obj1 + classes)
    float scale = output->desc.scale;
    int stride_h = input_h / grid_h;  // e.g., 192/12 = 16
    int stride_w = input_w / grid_w;  // e.g., 320/20 = 16
    int is_float32 = (output->desc.dtype == MARS_DTYPE_FLOAT32);

    printf("  TinyDet output: grid=%dx%d, channels=%d (%d classes), stride=%dx%d, scale=%.6f, dtype=%s\n",
           grid_w, grid_h, channels, num_classes, stride_w, stride_h, scale, is_float32 ? "float32" : "int8");

    const int8_t* data_i8 = (const int8_t*)output->vaddr;
    const float* data_f32 = (const float*)output->vaddr;
    int spatial_size = grid_h * grid_w;

    // Select class names based on number of classes
    const char** class_names;
    if (num_classes == 3) {
        class_names = tinydet_classes;
    } else if (num_classes == 4) {
        class_names = security_classes;
    } else {
        class_names = coco_classes;
    }

    // Debug: print a few raw values at [0,0] for each channel
    printf("  Sample values at [0,0]: ");
    for (int c = 0; c < channels && c < 8; c++) {
        int idx = is_nchw ? (c * spatial_size) : c;  // NCHW vs NHWC indexing
        if (is_float32) {
            printf("%.3f ", data_f32[idx]);
        } else {
            printf("%d ", data_i8[idx]);
        }
    }
    printf("\n");

    // Helper macros for NCHW and NHWC access
    #define GET_VAL_NHWC(y, x, c) (is_float32 ? data_f32[((y) * grid_w + (x)) * channels + (c)] : \
                                                (data_i8[((y) * grid_w + (x)) * channels + (c)] * scale))
    #define GET_VAL_NCHW(c, y, x) (is_float32 ? data_f32[(c) * spatial_size + (y) * grid_w + (x)] : \
                                                (data_i8[(c) * spatial_size + (y) * grid_w + (x)] * scale))
    #define GET_VAL(ch, y, x) (is_nchw ? GET_VAL_NCHW(ch, y, x) : GET_VAL_NHWC(y, x, ch))

    // Debug: print ALL 8 channels at [3,3] to verify channel layout
    printf("  All 8 channels at [3,3]:\n");
    for (int c = 0; c < 8; c++) {
        float val = GET_VAL(c, 3, 3);
        printf("    ch[%d]=%.2f ", c, val);
    }
    printf("\n");
    printf("  Interpretation: obj=ch0, box=ch1-4, person=ch5, cat=ch6, dog=ch7\n");

    for (int y = 0; y < grid_h; y++) {
        for (int x = 0; x < grid_w; x++) {
            // TinyDet format: [obj, x_off, y_off, w, h, cls0, cls1, cls2]
            // Channel 0 = objectness, Channels 1-4 = box, Channels 5-7 = classes
            float obj_val = GET_VAL(0, y, x);
            float obj_conf = sigmoid(obj_val);

            if (obj_conf < CONF_THRESHOLD) continue;

            // Find best class (channels 5+)
            int best_class = 0;
            float best_class_conf = 0.0f;
            for (int c = 0; c < num_classes; c++) {
                float cls_val = GET_VAL(5 + c, y, x);
                float cls_conf = sigmoid(cls_val);
                // Debug: print class loop for [3,3]
                if (y == 3 && x == 3) {
                    printf("  [3,3] c=%d ch=%d val=%.2f sig=%.3f best=%d\n", c, 5+c, cls_val, cls_conf, best_class);
                }
                if (cls_conf > best_class_conf) {
                    best_class_conf = cls_conf;
                    best_class = c;
                }
            }
            if (y == 3 && x == 3) {
                printf("  [3,3] FINAL: best_class=%d best_conf=%.3f\n", best_class, best_class_conf);
            }

            float final_conf = obj_conf * best_class_conf;
            if (final_conf < CONF_THRESHOLD) continue;

            // Decode bounding box (channels 1-4: x_off, y_off, w, h)
            // TinyDet training uses sigmoid on ALL box outputs:
            //   x_off = sigmoid(pred[1]) -> offset within cell (0-1)
            //   y_off = sigmoid(pred[2]) -> offset within cell (0-1)
            //   w = sigmoid(pred[3]) -> normalized width (0-1)
            //   h = sigmoid(pred[4]) -> normalized height (0-1)
            float x_off = sigmoid(GET_VAL(1, y, x));
            float y_off = sigmoid(GET_VAL(2, y, x));
            float w_norm = sigmoid(GET_VAL(3, y, x));
            float h_norm = sigmoid(GET_VAL(4, y, x));

            // Debug: print raw and sigmoid box values for first 5 detections
            if (num_dets < 5) {
                printf("  [%d,%d] RAW box: x=%.4f y=%.4f w=%.4f h=%.4f -> sigmoid: x=%.4f y=%.4f w=%.4f h=%.4f\n",
                       y, x, GET_VAL(1, y, x), GET_VAL(2, y, x), GET_VAL(3, y, x), GET_VAL(4, y, x),
                       x_off, y_off, w_norm, h_norm);
            }

            // Center position: (cell + offset) / grid_size = normalized position
            float cx = (x + x_off) / grid_w;  // normalized cx (0-1)
            float cy = (y + y_off) / grid_h;  // normalized cy (0-1)

            // Convert to x1, y1, x2, y2 (all normalized 0-1)
            float x1 = cx - w_norm / 2;
            float y1 = cy - h_norm / 2;
            float x2 = cx + w_norm / 2;
            float y2 = cy + h_norm / 2;

            // Clamp to [0, 1]
            if (x1 < 0) x1 = 0; if (x1 > 1) x1 = 1;
            if (y1 < 0) y1 = 0; if (y1 > 1) y1 = 1;
            if (x2 < 0) x2 = 0; if (x2 > 1) x2 = 1;
            if (y2 < 0) y2 = 0; if (y2 > 1) y2 = 1;

            if (num_dets < max_detections) {
                detections[num_dets].x1 = x1;
                detections[num_dets].y1 = y1;
                detections[num_dets].x2 = x2;
                detections[num_dets].y2 = y2;
                detections[num_dets].confidence = final_conf;
                detections[num_dets].class_id = best_class;
                num_dets++;

                printf("  Detection at [%d,%d]: %s %.1f%% box=[%.2f,%.2f,%.2f,%.2f]\n",
                       y, x, class_names[best_class], final_conf * 100, x1, y1, x2, y2);
            }
        }
    }

    #undef GET_VAL
    #undef GET_VAL_NHWC
    #undef GET_VAL_NCHW

    // Apply NMS
    if (num_dets > 0) {
        int before_nms = num_dets;
        num_dets = apply_nms(detections, num_dets, NMS_THRESHOLD);
        printf("  NMS: %d -> %d detections\n", before_nms, num_dets);
    }

    return num_dets;
}

// Decode all 3 YOLOv5 detection heads (for full YOLOv5 models)
static int decode_yolov5_heads(mars_model_t* model, Detection* detections, int max_detections) {
    int num_dets = 0;
    int num_outputs = mars_get_num_outputs(model);

    printf("\nDecoding %d detection heads...\n", num_outputs);

    for (int i = 0; i < num_outputs && i < 3; i++) {
        mars_runtime_tensor_t* output = mars_get_output(model, i);
        if (!output || !output->vaddr) {
            printf("Warning: Failed to get output %d\n", i);
            continue;
        }

        printf("  Head %d: shape=[%d,%d,%d,%d] scale=%.6f\n",
               i, output->desc.shape[0], output->desc.shape[1],
               output->desc.shape[2], output->desc.shape[3], output->desc.scale);

        num_dets = decode_head((const int8_t*)output->vaddr, output->desc.scale,
                               i, detections, max_detections, num_dets);
    }

    // Apply NMS to remove overlapping detections
    if (num_dets > 0) {
        int before_nms = num_dets;
        num_dets = apply_nms(detections, num_dets, NMS_THRESHOLD);
        printf("  NMS: %d -> %d detections (threshold=%.2f)\n", before_nms, num_dets, NMS_THRESHOLD);
    }

    return num_dets;
}

// Save detections to JSON file (for WebUI integration)
static void save_detections_json(const char* json_path, Detection* dets, int num_dets,
                                  float inference_ms, const char** class_names) {
    FILE* f = fopen("/tmp/detections.json.tmp", "w");
    if (!f) return;

    fprintf(f, "{\"inference_ms\":%.1f,\"count\":%d,\"detections\":[", inference_ms, num_dets);
    for (int i = 0; i < num_dets; i++) {
        if (i > 0) fprintf(f, ",");
        fprintf(f, "{\"class\":\"%s\",\"conf\":%.3f,\"box\":[%.4f,%.4f,%.4f,%.4f]}",
                class_names[dets[i].class_id], dets[i].confidence,
                dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);
    }
    fprintf(f, "]}\n");
    fclose(f);
    rename("/tmp/detections.json.tmp", json_path);
}

// Get file modification time
static time_t get_mtime(const char* path) {
    struct stat st;
    return (stat(path, &st) == 0) ? st.st_mtime : 0;
}

int main(int argc, char* argv[]) {
    const char* model_path = "/opt/yolov5n_qdq.mars";
    const char* image_path = "/tmp/snapshot.jpg";
    const char* output_path = NULL;
    const char* json_path = NULL;
    int daemon_mode = 0;
    int poll_ms = 500;

    // Parse arguments - collect positional args and flags
    int positional_count = 0;
    const char* positional[3] = {NULL, NULL, NULL};

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--daemon") == 0) {
            daemon_mode = 1;
            json_path = "/tmp/detections.json";
        } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            poll_ms = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            g_conf_threshold = atof(argv[++i]);
            if (g_conf_threshold > 1.0f) g_conf_threshold /= 100.0f;  // Allow percentage input
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            g_nms_threshold = atof(argv[++i]);
            if (g_nms_threshold > 1.0f) g_nms_threshold /= 100.0f;
        } else if (strcmp(argv[i], "--debug") == 0) {
            mars_set_debug(1);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [model.mars] [image.jpg] [output.jpg]\n", argv[0]);
            printf("       %s -d model.mars  # daemon mode\n", argv[0]);
            printf("Options:\n");
            printf("  -d, --daemon   Run as daemon, watch /tmp/snapshot.jpg\n");
            printf("  -j <path>      JSON output path (default: /tmp/detections.json)\n");
            printf("  -p <ms>        Poll interval in ms (default: 500)\n");
            printf("  -t <thresh>    Detection confidence threshold (default: 0.25 or 25%%)\n");
            printf("  -n <thresh>    NMS IoU threshold (default: 0.45)\n");
            printf("  --debug        Enable debug output for conv layers\n");
            return 0;
        } else if (argv[i][0] != '-' && positional_count < 3) {
            positional[positional_count++] = argv[i];
        }
    }

    // Assign positional args
    if (positional_count >= 1) model_path = positional[0];
    if (positional_count >= 2) image_path = positional[1];
    if (positional_count >= 3) output_path = positional[2];

    // In daemon mode, image_path is always /tmp/snapshot.jpg
    if (daemon_mode) {
        image_path = "/tmp/snapshot.jpg";
    }

    if (daemon_mode) {
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        printf("Mars Detection Daemon starting...\n");
        printf("  Model: %s\n", model_path);
        printf("  Watch: %s\n", image_path);
        printf("  JSON:  %s\n", json_path);
        printf("  Poll:  %d ms\n", poll_ms);
    } else {
        printf("\n");
        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  Mars YOLOv5 Detection                                   ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");
    }

    // Initialize NNA
    printf("Initializing NNA...\n");
    if (nna_init() != NNA_SUCCESS) {
        printf("Failed to initialize NNA\n");
        return 1;
    }

    // Load model
    printf("Loading model: %s\n", model_path);
    mars_model_t* model = NULL;
    mars_error_t err = mars_load_file(model_path, &model);
    if (err != MARS_OK) {
        printf("Failed to load model: %s\n", mars_get_error_string(err));
        nna_deinit();
        return 1;
    }

    // Get input tensor to determine actual input size
    mars_runtime_tensor_t* input = mars_get_input(model, 0);
    if (!input || !input->vaddr) {
        printf("Failed to get input tensor\n");
        mars_free(model);
        nna_deinit();
        return 1;
    }

    // Get input dimensions - detect NCHW vs NHWC
    // NCHW: [1, 3, 160, 160] - channels (3) in position 1
    // NHWC: [1, 160, 160, 3] - channels (3) in position 3
    int s1 = input->desc.shape[1];
    int s2 = input->desc.shape[2];
    int s3 = input->desc.shape[3];
    int input_is_nchw = (s1 < s2 && s1 < s3);  // smallest dim in position 1 = NCHW

    int input_h, input_w, input_c;
    if (input_is_nchw) {
        input_c = s1;
        input_h = s2;
        input_w = s3;
        printf("Model input (NCHW): %dx%dx%d (HxWxC)\n", input_h, input_w, input_c);
    } else {
        input_h = s1;
        input_w = s2;
        input_c = s3;
        printf("Model input (NHWC): %dx%dx%d (HxWxC)\n", input_h, input_w, input_c);
    }

    // Allocate image buffer from regular heap (doesn't need NNA memory)
    uint8_t* rgb_image = (uint8_t*)malloc(input_h * input_w * 3);
    if (!rgb_image) {
        printf("Failed to allocate image buffer\n");
        mars_free(model);
        nna_deinit();
        return 1;
    }

    // Pre-compute model type info - detect output format too
    int num_outputs = mars_get_num_outputs(model);
    mars_runtime_tensor_t* out0 = mars_get_output(model, 0);
    int os1 = out0 ? out0->desc.shape[1] : 0;
    int os2 = out0 ? out0->desc.shape[2] : 0;
    int os3 = out0 ? out0->desc.shape[3] : 0;
    int out_is_nchw = (os1 < os2 && os1 < os3);
    int out_channels = out_is_nchw ? os1 : os3;
    int is_tinydet = (num_outputs == 1 && out_channels < 255);
    int num_classes = out_channels - 5;  // output_channels = 5 + num_classes
    const char** class_names_ptr;
    if (is_tinydet && num_classes == 3) {
        class_names_ptr = tinydet_classes;
    } else if (is_tinydet && num_classes == 4) {
        class_names_ptr = security_classes;
    } else {
        class_names_ptr = coco_classes;
    }

    /* Use static to avoid stack overflow on embedded systems */
    static Detection detections[200];
    time_t last_mtime = 0;

    // Main loop (runs once in normal mode, continuously in daemon mode)
    do {
        // In daemon mode, wait for file change
        if (daemon_mode) {
            time_t mtime = get_mtime(image_path);
            if (mtime <= last_mtime) {
                usleep(poll_ms * 1000);
                continue;
            }
            last_mtime = mtime;
        }

        // Load original image for output drawing
        int orig_w = 0, orig_h = 0;
        uint8_t* orig_image = NULL;
        if (output_path && !daemon_mode) {
            orig_image = load_original_image(image_path, &orig_w, &orig_h);
        }

        // Load image (resized for inference)
        if (!daemon_mode) printf("Loading image: %s\n", image_path);
        if (load_test_image(image_path, rgb_image, input_w, input_h) != 0) {
            fprintf(stderr, "ERROR: Failed to load image: %s\n", image_path);
            if (daemon_mode) { usleep(poll_ms * 1000); continue; }
            if (orig_image) { free(orig_image); orig_image = NULL; }
            free(rgb_image);
            mars_free(model);
            nna_deinit();
            return 1;
        }

        if (!daemon_mode) printf("Preprocessing image...\n");
        // Check input tensor dtype and format to use appropriate preprocessing
        int is_float32_input = (input->desc.dtype == MARS_DTYPE_FLOAT32);
        int is_nhwc = (input->desc.format == MARS_FORMAT_NHWC);

        if (is_float32_input && is_nhwc) {
            preprocess_image_float32_nhwc(rgb_image, (float*)input->vaddr, input_w, input_h);
            nna_cache_flush(input->vaddr, input_h * input_w * input_c * sizeof(float));
        } else if (is_float32_input) {
            // NCHW format
            preprocess_image_float32_nchw(rgb_image, (float*)input->vaddr, input_w, input_h);
            nna_cache_flush(input->vaddr, input_h * input_w * input_c * sizeof(float));
        } else if (is_nhwc) {
            preprocess_image_int8(rgb_image, (int8_t*)input->vaddr, input_w, input_h, input->desc.scale);
            nna_cache_flush(input->vaddr, input_h * input_w * input_c);
        } else {
            // INT8 NCHW format
            preprocess_image_int8_nchw(rgb_image, (int8_t*)input->vaddr, input_w, input_h, input->desc.scale);
            nna_cache_flush(input->vaddr, input_h * input_w * input_c);
        }

        // Run inference with timing
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        if (!daemon_mode) printf("Running inference...\n");
        int ret = mars_run(model);
        if (ret != MARS_OK) {
            if (!daemon_mode) printf("Inference failed: %d\n", ret);
            if (daemon_mode) { usleep(poll_ms * 1000); continue; }
            else break;
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        float inference_ms = (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_nsec - t0.tv_nsec) / 1e6f;

        memset(detections, 0, sizeof(detections));
        int num_dets = 0;

        if (is_tinydet) {
            if (!daemon_mode) printf("Detected TinyDet model (output channels=%d)\n", out_channels);
            num_dets = decode_tinydet_output(model, detections, 200, input_w, input_h);
        } else {
            if (!daemon_mode) printf("Detected YOLOv5 model (%d outputs)\n", num_outputs);
            num_dets = decode_yolov5_heads(model, detections, 200);
        }

        // Daemon mode: save JSON and print summary
        if (daemon_mode) {
            save_detections_json(json_path, detections, num_dets, inference_ms, class_names_ptr);
            fprintf(stderr, "[%ld] %d detections (%.0fms)\n", (long)last_mtime, num_dets, inference_ms);
        } else {
            // Normal mode: pretty print
            printf("\n");
            printf("╔══════════════════════════════════════════════════════════╗\n");
            printf("║  Detections: %-4d                                        ║\n", num_dets);
            printf("╚══════════════════════════════════════════════════════════╝\n\n");

            for (int i = 0; i < num_dets && i < 20; i++) {
                Detection* d = &detections[i];
                printf("  [%2d] %s (%.1f%%): [%.2f, %.2f, %.2f, %.2f]\n",
                       i, class_names_ptr[d->class_id], d->confidence * 100,
                       d->x1, d->y1, d->x2, d->y2);
            }

            if (num_dets == 0) {
                printf("  No detections above threshold (%.0f%%)\n", CONF_THRESHOLD * 100);
            }

            // Draw bounding boxes and save output image if requested
            if (output_path && num_dets > 0) {
                printf("\nDrawing %d bounding boxes...\n", num_dets);
                // Use original image if available, otherwise use resized
                if (orig_image && orig_w > 0 && orig_h > 0) {
                    printf("Drawing on original %dx%d image\n", orig_w, orig_h);
                    draw_detections(orig_image, orig_w, orig_h, detections, num_dets);
                    save_image(output_path, orig_image, orig_w, orig_h);
                } else {
                    draw_detections(rgb_image, input_w, input_h, detections, num_dets);
                    save_image(output_path, rgb_image, input_w, input_h);
                }
            } else if (output_path) {
                if (orig_image && orig_w > 0 && orig_h > 0) {
                    save_image(output_path, orig_image, orig_w, orig_h);
                } else {
                    save_image(output_path, rgb_image, input_w, input_h);
                }
            }
        }

        // Free original image if allocated
        if (orig_image) { free(orig_image); orig_image = NULL; }

    } while (daemon_mode && g_running);

    // Cleanup
    if (!daemon_mode) printf("\nCleaning up...\n");
    else fprintf(stderr, "Shutting down...\n");
    free(rgb_image);
    mars_free(model);
    nna_deinit();
    if (!daemon_mode) printf("Done!\n");

    return 0;
}

