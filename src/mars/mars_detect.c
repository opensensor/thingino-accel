/**
 * Mars YOLOv5 Detection Test
 *
 * Loads a JPEG image, runs YOLOv5 inference, and outputs detections
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "mars.h"
#include "mars_runtime.h"
#include "nna.h"

/* External cache flush for MIPS */
extern void nna_cache_flush(void *ptr, size_t size);

// YOLO parameters
#define CONF_THRESHOLD 0.35f  // Detection confidence threshold
#define NMS_THRESHOLD 0.45f
#define NUM_CLASSES 80
#define INPUT_SIZE 640
#define NUM_ANCHORS 3

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

// COCO class names
static const char* class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

typedef struct {
    float x1, y1, x2, y2;  // Bounding box
    float confidence;
    int class_id;
} Detection;

// Sigmoid function
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Simple JPEG loader using stb_image-style approach
// For now, we'll use a simpler PPM format or raw RGB
static int load_test_image(const char* path, uint8_t* rgb_data, int target_w, int target_h) {
    // Try to load as raw RGB first (for testing)
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Cannot open image: %s\n", path);
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
        // PPM format - read header more carefully
        int w = 0, h = 0, maxval = 0;
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
        if (fscanf(f, "%d", &w) != 1) { fclose(f); return -1; }

        // Skip whitespace
        while ((c = fgetc(f)) != EOF && (c == ' ' || c == '\t' || c == '\n' || c == '\r'));

        // Skip comments
        while (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n');
            c = fgetc(f);
        }

        // Read height
        ungetc(c, f);
        if (fscanf(f, "%d", &h) != 1) { fclose(f); return -1; }

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

        printf("Loading PPM image: %dx%d (maxval=%d)\n", w, h, maxval);
        fflush(stdout);

        if (w == target_w && h == target_h) {
            // Direct read - no resize needed
            size_t bytes_read = fread(rgb_data, 1, w * h * 3, f);
            printf("Read %zu bytes directly\n", bytes_read);
            fclose(f);
            return (bytes_read == (size_t)(w * h * 3)) ? 0 : -1;
        }

        // Read and resize to target size
        uint8_t* temp = (uint8_t*)malloc(w * h * 3);
        if (!temp) { fclose(f); return -1; }

        size_t bytes_read = fread(temp, 1, w * h * 3, f);
        printf("Read %zu bytes, resizing from %dx%d to %dx%d\n",
               bytes_read, w, h, target_w, target_h);

        // Simple nearest-neighbor resize
        for (int y = 0; y < target_h; y++) {
            for (int x = 0; x < target_w; x++) {
                int src_x = x * w / target_w;
                int src_y = y * h / target_h;
                int src_idx = (src_y * w + src_x) * 3;
                int dst_idx = (y * target_w + x) * 3;
                rgb_data[dst_idx + 0] = temp[src_idx + 0];
                rgb_data[dst_idx + 1] = temp[src_idx + 1];
                rgb_data[dst_idx + 2] = temp[src_idx + 2];
            }
        }
        free(temp);
        fclose(f);
        return 0;
    }
    
    fclose(f);
    
    // For JPEG, we'd need libjpeg - for now generate test pattern
    printf("Note: JPEG loading not implemented, using gradient test pattern\n");
    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            int idx = (y * target_w + x) * 3;
            rgb_data[idx + 0] = (uint8_t)(x * 255 / target_w);  // R gradient
            rgb_data[idx + 1] = (uint8_t)(y * 255 / target_h);  // G gradient
            rgb_data[idx + 2] = 128;  // B constant
        }
    }
    return 0;
}

// Preprocess RGB image to INT8 NHWC format
// Uses symmetric quantization: int8 = float / scale (zero_point = 0)
static void preprocess_image(const uint8_t* rgb, int8_t* output, int w, int h, float scale) {
    for (int i = 0; i < w * h * 3; i++) {
        float val = rgb[i] / 255.0f;  // Normalize to [0, 1]
        int32_t quantized = (int32_t)(val / scale);
        // Clamp to int8 range
        if (quantized > 127) quantized = 127;
        if (quantized < -128) quantized = -128;
        output[i] = (int8_t)quantized;
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

    // Debug first detection
    printf("\n[Scale %d] Grid=%dx%d, stride=%d, scale=%.6f\n",
           scale_idx, grid_h, grid_w, stride, scale);
    printf("  First 85 raw values (anchor 0 at pos 0,0): ");
    for (int i = 0; i < 85; i++) {
        if (i == 5) printf("| ");  // Separate box from classes
        printf("%d ", data[i]);
    }
    printf("\n");

    // Decode first position for debug
    {
        float bx = data[0] * scale;
        float by = data[1] * scale;
        float bw = data[2] * scale;
        float bh = data[3] * scale;
        float obj = data[4] * scale;
        printf("  Pos[0,0,a0]: raw=[%d,%d,%d,%d,%d] dequant=[%.3f,%.3f,%.3f,%.3f,%.3f]\n",
               data[0], data[1], data[2], data[3], data[4], bx, by, bw, bh, obj);
        printf("               sigmoid=[%.3f,%.3f,%.3f,%.3f,%.3f]\n",
               sigmoid(bx), sigmoid(by), sigmoid(bw), sigmoid(bh), sigmoid(obj));
    }

    // NHWC layout: [1, H, W, 255]
    // 255 = 3 anchors * 85 values per anchor
    // For each position, data is interleaved: anchor0[85], anchor1[85], anchor2[85]

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

                // Dequantize box values
                float bx = box[0] * scale;
                float by = box[1] * scale;
                float bw = box[2] * scale;
                float bh = box[3] * scale;
                float obj = box[4] * scale;

                // Apply sigmoid to objectness
                float obj_conf = sigmoid(obj);

                // Track max objectness for debug
                if (obj_conf > max_obj_conf) {
                    max_obj_conf = obj_conf;
                    max_obj_y = y;
                    max_obj_x = x;
                    max_obj_a = a;
                    max_obj_raw = box[4];
                }

                if (obj_conf < CONF_THRESHOLD) continue;
                if (num_dets >= max_detections) continue;

                // Find best class
                int best_class = 0;
                float best_score = -1e9f;
                for (int c = 0; c < NUM_CLASSES; c++) {
                    float score = box[5 + c] * scale;
                    if (score > best_score) {
                        best_score = score;
                        best_class = c;
                    }
                }

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
static void draw_detections(uint8_t* rgb, int w, int h, Detection* dets, int num_dets) {
    for (int i = 0; i < num_dets; i++) {
        Detection* d = &dets[i];
        int x1 = (int)d->x1, y1 = (int)d->y1;
        int x2 = (int)d->x2, y2 = (int)d->y2;

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

// Save image as PPM
static int save_ppm(const char* path, const uint8_t* rgb, int w, int h) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        printf("Error: Cannot open %s for writing\n", path);
        return -1;
    }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(rgb, 1, w * h * 3, f);
    fclose(f);
    printf("Saved output image: %s\n", path);
    return 0;
}

// Decode all 3 YOLOv5 detection heads
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

    return num_dets;
}

int main(int argc, char* argv[]) {
    const char* model_path = "/opt/yolov5n_qdq.mars";
    const char* image_path = "/tmp/snapshot.jpg";
    const char* output_path = NULL;

    if (argc > 1) model_path = argv[1];
    if (argc > 2) image_path = argv[2];
    if (argc > 3) output_path = argv[3];
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Mars YOLOv5 Detection                                   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

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

    // Allocate image buffer
    uint8_t* rgb_image = (uint8_t*)malloc(INPUT_SIZE * INPUT_SIZE * 3);

    // Load image
    printf("Loading image: %s\n", image_path);
    if (load_test_image(image_path, rgb_image, INPUT_SIZE, INPUT_SIZE) != 0) {
        printf("Using test pattern instead\n");
    }

    // Get input tensor and preprocess
    mars_runtime_tensor_t* input = mars_get_input(model, 0);
    if (!input || !input->vaddr) {
        printf("Failed to get input tensor\n");
        free(rgb_image);
        mars_free(model);
        nna_deinit();
        return 1;
    }

    printf("Preprocessing image (scale=%.6f)...\n", input->desc.scale);
    printf("  [DEBUG] Input tensor id=%u, vaddr=%p\n", input->desc.id, input->vaddr);
    printf("  [DEBUG] RGB first 16 bytes: ");
    for (int i = 0; i < 16; i++) printf("%d ", rgb_image[i]);
    printf("\n");

    preprocess_image(rgb_image, (int8_t*)input->vaddr, INPUT_SIZE, INPUT_SIZE, input->desc.scale);

    printf("  [DEBUG] Preprocessed first 16 bytes: ");
    int8_t* inp = (int8_t*)input->vaddr;
    for (int i = 0; i < 16; i++) printf("%d ", inp[i]);
    printf("\n");

    /* Flush cache to ensure data is in memory before inference */
    nna_cache_flush(input->vaddr, INPUT_SIZE * INPUT_SIZE * 3);

    // Run inference
    printf("Running inference...\n");
    int ret = mars_run(model);
    if (ret != MARS_OK) {
        printf("Inference failed: %d\n", ret);
        mars_free(model);
        nna_deinit();
        return 1;
    }

    // Check number of outputs and decode detections
    int num_outputs = mars_get_num_outputs(model);
    printf("\nModel has %d outputs\n", num_outputs);

    Detection detections[500];
    int num_dets = decode_yolov5_heads(model, detections, 500);

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Detections: %-4d                                        ║\n", num_dets);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // Print top detections
    for (int i = 0; i < num_dets && i < 20; i++) {
        Detection* d = &detections[i];
        printf("  [%2d] %s (%.1f%%): [%.0f, %.0f, %.0f, %.0f]\n",
               i, class_names[d->class_id], d->confidence * 100,
               d->x1, d->y1, d->x2, d->y2);
    }

    if (num_dets == 0) {
        printf("  No detections above threshold (%.0f%%)\n", CONF_THRESHOLD * 100);
    }

    // Draw bounding boxes and save output image if requested
    if (output_path && num_dets > 0) {
        printf("\nDrawing %d bounding boxes...\n", num_dets);
        draw_detections(rgb_image, INPUT_SIZE, INPUT_SIZE, detections, num_dets);
        save_ppm(output_path, rgb_image, INPUT_SIZE, INPUT_SIZE);
    } else if (output_path) {
        // Save original image even if no detections
        save_ppm(output_path, rgb_image, INPUT_SIZE, INPUT_SIZE);
    }

    // Cleanup
    printf("\nCleaning up...\n");
    free(rgb_image);
    mars_free(model);
    nna_deinit();
    printf("Done!\n");

    return 0;
}

