/*
 * thingino-accel - YOLOv5s Detection Example
 *
 * Loads YOLOv5s .mgk model, processes /tmp/snapshot.jpg, outputs detections
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>

/* NNA library headers */
extern "C" {
#include "nna.h"
#include "nna_model.h"
#include "nna_tensor.h"
}

/* stb_image for JPEG loading */
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"

/* YOLO configuration */
#define YOLO_INPUT_WIDTH  640
#define YOLO_INPUT_HEIGHT 640
#define YOLO_NUM_CLASSES  80
#define YOLO_CONF_THRESH  0.25f
#define YOLO_NMS_THRESH   0.45f

/* Default paths */
#define DEFAULT_MODEL_PATH "/tmp/yolov5s_t41_magik_post_release.mgk"
#define DEFAULT_IMAGE_PATH "/tmp/snapshot.jpg"

/* Detection structure */
struct Detection {
    float x0, y0, x1, y1;  /* Bounding box coordinates */
    float confidence;
    int class_id;
};

/* COCO class names (subset for display) */
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

/* Print header */
static void print_header(const char *title) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  %-54s  ║\n", title);
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

/* Load and preprocess image for YOLO input */
static uint8_t* load_and_preprocess_image(const char *path, int *orig_w, int *orig_h) {
    int channels;
    unsigned char *img = stbi_load(path, orig_w, orig_h, &channels, 3);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", path);
        return NULL;
    }

    printf("Loaded image: %dx%d channels=%d\n", *orig_w, *orig_h, channels);

    /* Allocate output buffer for resized RGBA image */
    size_t output_size = YOLO_INPUT_WIDTH * YOLO_INPUT_HEIGHT * 4;
    uint8_t *output = (uint8_t*)malloc(output_size);
    if (!output) {
        stbi_image_free(img);
        return NULL;
    }

    /* Calculate letterbox resize parameters */
    float scale = fminf((float)YOLO_INPUT_WIDTH / *orig_w,
                        (float)YOLO_INPUT_HEIGHT / *orig_h);
    int new_w = (int)(*orig_w * scale);
    int new_h = (int)(*orig_h * scale);
    int pad_x = (YOLO_INPUT_WIDTH - new_w) / 2;
    int pad_y = (YOLO_INPUT_HEIGHT - new_h) / 2;

    printf("Resize: %dx%d -> %dx%d (scale=%.3f, pad=%d,%d)\n",
           *orig_w, *orig_h, new_w, new_h, scale, pad_x, pad_y);

    /* Fill with gray (114) for letterboxing */
    memset(output, 114, output_size);

    /* Resize image to temp buffer */
    uint8_t *resized = (uint8_t*)malloc(new_w * new_h * 3);
    if (!resized) {
        free(output);
        stbi_image_free(img);
        return NULL;
    }

    stbir_resize_uint8(img, *orig_w, *orig_h, 0,
                       resized, new_w, new_h, 0, 3);

    /* Copy resized image to RGBA output with letterboxing */
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            int dst_idx = ((y + pad_y) * YOLO_INPUT_WIDTH + (x + pad_x)) * 4;
            int src_idx = (y * new_w + x) * 3;
            output[dst_idx + 0] = resized[src_idx + 0];  /* R */
            output[dst_idx + 1] = resized[src_idx + 1];  /* G */
            output[dst_idx + 2] = resized[src_idx + 2];  /* B */
            output[dst_idx + 3] = 0;                      /* A */
        }
    }

    free(resized);
    stbi_image_free(img);
    return output;
}

/* Sigmoid function */
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* Intersection over Union */
static float iou(const Detection &a, const Detection &b) {
    float x0 = fmaxf(a.x0, b.x0);
    float y0 = fmaxf(a.y0, b.y0);
    float x1 = fminf(a.x1, b.x1);
    float y1 = fminf(a.y1, b.y1);

    float inter = fmaxf(0, x1 - x0) * fmaxf(0, y1 - y0);
    float area_a = (a.x1 - a.x0) * (a.y1 - a.y0);
    float area_b = (b.x1 - b.x0) * (b.y1 - b.y0);

    return inter / (area_a + area_b - inter + 1e-6f);
}

/* Non-Maximum Suppression */
static void nms(std::vector<Detection> &dets, float thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection &a, const Detection &b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> result;

    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            if (dets[i].class_id == dets[j].class_id &&
                iou(dets[i], dets[j]) > thresh) {
                suppressed[j] = true;
            }
        }
    }
    dets = result;
}

/* YOLOv5 anchor boxes */
static const float ANCHORS[3][6] = {
    {10, 13, 16, 30, 33, 23},      /* P3/8 stride */
    {30, 61, 62, 45, 59, 119},     /* P4/16 stride */
    {116, 90, 156, 198, 373, 326}  /* P5/32 stride */
};
static const int STRIDES[3] = {8, 16, 32};

/* Parse YOLO output tensor and generate detections */
static void parse_yolo_output(const nna_tensor_t *output, int stride_idx,
                              std::vector<Detection> &dets,
                              int orig_w, int orig_h) {
    const nna_shape_t *shape = nna_tensor_shape(output);
    if (!shape || shape->ndim < 3) {
        fprintf(stderr, "Invalid output tensor shape\n");
        return;
    }

    /* Output shape: [1, H, W, num_anchors * (5 + num_classes)] */
    /* or [1, num_anchors, H, W, 5 + num_classes] depending on model */
    int h = shape->dims[1];
    int w = shape->dims[2];
    int stride = STRIDES[stride_idx];

    /* For now, log output shape for debugging */
    printf("  Output[%d] shape: [%d, %d, %d, %d] stride=%d\n",
           stride_idx, shape->dims[0], h, w, shape->dims[3], stride);

    /* TODO: Parse actual detections based on model output format */
    /* This requires knowing the exact output format of the .mgk model */
}

/* Scale detection boxes back to original image coordinates */
static void scale_detections(std::vector<Detection> &dets,
                             int orig_w, int orig_h) {
    float scale = fminf((float)YOLO_INPUT_WIDTH / orig_w,
                        (float)YOLO_INPUT_HEIGHT / orig_h);
    float pad_x = (YOLO_INPUT_WIDTH - orig_w * scale) / 2;
    float pad_y = (YOLO_INPUT_HEIGHT - orig_h * scale) / 2;

    for (auto &det : dets) {
        det.x0 = (det.x0 - pad_x) / scale;
        det.y0 = (det.y0 - pad_y) / scale;
        det.x1 = (det.x1 - pad_x) / scale;
        det.y1 = (det.y1 - pad_y) / scale;

        /* Clamp to image bounds */
        det.x0 = fmaxf(0, fminf(det.x0, (float)orig_w - 1));
        det.y0 = fmaxf(0, fminf(det.y0, (float)orig_h - 1));
        det.x1 = fmaxf(0, fminf(det.x1, (float)orig_w - 1));
        det.y1 = fmaxf(0, fminf(det.y1, (float)orig_h - 1));
    }
}

/* Main function */
int main(int argc, char **argv) {
    const char *model_path = (argc > 1) ? argv[1] : DEFAULT_MODEL_PATH;
    const char *image_path = (argc > 2) ? argv[2] : DEFAULT_IMAGE_PATH;

    print_header("thingino-accel - YOLOv5s Detection");

    printf("Model: %s\n", model_path);
    printf("Image: %s\n", image_path);
    printf("\n");

    /* Initialize NNA hardware */
    printf("[1] Initializing NNA...\n");
    if (nna_init() != NNA_SUCCESS) {
        fprintf(stderr, "Failed to initialize NNA\n");
        return 1;
    }
    printf("    NNA initialized successfully\n");

    /* Load model */
    printf("[2] Loading model...\n");
    nna_model_t *model = nna_model_load(model_path, NULL);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        nna_deinit();
        return 1;
    }

    nna_model_info_t info;
    nna_model_get_info(model, &info);
    printf("    Model loaded: %u inputs, %u outputs\n",
           info.num_inputs, info.num_outputs);

    /* Load and preprocess image */
    printf("[3] Loading image...\n");
    int orig_w, orig_h;
    uint8_t *input_data = load_and_preprocess_image(image_path, &orig_w, &orig_h);
    if (!input_data) {
        nna_model_unload(model);
        nna_deinit();
        return 1;
    }

    /* Get input tensor and copy preprocessed data */
    printf("[4] Setting input data...\n");
    nna_tensor_t *input = nna_model_get_input(model, 0);
    if (!input) {
        fprintf(stderr, "Failed to get input tensor\n");
        free(input_data);
        nna_model_unload(model);
        nna_deinit();
        return 1;
    }

    void *input_ptr = nna_tensor_data(input);
    if (input_ptr) {
        size_t input_size = YOLO_INPUT_WIDTH * YOLO_INPUT_HEIGHT * 4;
        memcpy(input_ptr, input_data, input_size);
        printf("    Copied %zu bytes to input tensor\n", input_size);
    }
    free(input_data);

    /* Run inference */
    printf("[5] Running inference...\n");
    int ret = nna_model_run(model);
    if (ret != NNA_SUCCESS) {
        fprintf(stderr, "Inference failed with code %d\n", ret);
        nna_model_unload(model);
        nna_deinit();
        return 1;
    }
    printf("    Inference completed\n");

    /* Parse outputs */
    printf("[6] Parsing detections...\n");
    std::vector<Detection> detections;

    for (uint32_t i = 0; i < info.num_outputs && i < 3; i++) {
        const nna_tensor_t *output = nna_model_get_output(model, i);
        if (output) {
            parse_yolo_output(output, i, detections, orig_w, orig_h);
        }
    }

    /* Apply NMS */
    nms(detections, YOLO_NMS_THRESH);

    /* Scale to original image coordinates */
    scale_detections(detections, orig_w, orig_h);

    /* Print results */
    printf("\n");
    print_header("Detection Results");
    printf("Found %zu detections:\n\n", detections.size());

    for (size_t i = 0; i < detections.size(); i++) {
        const Detection &det = detections[i];
        const char *class_name = (det.class_id < YOLO_NUM_CLASSES)
                                 ? CLASS_NAMES[det.class_id] : "unknown";
        printf("  [%zu] %s: %.1f%% at (%.0f, %.0f) - (%.0f, %.0f)\n",
               i + 1, class_name, det.confidence * 100,
               det.x0, det.y0, det.x1, det.y1);
    }

    if (detections.empty()) {
        printf("  (No detections above threshold %.2f)\n", YOLO_CONF_THRESH);
    }

    /* Cleanup */
    printf("\n[7] Cleanup...\n");
    nna_model_unload(model);
    nna_deinit();
    printf("    Done!\n\n");

    return 0;
}

