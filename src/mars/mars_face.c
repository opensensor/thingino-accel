/**
 * Mars Face Detection
 *
 * Loads a JPEG image, runs face detection inference using the face_cnn model
 * Output format: Region layer with 5 anchors, 1 class (face)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

#include "mars.h"
#include "mars_runtime.h"
#include "nna.h"
#include "nna_memory.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

extern void nna_cache_flush(void *ptr, size_t size);

// Face detection parameters
#define CONF_THRESHOLD 0.5f  // Higher threshold to reduce false positives
#define NMS_THRESHOLD 0.4f
#define INPUT_SIZE 416
#define NUM_ANCHORS 5
#define GRID_SIZE 13

// Anchors from face_cnn config (width, height pairs)
static const float anchors[NUM_ANCHORS][2] = {
    {0.7f, 0.86f},
    {2.1f, 2.1f},
    {4.0f, 4.16f},
    {8.1f, 8.1f},
    {12.0f, 12.16f}
};

typedef struct {
    float x, y, w, h;
    float confidence;
} FaceBox;

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float iou(FaceBox* a, FaceBox* b) {
    float x1 = fmaxf(a->x - a->w/2, b->x - b->w/2);
    float y1 = fmaxf(a->y - a->h/2, b->y - b->h/2);
    float x2 = fminf(a->x + a->w/2, b->x + b->w/2);
    float y2 = fminf(a->y + a->h/2, b->y + b->h/2);
    
    float inter = fmaxf(0, x2 - x1) * fmaxf(0, y2 - y1);
    float area_a = a->w * a->h;
    float area_b = b->w * b->h;
    return inter / (area_a + area_b - inter + 1e-6f);
}

// Decode face detections from Region layer output (float32 version)
// Output format: [1, 13, 13, 30] where 30 = 5 anchors * 6 values (x,y,w,h,obj,class)
static int decode_faces_float(const float* data, FaceBox* faces, int max_faces) {
    int num_faces = 0;
    const int stride = INPUT_SIZE / GRID_SIZE;  // 32

    // Debug: print first few raw values
    printf("  Debug: first 30 output values (float32):\n    ");
    for (int i = 0; i < 30; i++) {
        printf("%.2f ", data[i]);
    }
    printf("\n");

    // Debug: find max confidence
    float max_obj_conf = 0.0f;
    int max_obj_x = 0, max_obj_y = 0, max_obj_a = 0;

    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            for (int a = 0; a < NUM_ANCHORS; a++) {
                // NHWC layout: [1, H, W, C] where C = 30
                int base_idx = (y * GRID_SIZE + x) * 30 + a * 6;

                float tx = data[base_idx + 0];
                float ty = data[base_idx + 1];
                float tw = data[base_idx + 2];
                float th = data[base_idx + 3];
                float obj = data[base_idx + 4];
                float cls = data[base_idx + 5];

                float obj_conf = sigmoid(obj);
                float cls_conf = sigmoid(cls);
                float confidence = obj_conf * cls_conf;

                if (obj_conf > max_obj_conf) {
                    max_obj_conf = obj_conf;
                    max_obj_x = x;
                    max_obj_y = y;
                    max_obj_a = a;
                }

                if (confidence < CONF_THRESHOLD) continue;

                // Decode box (Darknet/YOLO v2 style)
                float cx = (sigmoid(tx) + x) * stride;
                float cy = (sigmoid(ty) + y) * stride;

                // Clamp tw/th to prevent exp() overflow
                if (tw > 10.0f) tw = 10.0f;
                if (th > 10.0f) th = 10.0f;
                if (tw < -10.0f) tw = -10.0f;
                if (th < -10.0f) th = -10.0f;

                float w = expf(tw) * anchors[a][0] * stride;
                float h = expf(th) * anchors[a][1] * stride;

                if (num_faces < max_faces) {
                    faces[num_faces].x = cx;
                    faces[num_faces].y = cy;
                    faces[num_faces].w = w;
                    faces[num_faces].h = h;
                    faces[num_faces].confidence = confidence;
                    num_faces++;
                }
            }
        }
    }

    printf("  Debug: max objectness conf=%.4f at grid[%d,%d] anchor=%d\n",
           max_obj_conf, max_obj_x, max_obj_y, max_obj_a);
    printf("  Debug: num_faces before NMS = %d (threshold=%.2f)\n", num_faces, CONF_THRESHOLD);

    // Simple NMS
    for (int i = 0; i < num_faces; i++) {
        if (faces[i].confidence == 0) continue;
        for (int j = i + 1; j < num_faces; j++) {
            if (faces[j].confidence == 0) continue;
            if (iou(&faces[i], &faces[j]) > NMS_THRESHOLD) {
                faces[j].confidence = 0;
            }
        }
    }

    // Compact remaining faces
    int count = 0;
    for (int i = 0; i < num_faces; i++) {
        if (faces[i].confidence > 0) {
            if (count != i) faces[count] = faces[i];
            count++;
        }
    }

    return count;
}

// Decode face detections from Region layer output (int8 version)
// Output format: [1, 13, 13, 30] where 30 = 5 anchors * 6 values (x,y,w,h,obj,class)
static int decode_faces_int8(const int8_t* data, float scale, FaceBox* faces, int max_faces) {
    int num_faces = 0;
    const int stride = INPUT_SIZE / GRID_SIZE;  // 32

    // Debug: print first few raw values
    printf("  Debug: first 30 output values (raw INT8):\n    ");
    for (int i = 0; i < 30; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");

    // Debug: scan all objectness values and find statistics
    int8_t min_obj = 127, max_obj = -128;
    int num_positive_obj = 0;
    int num_saturated_low = 0, num_saturated_high = 0;
    int total_cells = GRID_SIZE * GRID_SIZE * NUM_ANCHORS;

    for (int i = 0; i < total_cells; i++) {
        int grid_idx = i / NUM_ANCHORS;
        int anchor = i % NUM_ANCHORS;
        int base_idx = grid_idx * 30 + anchor * 6;
        int8_t obj_raw = data[base_idx + 4];

        if (obj_raw > max_obj) max_obj = obj_raw;
        if (obj_raw < min_obj) min_obj = obj_raw;
        if (obj_raw > 0) num_positive_obj++;
        if (obj_raw == -128) num_saturated_low++;
        if (obj_raw == 127) num_saturated_high++;
    }
    printf("  Debug: objectness stats across all %d cells:\n", total_cells);
    printf("    min=%d max=%d positive=%d saturated_low=%d saturated_high=%d\n",
           min_obj, max_obj, num_positive_obj, num_saturated_low, num_saturated_high);

    // Debug: find max confidence
    float max_obj_conf = 0.0f;
    int max_obj_x = 0, max_obj_y = 0, max_obj_a = 0;
    int8_t max_obj_raw = -128;

    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            for (int a = 0; a < NUM_ANCHORS; a++) {
                // NHWC layout: [1, H, W, C] where C = 30
                int base_idx = (y * GRID_SIZE + x) * 30 + a * 6;

                // Dequantize values
                float tx = data[base_idx + 0] * scale;
                float ty = data[base_idx + 1] * scale;
                float tw = data[base_idx + 2] * scale;
                float th = data[base_idx + 3] * scale;
                float obj = data[base_idx + 4] * scale;
                float cls = data[base_idx + 5] * scale;

                float obj_conf = sigmoid(obj);
                float cls_conf = sigmoid(cls);
                float confidence = obj_conf * cls_conf;

                if (obj_conf > max_obj_conf) {
                    max_obj_conf = obj_conf;
                    max_obj_x = x;
                    max_obj_y = y;
                    max_obj_a = a;
                    max_obj_raw = data[base_idx + 4];
                }

                if (confidence < CONF_THRESHOLD) continue;

                // Decode box (Darknet/YOLO v2 style)
                float cx = (sigmoid(tx) + x) * stride;
                float cy = (sigmoid(ty) + y) * stride;

                // Clamp tw/th to prevent exp() overflow (raw INT8 can be very large)
                if (tw > 10.0f) tw = 10.0f;
                if (th > 10.0f) th = 10.0f;
                if (tw < -10.0f) tw = -10.0f;
                if (th < -10.0f) th = -10.0f;

                float w = expf(tw) * anchors[a][0] * stride;
                float h = expf(th) * anchors[a][1] * stride;
                
                if (num_faces < max_faces) {
                    /* Debug first 5 detections */
                    if (num_faces < 5) {
                        printf("  Debug det[%d]: grid[%d,%d] anc=%d raw: tx=%d ty=%d tw=%d th=%d\n",
                               num_faces, x, y, a,
                               data[base_idx+0], data[base_idx+1],
                               data[base_idx+2], data[base_idx+3]);
                        printf("    dequant: tx=%.3f ty=%.3f tw=%.3f th=%.3f\n",
                               tx, ty, tw, th);
                        printf("    sigmoid(tx)=%.3f sigmoid(ty)=%.3f exp(tw)=%.3f exp(th)=%.3f\n",
                               sigmoid(tx), sigmoid(ty), expf(tw), expf(th));
                        printf("    anchor[%d]=[%.2f,%.2f] stride=%d\n", a, anchors[a][0], anchors[a][1], stride);
                        printf("    cx=%.1f cy=%.1f w=%.1f h=%.1f conf=%.3f\n", cx, cy, w, h, confidence);
                    }
                    faces[num_faces].x = cx;
                    faces[num_faces].y = cy;
                    faces[num_faces].w = w;
                    faces[num_faces].h = h;
                    faces[num_faces].confidence = confidence;
                    num_faces++;
                }
            }
        }
    }

    printf("  Debug: max objectness conf=%.4f (raw=%d) at grid[%d,%d] anchor=%d\n",
           max_obj_conf, max_obj_raw, max_obj_x, max_obj_y, max_obj_a);

    // Debug: print raw values for the max objectness cell
    {
        int base_idx = (max_obj_y * GRID_SIZE + max_obj_x) * 30 + max_obj_a * 6;
        printf("  Debug: best cell raw INT8: tx=%d ty=%d tw=%d th=%d obj=%d cls=%d\n",
               data[base_idx+0], data[base_idx+1], data[base_idx+2], data[base_idx+3],
               data[base_idx+4], data[base_idx+5]);
        printf("  Debug: best cell dequant: tx=%.3f ty=%.3f tw=%.3f th=%.3f\n",
               data[base_idx+0]*scale, data[base_idx+1]*scale,
               data[base_idx+2]*scale, data[base_idx+3]*scale);
    }

    printf("  Debug: num_faces before NMS = %d (threshold=%.2f)\n", num_faces, CONF_THRESHOLD);

    // Simple NMS
    for (int i = 0; i < num_faces; i++) {
        if (faces[i].confidence == 0) continue;
        for (int j = i + 1; j < num_faces; j++) {
            if (faces[j].confidence == 0) continue;
            if (iou(&faces[i], &faces[j]) > NMS_THRESHOLD) {
                if (faces[i].confidence > faces[j].confidence) {
                    faces[j].confidence = 0;
                } else {
                    faces[i].confidence = 0;
                    break;
                }
            }
        }
    }
    
    // Compact array
    int final_count = 0;
    for (int i = 0; i < num_faces; i++) {
        if (faces[i].confidence > 0) {
            if (i != final_count) {
                faces[final_count] = faces[i];
            }
            final_count++;
        }
    }
    
    return final_count;
}

// Preprocess image: resize to 416x416, normalize to INT8
static void preprocess_image(const uint8_t* src, int src_w, int src_h, int8_t* dst) {
    float scale_x = (float)src_w / INPUT_SIZE;
    float scale_y = (float)src_h / INPUT_SIZE;

    for (int y = 0; y < INPUT_SIZE; y++) {
        for (int x = 0; x < INPUT_SIZE; x++) {
            int src_x = (int)(x * scale_x);
            int src_y = (int)(y * scale_y);
            if (src_x >= src_w) src_x = src_w - 1;
            if (src_y >= src_h) src_y = src_h - 1;

            int src_idx = (src_y * src_w + src_x) * 3;
            int dst_idx = (y * INPUT_SIZE + x) * 3;

            // Normalize [0,255] -> [-128,127] (INT8 range)
            dst[dst_idx + 0] = (int8_t)(src[src_idx + 0] - 128);
            dst[dst_idx + 1] = (int8_t)(src[src_idx + 1] - 128);
            dst[dst_idx + 2] = (int8_t)(src[src_idx + 2] - 128);
        }
    }
}

// Draw a box on the image
static void draw_box(uint8_t* img, int w, int h, int x1, int y1, int x2, int y2,
                     uint8_t r, uint8_t g, uint8_t b) {
    // Clamp coordinates
    if (x1 < 0) x1 = 0; if (x1 >= w) x1 = w - 1;
    if (x2 < 0) x2 = 0; if (x2 >= w) x2 = w - 1;
    if (y1 < 0) y1 = 0; if (y1 >= h) y1 = h - 1;
    if (y2 < 0) y2 = 0; if (y2 >= h) y2 = h - 1;

    // Draw horizontal lines
    for (int x = x1; x <= x2; x++) {
        int idx1 = (y1 * w + x) * 3;
        int idx2 = (y2 * w + x) * 3;
        img[idx1] = r; img[idx1+1] = g; img[idx1+2] = b;
        img[idx2] = r; img[idx2+1] = g; img[idx2+2] = b;
    }
    // Draw vertical lines
    for (int y = y1; y <= y2; y++) {
        int idx1 = (y * w + x1) * 3;
        int idx2 = (y * w + x2) * 3;
        img[idx1] = r; img[idx1+1] = g; img[idx1+2] = b;
        img[idx2] = r; img[idx2+1] = g; img[idx2+2] = b;
    }
}

int main(int argc, char* argv[]) {
    const char* model_path = "/opt/face_cnn.mars";
    const char* image_path = "/tmp/snapshot.jpg";
    const char* output_path = NULL;

    if (argc > 1) model_path = argv[1];
    if (argc > 2) image_path = argv[2];
    if (argc > 3) output_path = argv[3];

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Mars Face Detection                                     ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // Initialize NNA
    printf("Initializing NNA...\n");
    if (nna_init() != NNA_SUCCESS) {
        fprintf(stderr, "Failed to initialize NNA\n");
        return 1;
    }

    // Load model
    printf("Loading model: %s\n", model_path);
    mars_model_t* model = NULL;
    mars_error_t err = mars_load_file(model_path, &model);
    if (err != MARS_OK || !model) {
        fprintf(stderr, "Failed to load model: %s\n", mars_get_error_string(err));
        nna_deinit();
        return 1;
    }

    // Load image
    printf("Loading image: %s\n", image_path);
    int img_w, img_h, img_c;
    uint8_t* img_data = stbi_load(image_path, &img_w, &img_h, &img_c, 3);
    if (!img_data) {
        fprintf(stderr, "Failed to load image\n");
        mars_free(model);
        nna_deinit();
        return 1;
    }
    printf("  Image: %dx%d\n", img_w, img_h);

    // Get input tensor
    mars_runtime_tensor_t* input = mars_get_input(model, 0);
    if (!input || !input->vaddr) {
        fprintf(stderr, "Failed to get input tensor\n");
        stbi_image_free(img_data);
        mars_free(model);
        nna_deinit();
        return 1;
    }

    // Preprocess
    printf("Preprocessing...\n");
    preprocess_image(img_data, img_w, img_h, (int8_t*)input->vaddr);
    nna_cache_flush(input->vaddr, INPUT_SIZE * INPUT_SIZE * 3);

    // Run inference
    printf("Running inference...\n");
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    mars_error_t ret = mars_run(model);

    gettimeofday(&t2, NULL);
    float elapsed = (t2.tv_sec - t1.tv_sec) * 1000.0f + (t2.tv_usec - t1.tv_usec) / 1000.0f;
    printf("  Inference time: %.2f ms\n", elapsed);

    if (ret != MARS_OK) {
        fprintf(stderr, "Inference failed: %s\n", mars_get_error_string(ret));
        stbi_image_free(img_data);
        mars_free(model);
        nna_deinit();
        return 1;
    }

    // Get output and decode faces
    mars_runtime_tensor_t* output = mars_get_output(model, 0);
    if (!output) {
        fprintf(stderr, "Failed to get output tensor\n");
        stbi_image_free(img_data);
        mars_free(model);
        nna_deinit();
        return 1;
    }

    int is_float = (output->desc.dtype == MARS_DTYPE_FLOAT32);
    printf("  Output shape: [%d,%d,%d,%d] scale=%.6f dtype=%s\n",
           output->desc.shape[0], output->desc.shape[1],
           output->desc.shape[2], output->desc.shape[3], output->desc.scale,
           is_float ? "float32" : "int8");

    static FaceBox faces[100];
    int num_faces;
    if (is_float) {
        num_faces = decode_faces_float((const float*)output->vaddr, faces, 100);
    } else {
        num_faces = decode_faces_int8((const int8_t*)output->vaddr, output->desc.scale, faces, 100);
    }

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Faces detected: %-4d                                    ║\n", num_faces);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // Print and draw detections
    float scale_x = (float)img_w / INPUT_SIZE;
    float scale_y = (float)img_h / INPUT_SIZE;

    for (int i = 0; i < num_faces; i++) {
        int x1 = (int)((faces[i].x - faces[i].w/2) * scale_x);
        int y1 = (int)((faces[i].y - faces[i].h/2) * scale_y);
        int x2 = (int)((faces[i].x + faces[i].w/2) * scale_x);
        int y2 = (int)((faces[i].y + faces[i].h/2) * scale_y);

        printf("  Face %d: [%d,%d,%d,%d] conf=%.2f\n", i, x1, y1, x2, y2, faces[i].confidence);

        // Draw green box
        draw_box(img_data, img_w, img_h, x1, y1, x2, y2, 0, 255, 0);
    }

    // Save output image
    if (output_path) {
        printf("\nSaving output: %s\n", output_path);
        stbi_write_jpg(output_path, img_w, img_h, 3, img_data, 90);
    }

    // Cleanup
    stbi_image_free(img_data);
    mars_free(model);
    nna_deinit();

    printf("\nDone.\n");
    return 0;
}

