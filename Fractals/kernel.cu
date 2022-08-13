﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GL_GLEXT_PROTOTYPES

#include "GL/glew.h"
#include "GL/glut.h"

#include "cuda_gl_interop.h"

#include <stdio.h>
#include <math.h>
#include <thrust/complex.h>

#include "Utils.h"

constexpr int window_size{ 768 };
constexpr int render_steps{ 1 }; // (1 << render_steps) must divide window_size
constexpr int max_iter{ 4096 };
constexpr float zoom_speed{ 2.0f };

GLuint bufferObj;
cudaGraphicsResource* resource;
uchar4* dev_resource;

thrust::complex<double> *origin;
double *scale;
int* strides;

double scale_d{ 1.0f };
int mouse_pos[2]{ 0, 0 };

int *dev_iter;                          // int[window_size][window_size]
thrust::complex<double> *dev_origin;    // complex
int *dev_strides;                       // int
double *dev_scale;                      // double
bool *dev_mask;                         // bool[window_size][window_size]

bool update_required{ false };
bool initial_draw{ true };

__global__ void compute_iter_kernel(int* iter, bool* mask, thrust::complex<double>* origin, double* scale, int* strides) {
    // map from threadIdx/BlockIdx to pixel position
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * (*strides);
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * (*strides);
    int offset = x + y * window_size;

    if (!mask[offset]) {
        // already computed
        if (-1 == iter[offset]) {
            // need to be copied
            int x_closest = x - (x % ((*strides) << 1));
            int y_closest = y - (y % ((*strides) << 1));
            int offset_closest = x_closest + y_closest * window_size;
            
            iter[offset] = iter[offset_closest];
            mask[offset] = false;
        }
        return;
    }

    thrust::complex<double> c((*scale) * (x - window_size / 2) / window_size + origin->real(),
                              (*scale) * (y - window_size / 2) / window_size + origin->imag());
    thrust::complex<double> z(0.0f, 0.0f);

    int i{ 0 };
    while (((z.real() * z.real() + z.imag() * z.imag()) < 4) && (i < max_iter)) {
        z = z * z + c;
        ++i;
    }
    iter[offset] = i;
    mask[offset] = false;
}

__global__ void compute_mask_kernel(bool* mask, int* iter, int* strides) {
    // map from threadIdx/BlockIdx to pixel position
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * (*strides);
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * (*strides);
    int offset = x + y * window_size;

    if ((x < (*strides)) ||
        (y < (*strides)) ||
        (x >= (window_size - (*strides))) ||
        (y >= (window_size - (*strides)))) {
        // always recompute on border
        return;
    }
    else {
        // if any neighbour pixel is different, we need to recompute
        bool need_to_recompute{ false };
        for (int dx{ -(*strides) }; dx <= (*strides); dx += (*strides)) {
            for (int dy{ -(*strides) }; dy <= (*strides); dy += (*strides)) {
                if ((0 != dx) || (0 != dy)) {
                    int offset_neighbour = (x + dx) + (y + dy) * window_size;
                    if (iter[offset] != iter[offset_neighbour]) {
                        need_to_recompute = true;
                    }
                }
            }
        }
        if (!need_to_recompute) {
            int next_strides = (*strides) >> 1;
            mask[offset] = false;
            mask[offset + next_strides] = false;
            mask[offset + next_strides * window_size] = false;
            mask[offset + next_strides + next_strides * window_size] = false;
            return;
        }
    }
}

__global__ void iter_to_pixel_kernel(uchar4* pixel, int* iter) {
    // map from threadIdx/BlockIdx to pixel position
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int offset = x + y * blockDim.x * gridDim.x;

    if (max_iter == iter[offset]) {
        pixel[offset].x = 0x0;
        pixel[offset].y = 0x0;
        pixel[offset].z = 0x0;
        pixel[offset].w = 0xff;
    }
    else {
        int value = abs((iter[offset] % 511) - 255);
        pixel[offset].x = value;
        pixel[offset].y = value;
        pixel[offset].z = 0xff;
        pixel[offset].w = 0xff;
    }
}

static void cuda_compute_fractal(void) {
    size_t size;

    HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));

    HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&dev_resource, &size, resource));

    cudaEvent_t start, stop;

    HANDLE_ERROR(cudaMemset(dev_iter, -1, sizeof(int) * window_size * window_size));
    HANDLE_ERROR(cudaMemset(dev_mask, true, sizeof(bool) * window_size * window_size));

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR(cudaEventRecord(start, 0));

    for (int i{ render_steps }; i > 0; --i) {
        (*strides) = 1 << (i - 1);

        dim3 grids((window_size / (*strides)) / 16, (window_size / (*strides)) / 16);
        dim3 threads(16, 16);

        compute_iter_kernel<<<grids, threads>>>(dev_iter, dev_mask, dev_origin, dev_scale, dev_strides);

        if (i > 1) {
            compute_mask_kernel<<<grids, threads>>>(dev_mask, dev_iter, dev_strides);
        }

    }
    dim3 grids(window_size / 16, window_size / 16);
    dim3 threads(16, 16);

    iter_to_pixel_kernel<<<grids, threads>>>(dev_resource, dev_iter);

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("\r%3.0f fps", 1000/elapsedTime);

    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));
}

static void draw_func(void) {
    glDrawPixels(window_size, window_size, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glutSwapBuffers();
}

static void key_func(unsigned char key, int x, int y) {
    switch (key) {
    case 27:
        // clean up OpenGL and CUDA

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glDeleteBuffers(1, &bufferObj);

        HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));

        HANDLE_ERROR(cudaFreeHost(origin));
        HANDLE_ERROR(cudaFreeHost(scale));
        HANDLE_ERROR(cudaFreeHost(strides));

        HANDLE_ERROR(cudaFree(dev_iter));
        HANDLE_ERROR(cudaFree(dev_mask));
        exit(0);
    }
}

static void mouse_func(int button, int state, int x, int y) {
    mouse_pos[0] = x;
    mouse_pos[1] = y;
    switch (state) {
    case GLUT_DOWN:
        switch (button) {
        case GLUT_RIGHT_BUTTON:
            scale_d = zoom_speed;
            update_required = true;
            break;
        case GLUT_LEFT_BUTTON:
            scale_d = 1.0f / zoom_speed;
            update_required = true;
            break;
        }
        break;
    case GLUT_UP:
        scale_d = 1.0f;
        update_required = false;
        break;
    }
}

static void motion_func(int x, int y) {
    mouse_pos[0] = x;
    mouse_pos[1] = y;
}

static void idle_func(void) {
    static double time;
    static double delta_time;

    delta_time = (glutGet(GLUT_ELAPSED_TIME) - time) / 1000.0f;
    time = glutGet(GLUT_ELAPSED_TIME);

    if (update_required || initial_draw) {
        thrust::complex<double> mouse_complex((*scale) * (mouse_pos[0] - window_size / 2) / window_size + origin->real(),
                                              (*scale) * ((window_size - mouse_pos[1]) - window_size / 2) / window_size + origin->imag());

        double s = pow(scale_d, delta_time);

        (*origin) = (*origin) + ((*origin) - mouse_complex) * (s - 1); // self.__mid + (self.__mid - point)*(1 - factor)/factor
        (*scale) *= s;

        cuda_compute_fractal();
        glutPostRedisplay();

        initial_draw = false;
    }
}

static void resize_func(int width, int height) {
    glutReshapeWindow(window_size, window_size);
}

int main(int argc, char** argv)
{
    cudaDeviceProp prop;
    int dev;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(window_size, window_size);
    glutCreateWindow("Fractal");

    glewInit();
    glGenBuffers(1, &bufferObj);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, window_size * window_size * 4, NULL, GL_DYNAMIC_DRAW_ARB);

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.pciBusID = 38;

    HANDLE_ERROR(cudaChooseDevice(&dev, &prop));

    HANDLE_ERROR(cudaGLSetGLDevice(dev));

    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsRegisterFlagsWriteDiscard));

    HANDLE_ERROR(cudaHostAlloc((void**)&origin, sizeof(thrust::complex<double>), cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc((void**)&scale, sizeof(double), cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc((void**)&strides, sizeof(int), cudaHostAllocWriteCombined | cudaHostAllocMapped));

    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_origin, origin, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_scale, scale, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_strides, strides, 0));

    (*origin) = std::move(thrust::complex<double>(0.0f, 0.0f));
    (*scale) = 4.0f;

    HANDLE_ERROR(cudaMalloc((void**)&dev_iter, sizeof(int) * window_size * window_size));
    HANDLE_ERROR(cudaMalloc((void**)&dev_mask, sizeof(bool) * window_size * window_size));

    glutKeyboardFunc(key_func);
    glutDisplayFunc(draw_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutIdleFunc(idle_func);
    glutReshapeFunc(resize_func);

    glutMainLoop();
}