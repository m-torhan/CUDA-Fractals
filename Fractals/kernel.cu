
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

constexpr int window_size = 768;
constexpr int max_iter = 1024;
constexpr float zoom_speed = 2.0f;

GLuint bufferObj;
cudaGraphicsResource* resource;
uchar4* dev_resource;

thrust::complex<double> origin(0.0f, 0.0f);
double scale = 4.0f;

double scale_d = 1.0f;
int mouse_pos[2] = { 0, 0 };

thrust::complex<double> *dev_origin;
double *dev_scale;

__global__ void kernel(uchar4* ptr, thrust::complex<double> *origin, double* scale) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    // now calculate the value at that position

    thrust::complex<double> c((*scale) * (x - window_size / 2) / window_size + origin->real(),
                             (*scale) * (y - window_size / 2) / window_size + origin->imag());
    thrust::complex<double> z(0.0f, 0.0f);

    int i = 0;
    while ((z.real() * z.real() + z.imag() * z.imag()) < 4 && i < max_iter) {
        z = z * z + c;
        ++i;
    }

    if (max_iter == i) {
        ptr[offset].x = 0x0;
        ptr[offset].y = 0x0;
        ptr[offset].z = 0x0;
        ptr[offset].w = 0xff;
    }
    else {
        int value = abs((i % 511) - 255);
        ptr[offset].x = value;
        ptr[offset].y = value;
        ptr[offset].z = 0xff;
        ptr[offset].w = 0xff;
    }
}

static void cuda_compute_fractal(void) {
    size_t size;

    HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));

    HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&dev_resource, &size, resource));

    dim3 grids(window_size / 16, window_size / 16);
    dim3 threads(16, 16);

    cudaEvent_t start, stop;

    HANDLE_ERROR(cudaMemcpy(dev_origin, &origin, sizeof(thrust::complex<double>), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_scale, &scale, sizeof(double), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR(cudaEventRecord(start, 0));

    kernel<<<grids, threads>>>(dev_resource, dev_origin, dev_scale);

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate: %3.1f ms. Scale: %f\n", elapsedTime, scale);

    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));
}

static void draw_func(void) {
    cuda_compute_fractal();

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

        HANDLE_ERROR(cudaFree(dev_origin));
        HANDLE_ERROR(cudaFree(dev_scale));
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
            break;
        case GLUT_LEFT_BUTTON:
            scale_d = 1.0f / zoom_speed;
            break;
        }
        break;
    case GLUT_UP:
        scale_d = 1.0f;
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

    if (scale_d != 1.0f) {
        thrust::complex<double> mouse_complex(scale * (mouse_pos[0] - window_size / 2) / window_size + origin.real(),
                                             scale * ((window_size - mouse_pos[1]) - window_size / 2) / window_size + origin.imag());

        double s = powf(scale_d, delta_time);

        origin = origin + (origin - mouse_complex) * (s - 1); // self.__mid + (self.__mid - point)*(1 - factor)/factor
        scale *= s;
        printf("\r%f %f %f %d %d\n", scale, scale_d, delta_time, mouse_pos[0], mouse_pos[1]);

        glutPostRedisplay();
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

    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsRegisterFlagsNone));

    HANDLE_ERROR(cudaMalloc((void**)&dev_origin, sizeof(thrust::complex<double>)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_scale, sizeof(double)));

    glutKeyboardFunc(key_func);
    glutDisplayFunc(draw_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutIdleFunc(idle_func);
    glutReshapeFunc(resize_func);

    glutMainLoop();
}