
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GL_GLEXT_PROTOTYPES

#include "GL/glew.h"
#include "GL/glut.h"

#include "cuda_gl_interop.h" 

#include <stdio.h>
#include <math.h>

constexpr int window_size = 768;

GLuint bufferObj;
cudaGraphicsResource* resource;

__global__ void kernel(uchar4* ptr) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    // now calculate the value at that position
    float c_re = 4 * static_cast<float>(x - window_size / 2) / window_size;
    float c_im = 4 * static_cast<float>(y - window_size / 2) / window_size;
    float z_re = 0.0f;
    float z_im = 0.0f;
    float tmp = 0.0f;
    int i = 0;
    while ((z_re * z_re + z_im * z_im) < 2 && i < 512) {
        tmp = z_re * z_re - z_im * z_im;
        z_im = 2 * z_re * z_im;
        z_re = tmp;
        z_re += c_re;
        z_im += c_im;
        ++i;
    }
    // accessing uchar4 vs. unsigned char*
    ptr[offset].x = 255 * (i != 512);
    ptr[offset].y = (3*i) % 256;
    ptr[offset].z = (5*i) % 256;
    ptr[offset].w = 255;
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

        cudaError_t cudaStatus = cudaGraphicsUnregisterResource(resource);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphicsUnregisterResource failed!");
            exit(1);
        }
        exit(0);
    }
}

int main(int argc, char** argv)
{
    cudaError_t cudaStatus;
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

    cudaStatus = cudaChooseDevice(&dev, &prop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaChooseDevice failed!");
        return 1;
    }

    cudaStatus = cudaGLSetGLDevice(dev);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGLSetGLDevice failed!");
        return 1;
    }

    cudaStatus = cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsRegisterFlagsNone);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed!");
        return 1;
    }

    uchar4* devPtr;
    size_t size;

    cudaStatus = cudaGraphicsMapResources(1, &resource, NULL);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsMapResources failed!");
        return 1;
    }

    cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsResourceGetMappedPointer failed!");
        return 1;
    }

    dim3 grids(window_size / 16, window_size / 16);
    dim3 threads(16, 16);

    kernel<<<grids, threads>>>(devPtr);

    cudaStatus = cudaGraphicsUnmapResources(1, &resource, NULL);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsUnmapResources failed!");
        return 1;
    }


    glutKeyboardFunc(key_func);
    glutDisplayFunc(draw_func);
    glutMainLoop();
}