
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
constexpr int max_iter{ 1024 };
constexpr float zoom_speed{ 2.0f };

GLuint bufferObj;
cudaGraphicsResource *resource;
uchar4* dev_resource;

thrust::complex<double> *origin;
double *scale;

thrust::complex<double> *dev_origin;
double *dev_scale;

double scale_d{ 1.0f };
int mouse_pos[2]{ 0, 0 };

bool update_required{ false };
bool initial_draw{ true };

__global__ void compute_iter_kernel(uchar4* pixel, thrust::complex<double>* origin, double* scale) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    thrust::complex<double> c((*scale) * (x - window_size / 2) / window_size + origin->real(),
                              (*scale) * (y - window_size / 2) / window_size + origin->imag());
    thrust::complex<double> z(0.0f, 0.0f);

    int i{ 0 };
    while (((z.real() * z.real() + z.imag() * z.imag()) < 4) && (i < max_iter)) {
        z = z * z + c;
        ++i;
    }

    if (max_iter == i) {
        pixel[offset].x = 0x0;
        pixel[offset].y = 0x0;
        pixel[offset].z = 0x0;
        pixel[offset].w = 0xff;
    }
    else {
        int value = abs((i % 511) - 255);
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

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR(cudaEventRecord(start, 0));

    dim3 grids(window_size / 16, window_size / 16);
    dim3 threads(16, 16);

    compute_iter_kernel<<<grids, threads>>>(dev_resource, dev_origin, dev_scale);

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

    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_origin, origin, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_scale, scale, 0));

    (*origin) = std::move(thrust::complex<double>(0.0f, 0.0f));
    (*scale) = 4.0f;

    glutKeyboardFunc(key_func);
    glutDisplayFunc(draw_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutIdleFunc(idle_func);
    glutReshapeFunc(resize_func);

    glutMainLoop();
}