#include "StdAfx.h"

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>
#include <rendercheck_gl.h>

#include "tools.h"

// Shared Library Test Functions
#define MAX_EPSILON 10
#define REFRESH_DELAY     10 //ms

const char *sSDKname = "postProcessGL";

unsigned int g_TotalErrors = 0;

// CheckFBO/BackBuffer class objects
CheckRender *g_CheckRender = NULL;

////////////////////////////////////////////////////////////////////////////////
// constants / global variables
unsigned int window_width = 512;
unsigned int window_height = 512;
unsigned int image_width = 512;
unsigned int image_height = 512;
int iGLUTWindowHandle = 0;          // handle to the GLUT window
GLuint textureId = 0;

bool enable_cuda     = true;

int   *pArgc = NULL;
char **pArgv = NULL;

// Forward declarations
void runStdProgram(int argc, char **argv);
void FreeResource();
void Cleanup(int iExitCode);

// GL functionality
bool initCUDA(int argc, char **argv, bool bUseGL);
bool initGL(int *argc, char **argv);

// rendering callbacks
void display();
void idle();
void keyboard(unsigned char key, int x, int y);
void reshape(int w, int h);
void mainMenu(int i);

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void process(int width, int height, int radius)
{

}

////////////////////////////////////////////////////////////////////////////////
//! render a simple 3D scene
////////////////////////////////////////////////////////////////////////////////
void renderScene(bool colorScale)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

	//Draw here

	glBindTexture(GL_TEXTURE_2D, textureId);

	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex3f(0, 0, 0);
	glTexCoord2f(0, 1); glVertex3f(0, 256, 0);
	glTexCoord2f(1, 1); glVertex3f(256, 256, 0);
	glTexCoord2f(1, 0); glVertex3f(256, 0, 0);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	renderScene(false);

    cudaDeviceSynchronize(); // ?

    // flip backbuffer
    glutSwapBuffers();
}

void timerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void
keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            Cleanup(EXIT_SUCCESS);
            break;
    }
}

void reshape(int w, int h)
{
    window_width = w;
    window_height = h;
}

void mainMenu(int i)
{
    keyboard((unsigned char) i, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

	//Use later for file loading
	/*if (checkCmdLineFlag(argc, (const char **)argv, "radius") &&
		checkCmdLineFlag(argc, (const char **)argv, "file"))
	{

		getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
		blur_radius = getCmdLineArgumentInt(argc, (const char **)argv, "radius");
	}*/

    pArgc = &argc;
    pArgv = argv;
	runStdProgram(argc, argv);

    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void FreeResource()
{
    cudaDeviceReset();
	til::TIL_ShutDown();

    if (iGLUTWindowHandle)
    {
        glutDestroyWindow(iGLUTWindowHandle);
    }

    // finalize logs and leave
    printf("Exiting...\n");
}

void Cleanup(int iExitCode)
{
    FreeResource();
    printf("%s\n", (iExitCode == EXIT_SUCCESS) ? "PASSED" : "FAILED");
    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Run standard demo loop with or without GL verification
////////////////////////////////////////////////////////////////////////////////
void runStdProgram(int argc, char **argv)
{
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (false == initGL(&argc, argv))
    {
        return;
    }

    // Now initialize CUDA context (GL context has been created already)
    initCUDA(argc, argv, true);

	//Initialize TinyImageLoader
	til::TIL_Init();

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // create menu
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    printf("\n"
           "\tControls\n"
           "\t(right click mouse button for Menu)\n"
           "\t[esc] - Quit\n\n"
          );

	//Load test program
	textureId = tools::loadTexture("./assets/test.png");

    // start rendering mainloop
    glutMainLoop();

    // Normally unused return path
    Cleanup(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize CUDA context
////////////////////////////////////////////////////////////////////////////////
bool initCUDA(int argc, char **argv, bool bUseGL)
{
    if (bUseGL)
    {
        findCudaGLDevice(argc, (const char **)argv);
    }
    else
    {
        findCudaDevice(argc, (const char **)argv);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    iGLUTWindowHandle = glutCreateWindow("Genetic Cuda");

    // initialize necessary OpenGL extensions
    glewInit();

    if (! glewIsSupported(
            "GL_VERSION_2_0 "
            "GL_ARB_pixel_buffer_object "
            "GL_EXT_framebuffer_object "
        ))
    {
        printf("ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.5, 0.5, 0.5, 1.0);

    glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glEnable(GL_TEXTURE_2D);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	glOrtho(0.0, window_width, window_height, 0.0, -1.0, 1.0);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    SDK_CHECK_ERROR_GL();

    return true;
}
