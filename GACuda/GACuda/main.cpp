#include "StdAfx.h"

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>

#include "framework.h"

#define REFRESH_DELAY     10 //ms

// constants / global variables
char* window_title = "Genetic Cuda";
unsigned int window_width = 512;
unsigned int window_height = 256;
int iGLUTWindowHandle = 0;          // handle to the GLUT window

static int fpsCount = 0;
static int fpsLimit = 1;
static unsigned int framecount = 0;
StopWatchInterface *timer = NULL;

int   *pArgc = NULL;
char **pArgv = NULL;

//program declaration
framework* program;

// Forward declarations
void runStdProgram(int argc, char **argv);
void FreeResource();
void Cleanup(int iExitCode);

// GL functionality
bool initCUDA(int argc, char **argv);
bool initGL(int *argc, char **argv);

// rendering callbacks
void display();
void idle();
void keyboard(unsigned char key, int x, int y);
void reshape(int w, int h);
void mainMenu(int i);

void display()
{
	sdkStartTimer(&timer);

	program->process();
	program->renderScene();
    cudaDeviceSynchronize();

	sdkStopTimer(&timer);

    // flip backbuffer
    glutSwapBuffers();
	framecount++;

	// Update fps counter, fps/title display and log
	if (++fpsCount == fpsLimit)
	{
		char cTitle[256];
		float t = sdkGetAverageTimerValue(&timer);
		float fps = 1000.0f / t;
		sprintf(cTitle, "%s: %.1f fps %u frames", window_title, fps, framecount);
		glutSetWindowTitle(cTitle);
		fpsCount = 0;
		fpsLimit = 1;// (int)((fps > 1.0f) ? fps : 1.0f);
		sdkResetTimer(&timer);
	}
}

void timerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
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


int main(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

    pArgc = &argc;
    pArgv = argv;

	runStdProgram(argc, argv);

    exit(EXIT_SUCCESS);
}

void FreeResource()
{
	sdkDeleteTimer(&timer);

	delete program;
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

void runStdProgram(int argc, char **argv)
{
    if (false == initGL(&argc, argv))
    {
        return;
    }

	char *ref_file = NULL;
	//Use later for file loading
	if (checkCmdLineFlag(argc, (const char **)argv, "file"))
	{

		getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
	}

    // Initialize CUDA context
    initCUDA(argc, argv);

	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);

	// Initialize TinyImageLoader
	til::TIL_Init();

    // Register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // create menu
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    printf("\n"
           "Controls\n"
           "(right click mouse button for Menu)\n"
           "[esc] - Quit\n\n"
          );

    // start rendering mainloop
	program = new framework(ref_file);
    glutMainLoop();

    // Normally unused return path
    Cleanup(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize CUDA context
////////////////////////////////////////////////////////////////////////////////
bool initCUDA(int argc, char **argv)
{
	findCudaGLDevice(argc, (const char **)argv);
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
    iGLUTWindowHandle = glutCreateWindow(window_title);

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
