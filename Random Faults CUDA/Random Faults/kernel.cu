/*
/*
 Random Faults in CUDA
 (C) Bedrich Benes 2020
 bbenes@purdue.edu
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <math.h>
#include <vector>			//Standard template library class
#include <GL/freeglut.h>

//in house created libraries
#include "vect3d.h"
#include "trackball.h"
#include "helper.h"         


//#pragma comment(lib, "freeglut.lib")

TrackBallC trackball;
bool mouseLeft, mouseMid, mouseRight;

GLint wWindow = 1200;
GLint hWindow = 800;


#define DEBUG
const int MAX = 256;
const int SCENE = 1;
const int maxSteps = 256;

GLint n = 1;
GLfloat a[MAX][MAX];
GLint fill = 1;
#define ELEV 0.0005f

//CUDA stuff
float *d_A;

void Cleanup(bool noError)
{
	cudaError_t error;
	// Free device memory
	if (d_A) error = cudaFree(d_A);
	if (!noError || error != cudaSuccess) printf("Something failed \n");
}

void checkCUDAError(const char* msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void Idle(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //clear all
	glEnable(GL_LIGHT0);
	trackball.Set3DViewCamera();
	glDisable(GL_LIGHTING);
	CoordSyst();
	glEnable(GL_LIGHTING);
	glCallList(SCENE);
	glutSwapBuffers();
}


Vect3d CrossProduct(Vect3d *a, Vect3d *b, Vect3d *cross)
{
	cross->SetX(a->GetY() * b->GetZ() - a->GetZ()* b->GetY());
	cross->SetY(a->GetZ() * b->GetX() - a->GetX()* b->GetZ());
	cross->SetZ(a->GetX() * b->GetY() - a->GetY()* b->GetX());
	return *cross;
}

void Display(void)
{
	int i, j;
	Vect3d v1, v2, v3, v13, v12, n;
	GLfloat materialColor[] = { 0.1f, 0.5f, 0.02f, 1.0f };
	GLfloat materialSpecular[] = { 0,0,0,1 };
	glNewList(SCENE, GL_COMPILE);
	glShadeModel(GL_SMOOTH);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, materialColor);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, materialSpecular);
	if (fill) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	else glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		/*glColor3d(0, 0, 1);
		glutSolidSphere(1.0, 100, 100);*/
	for (i = 0; i < MAX - 1; i++)
	{
		glBegin(GL_TRIANGLE_STRIP);
		for (j = 0; j < MAX-1; j++)
		{
//triangle 1
			v1.Set(i / (GLfloat)MAX, j / (GLfloat)MAX, a[i][j]);
			v2.Set((i + 1) / (GLfloat)MAX, j / (GLfloat)MAX, a[i + 1][j]);
			v3.Set((i + 1) / (GLfloat)MAX, (j + 1) / (GLfloat)MAX, a[i + 1][j+1]);
			v12.Set(v1.GetX() - v3.GetX(), v1.GetY() - v3.GetY(), v1.GetZ() - v3.GetZ());
			v13.Set(v1.GetX() - v2.GetX(), v1.GetY() - v2.GetY(), v1.GetZ() - v2.GetZ());
			n.Set(CrossProduct(&v13,&v12,&n));
			n.Normalize();
			glNormal3fv(n);
			glVertex3fv(v1);
			glVertex3fv(v2);
			glVertex3fv(v3);
//triangle 2
			v1.Set(i / (GLfloat)MAX, j / (GLfloat)MAX, a[i][j]);
			v2.Set((i + 1) / (GLfloat)MAX, (j + 1) / (GLfloat)MAX, a[i + 1][j+1]);
			v3.Set((i) / (GLfloat)MAX, (j + 1) / (GLfloat)MAX, a[i][j+1]);
			v12.Set(v1.GetX() - v3.GetX(), v1.GetY() - v3.GetY(), v1.GetZ() - v3.GetZ());
			v13.Set(v1.GetX() - v2.GetX(), v1.GetY() - v2.GetY(), v1.GetZ() - v2.GetZ());
			n.Set(CrossProduct(&v13, &v12, &n));
			n.Normalize();
			glNormal3fv(n);
			glVertex3fv(v1);
			glVertex3fv(v2);
			glVertex3fv(v3);
		}
		glEnd();
	}
	glEndList();
}

void DisplayUgly(void)
{
	int i, j;


	glNewList(SCENE, GL_COMPILE);
	glColor3ub(0, 0, 0);
	if (fill) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	else glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	for (i = 0; i < MAX - 1; i++)
	{
		
		glBegin(GL_QUAD_STRIP);
		for (j = 0; j < MAX; j++)
		{
			glColor3f(a[i][j], a[i][j], a[i][j]);
			glVertex3f(i / (GLfloat)MAX, j / (GLfloat)MAX, a[i][j]);
			glVertex3f((i + 1) / (GLfloat)MAX, j / (GLfloat)MAX, a[i + 1][j]);
		}
		glEnd();
	}
	glEndList();
}

void Init(void)
{
	int i, j;

	glClearColor(1.0, 1.0, 1.0, 1.0);
	glClearDepth(1000.f);
	glEnable(GL_DEPTH_TEST);
	for (i = 0; i < MAX; i++)
		for (j = 0; j < MAX; j++)  a[i][j] = 0.5;

}

void myReshape(int w, int h)
{
	glViewport(0, 0, w, h);
	wWindow = w;
	hWindow = h;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-0.2, 1.2, -0.2, 1.2, -10, 10);
}

//random fault CPU===============================================

int ranGen(int max) {//from 0 to MAX
	return rand() % max;
}

Vect3d randGenPoint() {
	return Vect3d((float)ranGen(MAX) / MAX, (float)ranGen(MAX) / MAX, 0.5);
}

float testImplctPln(Vect3d p_known, Vect3d randNormal, Vect3d p_test) {

	return randNormal.GetX() * (p_known.GetX() - p_test.GetX()) +
		randNormal.GetY() * (p_known.GetY() - p_test.GetY()) +
		randNormal.GetZ() * (p_known.GetZ() - p_test.GetZ());
}

void RandomFault(void)
{

	Vect3d ranP1, ranP2, normal_knwnPln, normal_RanPln;

	ranP1 = randGenPoint();
	ranP2 = randGenPoint();
	normal_knwnPln.Set(0.0, 0.0, 1.0);
	normal_RanPln.Set(Vect3d::Cross((ranP1 - ranP2), normal_knwnPln));

	for (int i = 0; i < MAX; i++) {
		for (int j = 0; j < MAX; j++) {

			Vect3d testP((float)i/MAX, (float)j/MAX, 0.5);
			if (testImplctPln(ranP1, normal_RanPln, testP) > 0) {
				a[i][j] += 0.001;
			}
			else a[i][j] -= 0.001;

		}
	}
}

//random fault GPU ====================================================
	__constant__ float d_const_ranP1[3];
	__constant__ float d_const_ranP2[3];
	__constant__ float d_const_nomalGivenPln[3];

	__device__ void crossProd(float *v1, float *v2, float *output) {

		output[0] = v1[1] * v2[2] - v1[2] * v2[1];
		output[1] = v1[2] * v2[0] - v1[0] * v2[2];
		output[2] = v1[0] * v2[1] - v1[1] * v2[0];
	}

	__device__ float testImplct(float* testP) {

		float vec_P1toP2[3] = { d_const_ranP2[0] - d_const_ranP1[0], d_const_ranP2[1] - d_const_ranP1[1], d_const_ranP2[2] - d_const_ranP1[2] };
		
		float normal_randPlane[3];

		crossProd(vec_P1toP2, d_const_nomalGivenPln, normal_randPlane);

		//test const pass
		//printf("test: normal_randPlane = { %f, %f, %f} \n", normal_randPlane[0], normal_randPlane[1], normal_randPlane[2]);

		return normal_randPlane[0] * (d_const_ranP1[0] - testP[0]) +
			normal_randPlane[1] * (d_const_ranP1[1] - testP[1]) +
			normal_randPlane[2] * (d_const_ranP1[2] - testP[2]);
	}

	__global__ void RandFaultKernel(float a[MAX][MAX],  const int N ) //number of steps to run
	//__global__ void RandFaultKernel(float a[MAX], const int N ) //number of steps to run
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockDim.y * blockIdx.y + threadIdx.y;
		if ((i >= N) || (j >= N)) return;//let threads out of N not to compute anything

		float point[3] = { (float)i / MAX, (float)j / MAX, 0.5 };

		if (testImplct(point) > 0) a[i][j] += 0.003;
		else a[i][j] -= 0.003;

	}

	//pass random point to constant memory
	//no need to free
void prepareConstMem() {
	cudaError_t err;

	float h_randP1[3] = { (float)ranGen(MAX)/MAX, (float)ranGen(MAX) / MAX , 0.5};
	float h_randP2[3] = { (float)ranGen(MAX)/MAX, (float)ranGen(MAX) / MAX , 0.5};
	float h_normalGivenPlan[3] = {0.0, 0.0, 1.0};

	err = cudaMemcpyToSymbol(d_const_ranP1, h_randP1, sizeof(float) * 3);
	if (err != cudaSuccess) {
		std::cout << "cuda constant memory copy fail" << std::endl;
		Cleanup(false);
	}

	err = cudaMemcpyToSymbol(d_const_ranP2, h_randP2, sizeof(float) * 3);
	if (err != cudaSuccess) {
		std::cout << "cuda constant memory copy fail" << std::endl;
		Cleanup(false);
	}

	err = cudaMemcpyToSymbol(d_const_nomalGivenPln, h_normalGivenPlan, sizeof(float) * 3);
	if (err != cudaSuccess) {
		std::cout << "cuda constant memory copy fail" << std::endl;
		Cleanup(false);
	}

}

void RandomFaultsCuda()
{

	cudaError_t error;
	const int sizeArray = sizeof(float) * MAX * MAX; 

	//allocate array on the device
	//sizeArray = sizeof(float) * MAX * MAX; //2D array of floats
	error = cudaMalloc((void**)&d_A, sizeArray);
	//Copy the 2D array from host memory to device memory
	error = cudaMemcpy(d_A, a, sizeArray, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) Cleanup(false);

	//prepare constant memory
	prepareConstMem();

	//prepare blocks and grid
	const int BLOCKSIZE = 16;
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(ceil((float)MAX / dimBlock.x), ceil((float)MAX / dimBlock.y));
	// Invoke kernel
	RandFaultKernel << <dimGrid, dimBlock >> > ((float(*)[MAX])d_A, MAX);
	//RandFaultKernel << <dimGrid, dimBlock >> > (&d_A, MAX);
	error = cudaGetLastError();
	if (error != cudaSuccess) printf("Something went wrong: %i\n", error);
	error = cudaThreadSynchronize();
	if (error != cudaSuccess) { printf("synchronization is wrong\n"); Cleanup(false); }
	// Copy result from device memory to host memory
	error = cudaMemcpy(a, d_A, sizeArray, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) { printf("could not copy from device\n"); Cleanup(false); }
	Cleanup(true);
}

void Key(unsigned char key, GLint i, GLint j)
{
	switch (key)
	{
	case 'f':
	case 'F': fill = (fill == 0); glutPostRedisplay(); break;
	case ' ': //run CPU implementation
	{

		long t1 = clock();
		RandomFault();
		//for (int i = 0; i < maxSteps; i++)
		//{
		//	char name[200];
		//	sprintf(name, "%i%% done\r", 100 * (i + 1) / maxSteps);
		//	glutSetWindowTitle(name);
		//}
		long t2 = clock();
		glutSetWindowTitle("Random Faults in Cuda");
		printf("CPU Running time: %i\n", t2 - t1);
		break;
	}

	case 'c': //run CUDA implementation
	{
		glutSetWindowTitle("Running CUDA");
		long t1 = clock();
		RandomFaultsCuda();
		long t2 = clock();
		glutSetWindowTitle("Random Faults in Cuda");
		printf("CUDA Running time: %i\n", t2 - t1);
		break;
	}
	case 27:
	case 'q':
	case 'Q': exit(0);
	}
	glutPostRedisplay();
}

void Mouse(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		trackball.Set(true, x, y);
		mouseLeft = true;
	}
	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{
		trackball.Set(false, x, y);
		mouseLeft = false;
	}
	if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN)
	{
		trackball.Set(true, x, y);
		mouseMid = true;
	}
	if (button == GLUT_MIDDLE_BUTTON && state == GLUT_UP)
	{
		trackball.Set(true, x, y);
		mouseMid = false;
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
	{
		trackball.Set(true, x, y);
		mouseRight = true;
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP)
	{
		trackball.Set(true, x, y);
		mouseRight = false;
	}
}

void MouseMotion(int x, int y) {
	if (mouseLeft)  trackball.Rotate(x, y);
	if (mouseMid)   trackball.Translate(x, y);
	if (mouseRight) trackball.Zoom(x, y);
//	glutPostRedisplay();
}


// Host code
int main(int argc, char** argv)
{
	std::srand(time(NULL));
	//srand(5);
	glutInitWindowSize(wWindow, hWindow);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutCreateWindow("Random Faults");
	Init();
	glutDisplayFunc(Display);
	//glutDisplayFunc(DisplayUgly);
	glutIdleFunc(Idle);
	glutKeyboardFunc(Key);
	glutReshapeFunc(myReshape);
	glutMouseFunc(Mouse);
	glutMotionFunc(MouseMotion);
	glutMainLoop();
	return 0;

}

