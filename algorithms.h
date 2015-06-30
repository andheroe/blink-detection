/*! \file algorithms.h
    \brief Computer vision algorithms for eyes tracking and image computing.

    This file containe original computer vision algorithms for eyes tracking and image computing
    on the base of OpenCV.
*/

#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include <iostream>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

/*! \var CvHaarClassifierCascade* face_cascade
    \brief Contains the haar cascade to detect the face.
*/
extern CvHaarClassifierCascade *face_cascade;

/*! \var CvHaarClassifierCascade* left_eye_cascade
    \brief Contains the haar cascade to detect the left eye.
*/
extern CvHaarClassifierCascade *left_eye_cascade;

/*! \var CvHaarClassifierCascade* right_eye_cascade
    \brief Contains the haar cascade to detect the right eye.
*/
extern CvHaarClassifierCascade *right_eye_cascade;


/*! \enum OBJECT
    \brief Is used for specify, face or one of eyes will be used.
*/
enum OBJECT {
    FACE,
    LEFT_EYE,
    RIGHT_EYE
};

struct edge {
    int index;
    int square;

    edge(int i, int s) : index(i), square(s) { }

    bool operator<(const edge &str) const {
        return (square > str.square);
    }
};


/*! \fn void DrawRect(CvRect rect, int r, int g, int b, IplImage* &frame)
    \brief Draws a rectangle on an image.
    \param rect Rectangle to draw.
    \param r Red component of the lines color.
    \param g Green component of the lines color.
    \param b Blue component of the lines color.
    \param frame Image to draw on.
*/
void DrawRect(CvRect rect, int r, int g, int b, IplImage *&frame);

void DrawRhombus(int* params, int r, int g, int b, int height, int width, IplImage* &frame);

/*! \fn void InitCascades()
    \brief Initialize haar cascades.
*/
void InitCascades();

/*! \fn CvRect Find(OBJECT obj,IplImage* frame,bool &status,CvRect* area=0)
    \brief Find an object on an image.
    \param obj Object to find.
    \param frame Image to draw on.
    \param status Parametr is set "true", when function ends succesfully, or "false" in other case.
    \param area Specify area to find on.
*/
CvRect Find(OBJECT obj, IplImage *frame, bool &status, CvRect *area = 0, float resize_koeff = 1);

/*! \fn CvRect ClarifyArea(OBJECT obj,CvRect* area)
    \brief Adapts face area to find one of the eyes.
    \param obj Object to find.
    \param frame Image to draw on.
    \param status Parametr is set "true", when function ends succesfully, or "false" in other case.
    \param area Specify area to find on.
*/
CvRect ClarifyArea(OBJECT obj, CvRect *area);

/*! \fn void ExpandArea(CvRect &area,float koeff)
    \brief Expands the area.
    \param area Area to expand.
    \param koeff Expanding coefficient.
*/
void ExpandArea(CvRect &area, float koeff);

/*! \fn bool EyesCorrect(CvRect left_eye, CvRect right_eye, IplImage *frame)
    \brief Returns "true" when found eyes seem to be correct, or "false" in other case.
    \param left_eye Left eye area.
    \param right_eye Right eye area.
    \param frame Image where eye were found.
*/
bool EyesCorrect(CvRect left_eye, CvRect right_eye, IplImage *frame);

float DistanceToMonitor(CvRect left_eye, CvRect right_eye, int centers_coeff, IplImage *frame);

int CalcEdgeSquare(int **mass, int i, int j, int z);

/*! \fn int GetEyeDist(IplImage* frame)
    \brief Gets distanse between eyelids.
    \param frame Image where eye were found.
*/
int *GetEyeDist(IplImage *frame);

CvPoint GetEyesCenter(CvRect left_eye, CvRect right_eye);

float GetDistanceBetweenEyes(CvRect left_eye, CvRect right_eye);

CvRect BuildFace(float cur_face_up_koeff, float cur_face_down_koeff, float cur_face_side_koeff, CvRect left_eye,
                 CvRect right_eye);

#endif // ALGORITHMS_H









