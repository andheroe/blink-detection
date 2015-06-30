#include <iostream>
#include "algorithms.h"

using namespace std;
using namespace cv;

struct eye_params {
    int length;
    int up_border;
    int down_border;
};

bool face_is_found = false;
bool face_refreshed = false;
bool left_eye_is_found = false;
bool right_eye_is_found = false;
bool eyes_are_closed = false;

int false_eyes_count = 0;
int find_eyes_refresh_count = 0;
int max_false_eyes = 5;
int find_eyes_refresh_iterations = 3;

float cur_face_up_koeff = 0;
float cur_face_down_koeff = 0;
float cur_face_side_koeff = 0;
float expand_koeff = 2;

eye_params left_eye_params;
eye_params right_eye_params;

CvRect face;
CvRect left_eye_area;
CvRect right_eye_area;
CvRect left_eye;
CvRect right_eye;

IplImage *grey_frame;

void SaveCurFaceKoeffs() {
    CvPoint center = GetEyesCenter(left_eye, right_eye);

    float centers_dist = GetDistanceBetweenEyes(left_eye, right_eye);

    cur_face_up_koeff = abs(center.y - face.y) / centers_dist;
    cur_face_down_koeff = abs(face.y + face.height - center.y) / centers_dist;
    cur_face_side_koeff = abs(center.x - face.x) / centers_dist;
}

int main(int argc, const char **argv) {
    CvCapture *capture = 0;
    Mat frame, frameCopy, image;

    capture = cvCaptureFromCAM(CV_CAP_ANY); //0=default, -1=any camera, 1..99=your camera
    if (!capture) {
        cout << "No camera detected" << endl;
    }

    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 640);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 480);

    cvNamedWindow("result", CV_WINDOW_AUTOSIZE);

    InitCascades(); //

    if (capture) {
        cout << "In capture ..." << endl;
        for (; ;) {
            IplImage *iplImg = cvQueryFrame(capture);
            frame = iplImg;

            if (frame.empty())
                break;
            if (iplImg->origin == IPL_ORIGIN_TL)
                frame.copyTo(frameCopy);
            else
                flip(frame, frameCopy, 0);

            if (false_eyes_count <= 0) {
                face_is_found = false;
                face = Find(FACE, iplImg, face_is_found, 0, 0.5);

                face_refreshed = true;
            }

            if (face_is_found) {
                if (find_eyes_refresh_count == 0) {
                    CvRect temp;

                    left_eye_area = left_eye;
                    if (false_eyes_count <= 0) {
                        left_eye_area = ClarifyArea(LEFT_EYE, &face);
                    } else {
                        ExpandArea(left_eye_area, expand_koeff);
                    }
                    temp = Find(LEFT_EYE, iplImg, left_eye_is_found, &left_eye_area);
                    if (left_eye_is_found) {
                        left_eye = temp;
                    }

                    right_eye_area = right_eye;
                    if (false_eyes_count <= 0)
                        right_eye_area = ClarifyArea(RIGHT_EYE, &face);
                    else
                        ExpandArea(right_eye_area, expand_koeff);
                    temp = Find(RIGHT_EYE, iplImg, right_eye_is_found, &right_eye_area);
                    if (right_eye_is_found)
                        right_eye = temp;
                }
                find_eyes_refresh_count++;

                if ((left_eye_is_found && right_eye_is_found) && (!EyesCorrect(left_eye, right_eye, iplImg))) {
                    left_eye_is_found = false;
                    right_eye_is_found = false;
                    false_eyes_count = 1;
                }

                if (left_eye_is_found && right_eye_is_found)
                    false_eyes_count = max_false_eyes;
                else
                    false_eyes_count--;
            }

            grey_frame = cvCreateImage(cvSize(iplImg->width, iplImg->height), iplImg->depth, 1);
            cvCvtColor(iplImg, grey_frame, CV_RGB2GRAY);

            if (left_eye_is_found && right_eye_is_found) {
                if (face_refreshed)
                    SaveCurFaceKoeffs();
                else
                    face = BuildFace(cur_face_up_koeff, cur_face_down_koeff, cur_face_side_koeff, left_eye, right_eye);

                DrawRect(face, 150, 150, 150, iplImg);

                int *eye_params;

                cvSetImageROI(grey_frame, left_eye);
                eye_params = GetEyeDist(grey_frame);
                left_eye_params.length = eye_params[0];
                left_eye_params.down_border = eye_params[1];
                left_eye_params.up_border = eye_params[2];
                cvResetImageROI(grey_frame);

                cvSetImageROI(iplImg, left_eye);
                DrawRhombus(eye_params, 0, 255, 0, left_eye.height, left_eye.width, iplImg);
                delete[] eye_params;
                cvResetImageROI(iplImg);

                cvSetImageROI(grey_frame, right_eye);
                eye_params = GetEyeDist(grey_frame);
                right_eye_params.length = eye_params[0];
                right_eye_params.down_border = eye_params[1];
                right_eye_params.up_border = eye_params[2];
                cvResetImageROI(grey_frame);

                cvSetImageROI(iplImg, right_eye);
                DrawRhombus(eye_params, 0, 255, 0, right_eye.height, right_eye.width, iplImg);
                delete[] eye_params;
                cvResetImageROI(iplImg);

                ///////////////////////////////////////////////////////////// BLINK DETECTION

                if ((left_eye_params.up_border >= left_eye.height / 2) || (right_eye_params.up_border >= right_eye.height / 2)) {
                    if (!eyes_are_closed) {
                        cout << "blink" << endl;
                    }
                    eyes_are_closed = true;
                    DrawRect(left_eye, 0, 0, 0, iplImg);
                    DrawRect(right_eye, 0, 0, 0, iplImg);
                } else {
                    eyes_are_closed = false;
                    DrawRect(left_eye, 255, 255, 255, iplImg);
                    DrawRect(right_eye, 255, 255, 255, iplImg);
                }

                /////////////////////////////////////////////////////////////
            } else {
                find_eyes_refresh_count = 0;
            }
            if (find_eyes_refresh_count > find_eyes_refresh_iterations) {
                find_eyes_refresh_count = 0;
            }

            face_refreshed = false;

            cvShowImage("result", iplImg);

            if (waitKey(10) >= 0)
                break;
        }
    }

    cvReleaseCapture(&capture);
    cvDestroyWindow("result");

    return 0;
}