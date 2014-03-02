/*
 * main.cpp
 *
 *  Created on: 02/03/2014
 *      Author: Ignacio Mellado-Bataller
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

Point2f objectPosition(20, 300);
Point2f objectSize(80, 80);

void drawObject(Mat &frame) {
	rectangle(frame, objectPosition, objectPosition + objectSize, CV_RGB(0, 0, 255), CV_FILLED);
	rectangle(frame, objectPosition + objectSize * 0.4 + Point2f(10, 10), objectPosition + objectSize * (1 - 0.4) + Point2f(10, 10), CV_RGB(255, 255, 0), CV_FILLED);
	rectangle(frame, objectPosition + objectSize * 0.4 - Point2f(10, 10), objectPosition + objectSize * (1 - 0.4) - Point2f(10, 10), CV_RGB(0, 0, 0), CV_FILLED);
}

void updateObject(double dt) {
	objectPosition = objectPosition + dt * Point2f(100, -40);
}

void drawScene(Mat &frame) {
	Size size(640, 480);
	frame = Mat::zeros(size, CV_8UC3);
	rectangle(frame, Rect(size.width * 0.4, size.height * 0.2, size.width * 0.2, size.height * 0.5), CV_RGB(0, 0, 255), CV_FILLED);
	drawObject(frame);
}

Mat objectHistogram;
Mat globalHistogram;

void getObjectHistogram(Mat &frame) {
	const int channels[] = { 0, 1 };
	const int histSize[] = { 64, 64 };
	float range[] = { 0, 256 };
	const float *ranges[] = { range, range };
	// Histogram in object region
	Mat objectROI = frame(Rect(objectPosition, Size(objectSize.x, objectSize.y)));
	calcHist(&objectROI, 1, channels, noArray(), objectHistogram, 2, histSize, ranges, true, false);
	// A priori color distribution with cumulative histogram
	calcHist(&frame, 1, channels, noArray(), globalHistogram, 2, histSize, ranges, true, true);
	// Boosting: Divide conditional probabilities in object area by a priori probabilities of colors
	for (int y = 0; y < objectHistogram.rows; y++) {
		for (int x = 0; x < objectHistogram.cols; x++) {
			objectHistogram.at<float>(y, x) /= globalHistogram.at<float>(y, x);
		}
	}
	normalize(objectHistogram, objectHistogram, 0, 255, NORM_MINMAX);
}

Rect trackingWindow(objectPosition.x, objectPosition.y, objectSize.x, objectSize.y);

void backProjection(const Mat &frame, const Mat &histogram, Mat &bp) {
	const int channels[] = { 0, 1 };
	float range[] = { 0, 256 };
	const float *ranges[] = { range, range };
	calcBackProject(&frame, 1, channels, objectHistogram, bp, ranges);
}

int main() {
	namedWindow("tracking", CV_WINDOW_AUTOSIZE);
	moveWindow("tracking", 100, 100);

	Mat frame, hsvFrame;
	double dt = 0.1;
	for (double t = 0; t < 5; t += dt) {
		drawScene(frame);
		// Convert to HSV
		cvtColor(frame, hsvFrame, CV_BGR2HSV);
		// Update object histogram
		getObjectHistogram(hsvFrame);
		// Compute P(O|B)
		Mat bp;
		backProjection(hsvFrame, objectHistogram, bp);
		// Tracking
		meanShift(bp, trackingWindow, TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 0.01));
		// Draw things
		imshow("bp", bp);
		rectangle(frame, trackingWindow, CV_RGB(255, 255, 255), 2);
		imshow("tracking", frame);
		waitKey(dt * 1000);
		// Move object
		updateObject(dt);
	}
	return 0;
}
