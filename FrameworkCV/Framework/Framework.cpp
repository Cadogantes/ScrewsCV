#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cmath>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "Header.h"	//thanks to this we can use calibrate.cpp to calibrate camera

using namespace cv;
using namespace std;
struct features{
	vector<Point2f> mc;	//mass center
	vector<double> theta; //orientation angle
	vector<Moments> mu; //moments
	double circumference, area;	//circumference and area 
	double hu[7];		//Hu moments invariants
	int index;		//database record will begin with this
	int length, diameter, headDiameter; //of a screw
	int weight; //a parameter used in auto-updating category (creating a template of average part in that category)
	int threads;	//number of threads in screw
	//shape factors
	double W1, W2, W3, W4, W6, W7, W8, W9;

	int type;
};

struct coupledImages{
	Mat bw;
	Mat color;
	Mat edge;
};

//declarations
double correctAngle(cv::Mat src);
coupledImages findROI(coupledImages src, bool showImages, bool preprocess);
cv::Mat clearSaltPepper(cv::Mat src, int areaThreshold);
int countThreads(cv::Mat src);
//end of declarations

double getW1(double area){		//circularity factor 1
	double W1 = 2 * sqrt(area / 3.1415926535);
		return W1;
}

double getW2(double circumference){	//circularity factor 2
	double W2 = circumference / 3.1415926535;
	return W2;
}

double getW3(double area, double circumference){	//Malinowska's factor
	double W3 = circumference / (2*sqrt(2*3.1415926535*area))-1;
	return W3;
}

double getW4(double area, Point2f massCenter, cv::Mat I){	//Blair-Bliss factor
	double integral=0;
	for (int i = 0; i < I.rows; i++)
		for (int j = 0; j < I.cols; j++){
		if (I.at<Vec3b>(i, j)[0] == 255)
			integral += pow(massCenter.x - j, 2) + pow(massCenter.y - i, 2);
		}
	double W4 = area / sqrt(2 * 3.1415926535 * integral);
	return W4;
}

double getW6(Point2f massCenter,cv::Mat I){	//Haralick factor
	double sum = 0;	double sumOfSquares = 0; int n = 0;

	//finding external contour of object in image I
	cv::Mat image = I;
		cv::cvtColor(image, image, CV_BGR2GRAY);	//without this findContours function doesn't work.
		vector<vector<Point> > contours;
		findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0)); //find contour in the picture
		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());

		approxPolyDP(Mat(contours[0]), contours_poly[0], 3, true);

		/// Draw polygonal contour + bonding rect
		Scalar contourColor = Scalar(255, 255, 255);
		Mat drawing = Mat::zeros(image.size(), CV_8UC3);
		drawContours(drawing, contours_poly, 0, contourColor, 1);	//creating image with just external contour drawn

		int productX, productY;
		for (int i = 0; i < drawing.rows; i++)
			for (int j = 0; j < drawing.cols; j++){
			if (drawing.at<Vec3b>(i, j)[0] == 255){
				productX = massCenter.x - j; productY = massCenter.y - i;
			sumOfSquares += pow(productX, 2) + pow(productY, 2);
			sum += sqrt(pow(productX, 2) + pow(productY - i, 2));
				n++;
			}
		}
	double W6 = sqrt(pow(sum,2)/(n*sumOfSquares-1));
	return W6;
}

double getW7(Point2f massCenter, cv::Mat I){		//Lp1 factor 
	double rMin = 2147483647; double rMax = 0; double radius = 0;

	//finding external contour of object in image I
	cv::Mat image = I;
	cv::cvtColor(image, image, CV_BGR2GRAY);	//without this findContours function doesn't work.
	vector<vector<Point> > contours;
	findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0)); //find contour in the picture
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	approxPolyDP(Mat(contours[0]), contours_poly[0], 3, true);

	/// Draw polygonal contour + bonding rect
	Scalar contourColor = Scalar(255, 255, 255);
	Mat drawing = Mat::zeros(image.size(), CV_8UC3);
	drawContours(drawing, contours_poly, 0, contourColor, 1);	//creating image with just external contour drawn

	for (int i = 0; i < drawing.rows; i++)
		for (int j = 0; j < drawing.cols; j++){
		if (drawing.at<Vec3b>(i, j)[0] == 255){
			radius = sqrt(pow(massCenter.x - j, 2) + pow(massCenter.y - i, 2));
			if (radius > rMax) rMax = radius;
			if (radius < rMin) rMin = radius;
		}
		}
	double W6 = rMin/rMax;
	return W6;
}

double getW8(double max, double circumference){	//Lp2 factor. max is the biggest dimension of contour
	double W8 = max / circumference;
	return W8;
}

double getW9(double area, double circumference){	//modified Malinowska's factor
	double W9 = 2 * sqrt(3.1415926535*area)/circumference ;
	return W9;
}

cv::Mat stretchContrast(cv::Mat& src)
{
	Mat dst;

	if (!src.data)
	{
		cout << "Usage: ./Histogram_Demo <path_to_image>" << endl;
		waitKey();
		return src;
	}

	/// Convert to grayscale
	//cvtColor(src, src, CV_BGR2GRAY);

	/// Apply Histogram Equalization
	equalizeHist(src, dst);

	return dst;
}

cv::Mat unsharpMask(cv::Mat src, int radius, double k)
{
	Mat blurred, temp, retval;
	cv::GaussianBlur(src, blurred, cv::Size(radius, radius), 0, 0);// get all the low frequency pixels 
	temp = src - blurred; // all the low frequency pixels will be 0 
	retval = src + k*(temp); // k is in [0.3,0.7]
	return retval;
	// in this final result , all low frequency pixels of IMGsharp is same as IMG, 
	// but all high frequency signals of IMGsharp is (1+k)times higher than IMG 
}

cv::Mat unsharp(cv::Mat frame, int blurKsize, double originalWeight, double blurredWeight)
{
	cv::Mat image = frame;
	cv::GaussianBlur(frame, image, cv::Size(blurKsize, blurKsize), 0, 0);
	cv::addWeighted(frame, originalWeight, image, blurredWeight, 0, image);
	return image;
}

cv::Mat detectWithSobel(cv::Mat src){	//detects edges of src image using Sobel's algorithm. 
	//Prepare the image for findContours
	cv::Mat image = src.clone();
	GaussianBlur(src, image, Size(3, 3), 0, 0, BORDER_DEFAULT); //blur
	cout << "\n" << "Initiating Sobel's edge detection algorithm..." << "\n";
	cvtColor(image, image, CV_BGR2GRAY);

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat src_gray = image;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	/// Gradient X
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, image);

	//cv::namedWindow("Sobel's before threshold", cv::WINDOW_NORMAL);
	//cv::imshow("Sobel's before threshold", image);

	// Otsu's thresholding
	cv::threshold(image, image, 0, 255, THRESH_BINARY + THRESH_OTSU);
	
	//cv::namedWindow("Sobel's after postprocessing", cv::WINDOW_NORMAL);
	//cv::imshow("Sobel's after postprocessing", image);

	cout << "\n" << "..finished" << "\n";
	return image;
}

coupledImages detectWithScharr(cv::Mat& src){	//detects edges of src image using Scharr's algorithm.
	int showImages = 0; //used to fine-tune algorithm

	if (showImages){
		cv::namedWindow("Scharr's input", cv::WINDOW_NORMAL);
		cv::imshow("Scharr's input", src);
	}
	coupledImages results;
	results.color = src.clone();
	Mat src_gray;
	cvtColor(src, src_gray, CV_BGR2GRAY);
	cout << "\n" << "Using Scharr's edge detection algorithm" << "\n";
	
	
	//Prepare the image for findContours
	Mat image;
	image= unsharp(src_gray, 5, 1.5, -0.5);
	//GaussianBlur(src, image, Size(3, 3), 0, 0, BORDER_DEFAULT);		//blur	

	//Sobel or Scharr edge detection	
	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	/// Gradient X
	Scharr(src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	Scharr(src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, image);

	if (0){
		cv::namedWindow("threshold test 1", cv::WINDOW_NORMAL);
		cv::imshow("threshold test 1", image);
		cv::imwrite("Gray Scharr.png", image);
	}

		erode(image, image, cv::Mat(), cv::Point(), 1);

	if (0){
		cv::namedWindow("eroded threshold test 1", cv::WINDOW_NORMAL);
		cv::imshow("eroded threshold test 1", image);
		cv::imwrite("Gray Scharr eroded.png", image);
	}
	// Otsu's thresholding
	GaussianBlur(image, image, Size(5, 5), 0, 0, BORDER_DEFAULT);		//blur	
	cv::threshold(image, image, 0, 255, THRESH_BINARY + THRESH_OTSU);
	cv::imwrite("Scharr after Otsu.png", image);

	results.bw = image;
	results.edge = image;
	return results;

}

coupledImages edgesPostprocess(coupledImages src, int showImages){
	coupledImages results;
	results.bw = src.bw.clone(); results.color = src.color.clone(); results.edge = src.edge.clone();
	Mat image = results.bw;
	int size = image.rows*image.cols;
	if (showImages){
		cv::namedWindow("threshold test 2", cv::WINDOW_NORMAL);
		cv::imshow("threshold test 2", image);
	}
	//postprocessing
	if (size < 100000){
		erode(image, image, cv::Mat(), cv::Point(), 1);
		dilate(image, image, cv::Mat(), cv::Point(), 2);
		results = findROI(results, 0, 0);
		//erode(image, image, cv::Mat(), cv::Point(), 3);
	}

	else if (size < 200000){
		erode(image, image, cv::Mat(), cv::Point(), 1);
		dilate(image, image, cv::Mat(), cv::Point(), 2);
		results = findROI(results, 0, 0);
		//erode(image, image, cv::Mat(), cv::Point(), 3);
	}

	else if (size < 500000){
		erode(image, image, cv::Mat(), cv::Point(), 1);
		dilate(image, image, cv::Mat(), cv::Point(), 4);
		results = findROI(results, 0, 0);
		//erode(image, image, cv::Mat(), cv::Point(), 3);
	}

	else if (size < 2000000){
		erode(image, image, cv::Mat(), cv::Point(), 1);
		dilate(image, image, cv::Mat(), cv::Point(), 6);
		results = findROI(results, 0, 0);
		//erode(image, image, cv::Mat(), cv::Point(), 7);
	}

	else{
		erode(image, image, cv::Mat(), cv::Point(), 1);
		dilate(image, image, cv::Mat(), cv::Point(), 8);
		results = findROI(results, 0, 0);
		//erode(image, image, cv::Mat(), cv::Point(), 6);
	}

	if (showImages){
		cv::namedWindow("processed Scharr's output", cv::WINDOW_NORMAL);
		cv::imshow("processed Scharr's output", results.bw);
		waitKey();
	}

	return results;
}

void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
	double angle;
	double hypotenuse;
	angle = atan2((double)p.y - q.y, (double)p.x - q.x); // angle in radians
	hypotenuse = sqrt((double)(p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
	//    double degrees = angle * 180 / CV_PI; // convert radians to degrees (0-180 range)
	//    cout << "Degrees: " << abs(degrees - 180) << endl; // angle in 0-360 degrees range
	// Here we lengthen the arrow by a factor of scale
	q.x = (int)(p.x - scale * hypotenuse * cos(angle));
	q.y = (int)(p.y - scale * hypotenuse * sin(angle));
	line(img, p, q, colour, 1, CV_AA);
	// create the arrow hooks
	p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
	p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
	line(img, p, q, colour, 1, CV_AA);
	p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
	p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
	line(img, p, q, colour, 1, CV_AA);
}

cv::Mat findInitialROI(cv::Mat original, double scale){//original- BGR image. Scale - safety margin for object detection. Initial roi will have size equal to scale*biggest_contour_rect
	cout << "\n" << "Searching for initial ROI..." << "\n";

	if (original.data == NULL)	//check if image has been loaded properly
	{
		cout << "\n" << "Error. Image load failed. Check image's path." << "\n";
		cv::waitKey(-1);
	}

	cv::Mat image = original;

	image = detectWithSobel(image);
	erode(image, image, cv::Mat(), cv::Point(), 2);
	dilate(image, image, cv::Mat(), cv::Point(), 4);
	//cv::namedWindow("Sobel output", cv::WINDOW_NORMAL);
	//cv::imshow("Sobel output", image);
	//waitKey();

	//finding contours and bounding rectangle 
	vector<vector<Point> > contours;
	findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0)); //find contours in the picture
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	int chosenContours = 0; double maxArea = 0;	//variables used to find the right (biggest) contour

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		double area = contourArea(contours[i]);
		if (maxArea < area)
		{
			maxArea = area;
			chosenContours = i;
		}

	}
	boundRect[chosenContours] = boundingRect(Mat(contours_poly[chosenContours])); //finds rectangle bounding contours that interest us

	// Resize bounding rect to get bigger area (in case we didn't detect things quite properly)
	int offset_x = 0; int offset_y = 0;	//offset values will come in use if extended bounding rect would result in coming out of the picture

	int widthChange = boundRect[chosenContours].width * (scale - 1);
	int heightChange = boundRect[chosenContours].height * (scale - 1);

	//top left corner x
	if (boundRect[chosenContours].x - widthChange / 2 > 0)
		boundRect[chosenContours].x -= widthChange / 2;
	else {
		offset_x = abs(boundRect[chosenContours].x - widthChange / 2); //in case we came out of pic during bounding rect enlargment we correct that and introduce offset
		boundRect[chosenContours].x = 0;
	}

	//top left corner y
	if (boundRect[chosenContours].y - heightChange / 2 > 0)
		boundRect[chosenContours].y -= heightChange / 2;
	else {
		offset_y = abs(boundRect[chosenContours].y - heightChange / 2);	//in case we came out of pic during bounding rect enlargment we correct that and introduce offset
		boundRect[chosenContours].y = 0;
	}

	//boudning box dimensions
	if (boundRect[chosenContours].width + (widthChange - offset_x) < (original.size().width - boundRect[chosenContours].x))
		boundRect[chosenContours].width += (widthChange - offset_x);
	else boundRect[chosenContours].width = (original.size().width - boundRect[chosenContours].x);

	if (boundRect[chosenContours].height + (heightChange - offset_y) < original.size().height - boundRect[chosenContours].y)
		boundRect[chosenContours].height += (heightChange - offset_y);
	else boundRect[chosenContours].height = original.size().height - boundRect[chosenContours].y;


	cv::Mat roi2(original, cv::Rect(boundRect[chosenContours].tl(), boundRect[chosenContours].br())); //roi in original image

	cout << "\n" << "...initial ROI found." << "\n";

	return roi2;	//roi2- cropped from original image,
}

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

coupledImages findROI(coupledImages src, bool showImages, bool color){
	coupledImages results=src;
	Mat original = src.color;
	if (original.data == NULL)	//check if image has been loaded properly
	{
		cout << "\n" << "Error. Image load failed. Check image's path." << "\n";
		cv::waitKey(-1);
	}

	cv::Mat image = original.clone();

	if (color){

		results = detectWithScharr(image);
		results = edgesPostprocess(results, 0);
		image = results.bw;
		
	}

	
	if (!color){
		results = src;
		image = results.bw;
	}
	if (type2str(image.type()) == "8UC3"){		//sometimes after processing black and white images are converted to 8UC3 type. We need 8UC1 to use findContours()
		cvtColor(image, image, CV_BGR2GRAY);
		threshold(image, image, 1, 255, THRESH_BINARY);
	}

		//finding contours and bounding rectangle 
		vector<vector<Point> > contours;
		findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0)); //find contours in the picture
		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());
		int chosenContours = 0; double maxArea = 0;	//variables used to find the right (biggest) contour

		if (color)
			cout << "\n" << "Looking for biggest contour" << "\n";

		for (int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			double area = contourArea(contours[i]);
			if (maxArea < area)
			{
				maxArea = area;
				chosenContours = i;
			}

		}
		boundRect[chosenContours] = boundingRect(Mat(contours_poly[chosenContours])); //finds rectangle bounding contours that interest us

		/// Draw polygonal contour + bonding rect
		Mat drawing = Mat::zeros(image.size(), CV_8UC3);
		Mat externalContour = Mat::zeros(image.size(), CV_8UC3);
		Scalar contourColor = Scalar(255, 255, 255);
		Scalar rectangleColor = Scalar(155, 155, 155);
		drawContours(drawing, contours_poly, chosenContours, contourColor, CV_FILLED);	//creating image with drawn filled contour
		drawContours(externalContour, contours_poly, chosenContours, contourColor, 1);	//creating image with just external contour drawn
		boundRect[chosenContours].x--; boundRect[chosenContours].y--; boundRect[chosenContours].width += 2; boundRect[chosenContours].height += 2; //creating one-pixel wide rectangle around roi- it helps with refinding contours;

		cv::Mat roi2(results.color, cv::Rect(boundRect[chosenContours].tl(), boundRect[chosenContours].br())); //roi in original image
		Mat roi3(drawing, cv::Rect(boundRect[chosenContours].tl(), boundRect[chosenContours].br())); //roi in processed image
		Mat roi4(results.edge, cv::Rect(boundRect[chosenContours].tl(), boundRect[chosenContours].br())); //this is raw output from Scharr's algorithm cropped to ROI

		results.bw = roi3; results.color = roi2; results.edge = roi4;	//roi2- cropped from original image, roi3- cropped from processed image (binary)

		return results;	
	}

cv::Mat rotateImage(cv::Mat src, double angle){//rotates src around its center by angle. If showImages is set to 1 it will also show the result in window.

	// get rotation matrix for rotating the image around its center
	cv::Point2f center(src.cols / 2.0, src.rows / 2.0);
	cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
	// determine bounding rectangle
	cv::Rect bbox = cv::RotatedRect(center, src.size(), angle).boundingRect();
	// adjust transformation matrix
	rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;

	cv::Mat dst;
	cv::warpAffine(src, dst, rot, bbox.size());	//rotates src image
	
	return dst;
}

cv::Mat clearSaltPepper(cv::Mat src, int areaThreshold){//removes from black and white src image all contours  which size are smaller than areaThreshold
	if (type2str(src.type()) == "8UC3"){		
		cvtColor(src, src, CV_BGR2GRAY);
		cv::threshold(src, src, 1, 255, THRESH_BINARY);
	}
	Mat image = src.clone();
	Mat drawing = Mat::zeros(src.size(), CV_8UC1);
	//finding contours and bounding rectangle 
	vector<vector<Point> > contours;
	findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0)); //find contours in the picture
	vector<vector<Point> > contours_poly(contours.size());
	int chosenContours = 0; double maxArea = 0;	//variables used to find the right (biggest) contour

	cout << "\n" << "Looking for biggest contour" << "\n";

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		double area = contourArea(contours[i]);
		if (area > areaThreshold)
			drawContours(drawing, contours_poly, i, Scalar(255), CV_FILLED);
	}
	imwrite("image after removal of small contours.png", drawing);
	return drawing;
}

double correctAngle(cv::Mat src){
	cout << "\n" << "Correcting angle" << "\n";
	double tax, angle, chosenAngle;
	double minTax = std::numeric_limits<double>::max();	//largest possible double
	Mat image;

	for (angle=-15; angle<=15; angle+=1){
		image = rotateImage(src, angle);
			tax = 0;
			for (int i = 0; i < image.rows; i++)
				for (int j = 0; j < image.cols; j++){
				if (image.at<Vec3b>(i, j)[0] == 255){
					tax += abs(pow(image.size().width / 2 - j, 2)); //we will measure and try to minimize squared distance from y axis of all points in image
			}
			}
			if (tax < minTax){
				minTax = tax;
				chosenAngle = angle;
			}
	}
	return chosenAngle;
}

int checkOrientation(cv::Mat src){
	int orientation = 0; int size; int start; int width; int y; int maxWidth = 0;

	for (int i = 1; i < src.rows; i++){
		size = 0;	//size we measure (width)
		start = 0;	//determines whether we've already started measuring

		for (int j = 1; j < src.cols; j++){
			if (src.at<Vec3b>(i, j)[0] == 255 && start == 0){
				start = 1;
				size++;
			}
			if (src.at<Vec3b>(i, j)[0] == 255 && start == 1) {
				size++;
				width = size;
			}
					
			}
		if (width>maxWidth){
			maxWidth = width;
			y = i;
		}
	}
	if (y > src.rows / 2)
		orientation = 1;

	//cv::Mat kontury = src.clone();
	//cvtColor(kontury, kontury, CV_BGR2GRAY);
	//vector<vector<Point> > contours;
	//vector<Vec4i> hierarchy;
	//findContours(kontury, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	///// Get the moments
	//vector<Moments> mu(contours.size());
	//for (int i = 0; i < contours.size(); i++)
	//{
	//	mu[i] = moments(contours[i], false);
	//}

	/////  Get the mass centers:
	//vector<Point2f> mc(contours.size());
	//for (int i = 0; i < contours.size(); i++)
	//{
	//	mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	//}

	//if (mc[0].y > src.rows / 2) orientation = 1; 

	return orientation;
}

int getDiameter2(cv::Mat img) { //this function looks for a diameter of a screw in binary image. Method is simplistic so its use is limited only to screws positioned right (with their axis parallel to y axis)
	Mat image = img.clone();
	CV_Assert(img.depth() != sizeof(uchar));
	Mat gray;

	if (image.channels() == 3)
	{
		cvtColor(image, gray, CV_BGR2GRAY);
	}
	else
	{
		gray = image;
	}
	cv::threshold(image, image, 200, 255, 0);	//let's turn it black and white

	//first we need to store diameter values on some representative part of screw
	vector<double> diameters;
	int diameter; int size = 0; 
	for (int i = (img.rows *0.4); i < (img.rows *0.8); i++){					//we will check a width of a few rows to be sure we've found the real diameter
		int start = 0; size = 0; diameter = 0;
		for (int j = 0; j < img.cols; j++){
			
			if (start == 1) //keep counting if you've already started
				size++;

			else if (gray.at<uchar>(i, j) == 255 && start == 0){ //look for the first white pixel
				start = 1; size++;
			}

			if (gray.at<uchar>(i, j) == 255 && start == 1) //if the pixel is white than you just might've found the edge- so better update that diameter
				if (size>diameter) diameter = size;
		}
		diameters.push_back(diameter);
	}

	//now we will look for maxima- screw nominal diameter is measured as maximum width
	vector<double> maxima;
	double before; double after; bool trend = 1; double current;
	for (int i = 1; i < diameters.size() - 1; i++){
		before = diameters[i - 1];
		after = diameters[i + 1];
		current = diameters[i];

		if (after <= current && trend == 1)
			maxima.push_back(current);

		if (after > current)
			trend = 1;
		else if (after <= current)
			trend = 0;
	}
	int denominator = maxima.size();
	if (denominator != 0){
		size = 0;
		for (int i = 0; i <= maxima.size() - 1; i++){
			size += maxima[i];
		}
		diameter = size / denominator;	//this will calculate diameter as a mean value of all diameters found;
	}
	else return -1;
	return diameter;

}

int getDiameter(cv::Mat img) { //this function looks for a diameter of a screw in binary image. Method is simplistic so its use is limited only to screws positioned right (with their axis parallel to y axis)

	CV_Assert(img.depth() != sizeof(uchar));
	Mat gray;

	if (img.channels() == 3)
	{
		cvtColor(img, gray, CV_BGR2GRAY);
	}
	else
	{
		gray = img;
	}
	cv::threshold(img, img, 200, 255, 0);	//let's turn it black and white

	int diameter = 0; int size = 0; int denominator = 0; int sum = 0;
	for (int i = (img.rows *0.6); i < (img.rows *0.7); i++){					//we will check a width of a few rows to be sure we've found the real diameter
		int start = 0; denominator++;
		for (int j = 0; j < img.cols; j++){

			if (start == 1) //keep counting if you've already started
				size++;

			else if (gray.at<uchar>(i, j) == 255 && start == 0){ //look for the first white pixel
				start = 1; size++;
			}

			if (gray.at<uchar>(i, j) == 255 && start == 1) //if the pixel is white than you just might've found the edge- so better update that diameter
				if (size>diameter) diameter = size;
		}
		sum += diameter;			//sum of all diameters found
	}

	if (denominator != 0)
		diameter = size / denominator;	//this will calculate diameter as a mean value of all diameters found;

	return diameter;

}

int loadData(features *Database, string name){ //this function will load database from file to Database struct and return the index of most recent position

	int index, maxIndex = -1;
	ifstream myfile(name);
	if (myfile.is_open()) //file has been opened succesfully
	{
		string dummy; getline(myfile,dummy);	//this line will extract database header (1st line) into dummy string
		for (int num; myfile >> num;){	//loop control- read from file as long as you have something to read

			index = num;	//we have already taken index from file so we have to use it like that
			Database[index].index = index;
			myfile >> Database[index].diameter;
			myfile >> Database[index].length;
			myfile >> Database[index].weight;
			myfile >> Database[index].headDiameter;
			myfile >> Database[index].threads;
			

			if (index > maxIndex) maxIndex = index;
		}
		
	}
	else //file could not be opened
	{
		cout << "File could not be opened. Database load failed (it might not exist)." << endl;
		return -1;
	}
	myfile.close();
	
	return maxIndex;
	}

void saveFeatures(features data){	//saves single data record to results.txt file and indexes it properly

	ifstream is("results.txt", ios::app); //opening an input stream for file results.txt
	std::string id;
	int newIndex = 1; int highestIndex = 1;

	getline(is, id);	//we will skip first line as it's a header
	while (getline(is, id, '\t')){					//this loop will look for the highest index in database
		highestIndex = stoi(id);
		if (highestIndex >= newIndex)
			newIndex = highestIndex + 1;	
		getline(is, id);			//skips to new line
	}
	is.close();

	ofstream fout("results.txt", ios::app); //opening an output stream for file database.txt
	/*checking whether file could be opened or not. If file does not exist or don't have write permissions, file
	stream could not be opened.*/
	if (fout.is_open())
	{
		//file opened successfully so we are here
		cout << "Results file opened successfully! Writing data from array to file." << endl;

		if (newIndex == 1)	//creates a header to let us know what the values below mean
			fout << "Index" << "\t" <<"Type"<<"\t"<< "S" << "\t" << "Head diameter" << "\t" << "Diameter" << "\t" << "Length" << "\t" <<"Threads"<<"\t"<< "hu 1" << "\t" << "hu 2" << "\t" << "hu 3" << "\t" << "hu 4" << "\t" << "hu 5" << "\t" << "hu 6" << "\t" << "hu 7" << "\t" << "weight" << "\t" << "W1" << "\t" << "W2" << "\t" << "W3" << "\t" << "W4" << "\t" << "W6" << "\t" << "W7" << "\t" << "W8" << "\t" << "W9" << "\t"<<"Contour length"<<"\n";

		fout << newIndex << "\t";	//saves the index of this new set of data
		fout << data.type << "\t";
		fout << data.area << "\t" << data.headDiameter  << "\t" << data.diameter << "\t" << data.length << "\t" << data.threads << "\t"; //saves area, length of contour and diameter
		for (int i = 0; i<7; i++)
		{
			fout << data.hu[i] << "\t"; //saves moment invariants
		}
		fout << data.weight << "\t";
		fout << data.W1 << "\t" << data.W2 << "\t" << data.W3 << "\t" << data.W4 << "\t" << data.W6 << "\t" << data.W7 << "\t" << data.W8 << "\t" << data.W9<<"\t";
		fout << data.circumference;
		fout << "\n";
		cout << "Data successfully saved into the file results.txt" << endl;

		fout.close();
	}
	else //file could not be opened
	{
		cout << "File could not be opened." << endl;
	}
}

void saveDatabase(features *Database, int databaseSize){ //saves whole database to txt file (replacing previous content)
	ofstream fout("database.txt", ios::trunc); //opening an output stream for file database.txt
	/*checking whether file could be opened or not. If file does not exist or don't have write permissions, file
	stream could not be opened.*/
	if (fout.is_open())
	{
		//file opened successfully so we are here
		cout << "Database file opened successfully! Writing data from array to file." << endl;

		for (int i = 0; i <= databaseSize; i++){
			if (i == 0)	//creates a header to let us know what the values below mean
				fout << "Index" << "\t" << "Diameter" << "\t" << "Length" << "\t" << "weight" << "\t"<<"Head diameter"<<"\t"<<"threads"<<"\n";

			fout << Database[i].index << "\t";	//saves the index of this new set of data
			fout <<  Database[i].diameter << "\t" << Database[i].length << "\t" << Database[i].weight << "\t"<<Database[i].headDiameter<<"\t"<<Database[i].threads; //saves area, length of contour and diameter
			fout << "\n";
		}
		cout << "Data successfully saved into the file database.txt" << endl;
		fout.close();
	}
	else //file could not be opened
	{
		cout << "File could not be opened." << endl;
	}
}

int adaptiveCompareData(features data, features *Database, int recentIndex, double errorMargin,double offset){//compares features of the image to determine whether or not similar record exists in database. If it does function returns its index and replaces the record with new (weighted) one
	cout << "\n" << "Comparing with existing database..." << "\n";
	int type;
	//double errorMargin = 15;	//(in %) how much can all features deviate from average for this record to be considered in the same category as average pattern.  IT CAN'T BE INT TYPE (errors later due to conversion to double)
	bool diameter, length, headDiameter, area; //let's chceck all of those seperately for now
	bool matchFound = 0;


	for (int i = 0; i <= recentIndex; i++){
		diameter = 0; length = 0;	headDiameter = 0;	area = 0;

		if ((Database[i].diameter - offset)*(1 - errorMargin / 100) <= data.diameter && (Database[i].diameter + offset)*(1 + errorMargin / 100) >= data.diameter)
			diameter = 1;
		if ((Database[i].length - offset)*(1 - errorMargin / 100) <= data.length && (Database[i].length + offset)*(1 + errorMargin / 100) >= data.length)
			length = 1;
		if ((Database[i].headDiameter - offset)*(1 - errorMargin / 100) <= data.headDiameter && (Database[i].headDiameter + offset)*(1 + errorMargin / 100) >= data.headDiameter)
			headDiameter = 1;


		if (diameter && length &&  headDiameter){
			Database[i].diameter = ((Database[i].diameter*Database[i].weight) + data.diameter) / (Database[i].weight + 1);
			Database[i].length = ((Database[i].length*Database[i].weight) + data.length) / (Database[i].weight + 1);
			Database[i].headDiameter = ((Database[i].headDiameter*Database[i].weight) + data.headDiameter) / (Database[i].weight + 1);

			Database[i].weight++;
			matchFound = 1;
			cout << "\n" << "Match in database found" << "\n";
			return i;
		}
	}

	if (!matchFound){
		recentIndex++;

		Database[recentIndex].diameter = data.diameter;
		Database[recentIndex].length = data.length;
		Database[recentIndex].area = data.area;
		Database[recentIndex].circumference = data.circumference;
		Database[recentIndex].index = recentIndex;
		Database[recentIndex].weight = 1;
		Database[recentIndex].headDiameter = data.headDiameter;

		for (int j = 0; j < 7; j++){
			Database[recentIndex].hu[j] = data.hu[j];
		}
	}
	cout << "\n" << "Match in database NOT found. Creating new type: " <<recentIndex<< "\n";
	return recentIndex;
}

int compareWithDefined(features data, features *Database, double errorMargin, int offset){//errorMargin(in %) how much can all features deviate from average for this record to be considered in the same category as average pattern.  
	cout << "\n" << "Comparing with existing database..." << "\n";
	int type;
	bool diameter, length, headDiameter, area; //let's chceck all of those seperately for now
	bool matchFound = 0;


	for (int i = 0; i <= 8; i++){
		diameter = 0; length = 0;	headDiameter = 0;	area = 0;

		if ((Database[i].diameter-offset)*(1 - errorMargin / 100) <= data.diameter && (Database[i].diameter+offset)*(1 + errorMargin / 100) >= data.diameter)
			diameter = 1;
		if ((Database[i].length-offset)*(1 - errorMargin / 100) <= data.length && (Database[i].length+offset)*(1 + errorMargin / 100) >= data.length)
			length = 1;
		if ((Database[i].headDiameter-offset)*(1 - errorMargin / 100) <= data.headDiameter && (Database[i].headDiameter+offset)*(1 + errorMargin / 100) >= data.headDiameter)
			headDiameter = 1;


		if (diameter && length &&  headDiameter){
			matchFound = 1;
			cout << "\n" << "Match in database found" << "\n";
			return i;
		}
	}

	if (!matchFound){
		cout << "\n" << "Couldn't find a match in database." << "\n";
		return -1;
		}
		
	}

int countThreads(cv::Mat src){//this function measures object diameter throughout it's length and later finds number of local maxima in diameter values. 
	Mat img = src.clone();

	std::vector<int> valuesVector;	//we will store all distances in this vector to make operations on it possible

	//first we must gather data we will write in vector
	int start; int width;
	for (int i = 0; i < img.rows; i++){
		int size = 0; start = 0; width = 0;
		for (int j = 0; j < img.cols-1; j++){
			if (img.at<uchar>(i, j) == 255 && start == 0){ //look for the first white pixel to staret measuring width
				start = 1;	
				size++;
			}
			else if (img.at<uchar>(i, j) == 255 && start == 1){	//if you encounter another white pixel in the same row- update width
				size++;
				width = size;
			}
			}
		valuesVector.push_back(width);
		}

	//now using those gathered data we will find how many local maxima there are in our profile
	int localMaximaCount;
	int maxima = 0; int before; int after; bool trend = 1; int current;
	for (int i = 1; i < valuesVector.size()-1; i++){
		before = valuesVector[i - 1];
		after = valuesVector[i + 1];
		current = valuesVector[i];

		if (after <= current && trend == 1)
			maxima++;

		if (after > current)
			trend = 1;
		else if (after <= current)
			trend = 0;
	}

	//Save all points in vector to analyze it externally
	std::ofstream f("somefile.txt",ios::app);
	for (vector<int>::const_iterator i = valuesVector.begin(); i != valuesVector.end(); ++i) {
		f << *i << '\t';
	}
	f << "\n" << ";";
	f.close();
 	return maxima;
}

int getThreadsLength(cv::Mat src, int radius){	//this countThreads measures distance from picture edge to screw edge and return number of local maxima found in that profile
	Mat img = src.clone();

	std::vector<int> valuesVector;	//we will store all distances in this vector to make operations on it possible
	std::vector<int> filteredVector;	//later we will filter those stored values and put them in this vector
	std::vector<int> meanVector;		//this one will help us get mean diameter value. We will use it to count zero-crossings of original-mean

	//first we must gather data we will write in vector
	int start; int width;
	for (int i = img.rows-1; i >0; i--){	//we will start at the beginning of the screw and go in the direction of the head
		int size = 0; width = 0;
		for (int j = 0; j < img.cols; j++){
			size++;
			if (img.at<uchar>(i, j) == 255){ //look for the first white pixel
				width= size;
				break;
			}
		}

		double diameter = (img.cols / 2 - width) * 2;
		if (valuesVector.size()>15 && diameter>0.75*img.cols) break;	//we don't care about screw's head. Yeah, screw it! Beware- "diameter" will be equal to image width in rows with no object, so let's make sure we actually started taking samples
		else if (diameter >img.cols*0.2)	meanVector.push_back(diameter);		//due to the way we count diameter it might be negative. Whoops.
			valuesVector.push_back(diameter);	//roughly equals distance from screw axis to its edge
			filteredVector.push_back(diameter); //let's put the same thing in vector of filtered values, it will help us with borderline values
	}

	//let's filter gathered data to smoothen things a bit
	for (int i = radius; i < valuesVector.size()-radius; i++){
		int newValue = 0;
		for (int j = -radius; j < radius; j++){
			if (j == 0)
				newValue += 2 * valuesVector[i];	//so let's give the actual value of this elemnt a bit more weight
			else
			newValue += valuesVector[i + j];
		}
		filteredVector[i] = newValue / (2 * radius + 2);	//normalize value before putting it back in
	}

	//let's also compute mean value of all those diameters
	double mean; double sum = 0;
	for (int i = 0; i < meanVector.size(); i++){
		sum += meanVector[i];
	}
	mean = sum / meanVector.size();

	//now let's subtract that mean from filtered values. Thus we will bring all those signals to fluctuate around 0's
	for (int i = 0; i < filteredVector.size(); i++){
		filteredVector[i] = filteredVector[i] - mean;
	}

	int length=0;

	int before, after, current;
	for (int i = 1; i < filteredVector.size() - 1; i++){
		before = filteredVector[i - 1];
		after = filteredVector[i + 1];
		current = filteredVector[i];

		if (before>0 && current < 0)
			length = i;
	}

	//Save all points in vector to analyze it externally
	std::ofstream f1("valuesVector.txt",ios::app);
	for (vector<int>::const_iterator i = valuesVector.begin(); i != valuesVector.end(); ++i) {
		f1 << *i << '\t';
	}
	f1 << "\n";
	f1.close();

	std::ofstream f2("filteredVector.txt", ios::app);
	for (vector<int>::const_iterator i = filteredVector.begin(); i != filteredVector.end(); ++i) {
		f2 << *i << '\t';
	}
	f2 << "\n";
	f2.close();

	std::ofstream f3("meanVector.txt", ios::app);
	for (vector<int>::const_iterator i = meanVector.begin(); i != meanVector.end(); ++i) {
		f3 << *i << '\t';
	}
	f3 << "\n";
	f3.close();

	return length;
}

features calculateFeatures(cv::Mat oryginal){ //calculates features of input image

	if (oryginal.data == NULL)		//check if image has been loaded properly
	{
		cout << "\n" << "Error. Image load failed. Check image's path." << "\n";
		cv::waitKey(-1);
	}
	cv::Mat kontury = oryginal.clone();
	cv::cvtColor(oryginal, kontury, CV_RGB2GRAY);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(kontury, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Get the moments
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}

	///  Get the mass centers:
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	vector<Point2f> mg(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mg[i] = Point2f(oryginal.cols / 2, oryginal.rows / 2);	//geometrical center in this case is center of our ROI
	}

	// Get angle
	//vector<RotatedRect> rRect(contours.size());
	vector<RotatedRect> minEllipse(contours.size());
	vector<double>theta(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		//rRect[i] = minAreaRect(contours[i]);
		minEllipse[i] = fitEllipse(contours[i]);
		theta[i] = minEllipse[i].angle;
	}

	features features;		//in this struct I keep the results like moments, area, circumference, etc. 
	features.mu = mu;
	features.mc = mc;
	features.theta = theta;


	/// Calculate the area with the moments 00 and compare with the result of the OpenCV function
	Mat drawing = Mat::zeros(oryginal.size(), CV_8UC3);

	for (int i = 0; i< contours.size(); i++)
	{
		features.circumference = arcLength(contours[i], true);	//calculate circumference of contour
		features.area = contourArea(contours[i]);		//calculate area of contour

		//printf(" * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f \n", i, mu[i].m00, features.S, features.L);
		Scalar color = Scalar(rand() % 256, rand() % 256, rand() % 256);
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		circle(drawing, mc[i], 4, color, -1, 8, 0);
		//circle(drawing, mg[i], 4, Scalar(0,255,0), -1, 8, 0);

	}

	double hu[7];
	//printf("features: \n");
	for (int i = 0; i < mu.size(); i++)
	{
		cv::HuMoments(mu[i], hu);	//calculate moment invariants
		for (int j = 0; j < 7; j++){
	//		hu[j] = log10(abs(hu[j]));	//changes values to logaritmic scale
			features.hu[j] = hu[j];
		}
		//print results
		//printf(" * Wartosci niezmiennikow podane w postaci logarytmow (Hx == log10(abs(Hx))) \n S: %.2f \n L: %.2f \n H1: %.2f \n H2: %.2f \n H3: %.2f \n H4: %.2f \n H5: %.2f \n H6: %.2f \n H7: %.2f \n\n", features.area, features.circumference, hu[0], hu[1], hu[2], hu[3], hu[4], hu[5], hu[6]);
	}
	features.weight = 1;	//as it is a single record its weight will always be 1


	return features;
}

features processImage(string name, bool showImages){ //name- name of image we want to process. showImages- if set to 1 it will show consecutive steps of process. 
	cout << "\n" << "Picture: " << name << "\n";
	cv::Mat original = cv::imread(name);
	if (original.data == NULL)	//check if image has been loaded properly
	{
		cout << "\n" << "Error. Image load failed. Check image's path." << "\n";
		cv::waitKey(-1);
	}
	cv::Mat image = original.clone();

	//a priori cropping of the image- if we know where to expect objects
		cv::Mat dummy(image, cv::Rect(0, 0, //top left corner x,y, coordinates
			0.85*image.cols , image.rows));	//width, height 
		image = dummy;

	image = findInitialROI(image, 2);
	name.erase(name.end() - 4, name.end());
	cv::imwrite(name+" initial ROI.jpg", image);

	coupledImages roi;
	coupledImages src; src.color = image;
	roi = findROI(src, 1, 1);	//find region of interest- in this case screw. Next operations will be performed only in that region to speed up calculations.
	cv::imwrite(name + " initial BW.jpg", roi.bw);
	//prepare image for threads number estimation using edged image
	Mat contour;
	dilate(roi.edge, contour, cv::Mat(), cv::Point(), 2);
	contour = clearSaltPepper(contour, (contour.rows*contour.cols) / 200);
	roi.bw = contour;	// CRITICAL PART FOR FEATURE CALCULATION! REPLACING OLD BW IMAGE WITH BETTER ONE
	roi = findROI(roi, 0, 0);

	if (0)
	{
		cv::namedWindow("ROI bw", cv::WINDOW_NORMAL);
		cv::imshow("ROI bw", roi.bw);

		cv::namedWindow("ROI color", cv::WINDOW_NORMAL);
		cv::imshow("ROI color", roi.color);
		cv::namedWindow("ROI edge", cv::WINDOW_NORMAL);
		cv::imshow("ROI edge", roi.edge);
		//waitKey();
	}

	features data = calculateFeatures(roi.bw); //calculates moments for contours found in input image 
	roi.bw = rotateImage(roi.bw, data.theta[0]);
	roi.color = rotateImage(roi.color, data.theta[0]);
	roi.edge = rotateImage(roi.edge, data.theta[0]);

	roi = findROI(roi, 0, 0);

	//making sure all images are oriented vertically
	if (roi.bw.size().height < roi.bw.size().width){
		roi.bw = rotateImage(roi.bw, 90);
		roi.color = rotateImage(roi.color, 90);
		roi.edge = rotateImage(roi.edge, 90);
	}

	roi = findROI(roi, 0, 0);

	// let's make sure they are screws all oriented the same way- heads up
		int orientation = checkOrientation(roi.bw);
	if (orientation==1){
		roi.bw = rotateImage(roi.bw, 180);
		roi.color = rotateImage(roi.color, 180);
		roi.edge = rotateImage(roi.edge, 180);
	}

	roi = findROI(roi, 0, 0);

	if (showImages){
		cv::namedWindow("rotated bw", cv::WINDOW_NORMAL);
		imshow("rotated bw", roi.bw);
		cv::namedWindow("rotated color", cv::WINDOW_NORMAL);
		imshow("rotated color", roi.color);
		cv::namedWindow("rotated threads", cv::WINDOW_NORMAL);
		imshow("rotated threads", roi.edge);
		waitKey();
	}

	//showing found roi in color image
	Mat compare;
	roi.color.copyTo(compare, roi.bw); //crops BGR version of ROI to detected contour
	imwrite(name + " detected.png", compare);
	if (showImages){
		cv::namedWindow("found ROI shown in original image", cv::WINDOW_NORMAL);
		imshow("found ROI shown in original image", compare);
		waitKey();
	}

	//calculate final features
	data = calculateFeatures(roi.bw);

	Mat mask;
	mask = clearSaltPepper(roi.edge, roi.edge.rows*roi.edge.cols / 300);	//using findContours distorts image a bit, in extreme cases we will loose data about threads as they will be smoothed. 
	dummy = roi.edge.clone();
	roi.edge = Mat::zeros(roi.edge.size(), CV_8UC1);
	dummy.copyTo(roi.edge, mask);	//Therefore we will find contours only to get mask that will allow us to get picture with edges without the noise
	data.threads = getThreadsLength(roi.edge,7); //estimate threads length. Use it AFTER calling calculateFeatures function

	data.length = roi.bw.size().height;
	data.diameter = getDiameter2(roi.bw);
	data.headDiameter = roi.bw.cols - 2;
	printf("\n height of rotated image: %dpx \n diameter of the screw: %ipx \n", roi.bw.size().height, data.diameter);

	std::cout << "\n" << "Calculating W1-W9" << "\n";
	data.W1 = getW1(data.area);
	data.W2 = getW2(data.circumference);
	data.W3 = getW3(data.area, data.circumference);
	data.W4 = getW4(data.area, data.mc[0], roi.bw);
	data.W6 = getW6(data.mc[0], roi.bw);
	data.W7 = getW7(data.mc[0], roi.bw);
	data.W8 = getW8(data.length, data.circumference);	//this arguments assumes that length is the biggest dimension of the object
	data.W9 = getW9(data.area, data.circumference);

	//save final pictures
	cv::imwrite(name+" rotated ROI.png", roi.bw);
	cv::imwrite(name + " rotated edge.png", roi.edge);

	if (showImages)
		waitKey();

	return data;
}

int main(){
	
	int databaseSize = 100;	int type = 0; 
	features *Database = new features[databaseSize]; //array of properties. It will serve as database to compare detected elements
	remove("results.txt"); //clears results from previous session
	remove("valuesVector.txt"); remove("filteredVector.txt"); remove("meanVector.txt");
	int recentIndex = loadData(Database,"database.txt");	//this function will load currently built database to Database struct and return index of most recent entry in database
	
	//loop
	//for (int i = 30; i <= 96; i++){
		//string fileName = "a"+to_string(i)+".jpg";
		string fileName = "lol.jpg";

		features data = processImage(fileName, 0); //processes image and returns features of the biggest object found in it

		type = adaptiveCompareData(data, Database, recentIndex, 10,15);
		//type = compareWithDefined(data, Database, 10,15);	
		data.type = type;
		if (type > recentIndex) recentIndex = type;	//after creating new type this line will update the recent index
		cout <<"\n"<< "Obiekt " + fileName + " is type: " << type<<"\n";
		saveFeatures(data);		//adds a single record to database

	//}	

	saveDatabase(Database, recentIndex);	//Ds whole database at once
	delete[] Database;	//frees memory allocated to database
	std::cout << "\n" << "Processing finished, closing program..."<<"\n";

	waitKey();
	return 0;	


	//for (int i = 1; i <= 96; i++){
	//	string fileName = "a" + to_string(i); 
	//	Mat src = imread(fileName+" detected.png");
	//	Mat src_gray;
	//	cvtColor(src, src_gray, CV_BGR2GRAY);
	//	cout << "\n" << "Detecting threads" << "\n";


	//	//Prepare the image for findContours
	//	Mat image;
	//	image = unsharpMask(src_gray, 5, 2);
	//	//GaussianBlur(src, image, Size(3, 3), 0, 0, BORDER_DEFAULT);		//blur	

	//	//Sobel or Scharr edge detection	
	//	/// Generate grad_x and grad_y
	//	Mat grad_x, grad_y;
	//	Mat abs_grad_x, abs_grad_y;
	//	int scale = 1;
	//	int delta = 0;
	//	int ddepth = CV_16S;

	//	/// Gradient X
	//	Scharr(src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
	//	convertScaleAbs(grad_x, abs_grad_x);

	//	/// Gradient Y
	//	Scharr(src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
	//	convertScaleAbs(grad_y, abs_grad_y);

	//	/// Total Gradient (approximate)
	//	addWeighted(abs_grad_x, -1, abs_grad_y, 2, 0, image);

	//	erode(image, image, cv::Mat(), cv::Point(), 1);

	//	// Otsu's thresholding
	//	GaussianBlur(image, image, Size(5, 5), 0, 0, BORDER_DEFAULT);		//blur	
	//	cv::threshold(image, image, 0, 255, THRESH_BINARY + THRESH_OTSU);
	//	cv::imwrite(fileName+" Y edges.png", image);
	//	//cv::namedWindow("Y edges", cv::WINDOW_NORMAL);
	//	//imshow("Y edges", image);
	//	//waitKey();
	//}

}
