//writen by WangJin    2018/1/21

#include <iostream>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace cv;
using namespace std;

float dist2Point(int x1, int y1, int x2, int y2)
{
	return std::sqrt(double(x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

int main(int argc, char* argv[]){
	Mat img, img_gray, output_img;
	img = imread(argv[1]);
	if (img.empty()){
		cout << "the image can not been read" << endl;
		return -1;
	}
	cvtColor(img, img_gray, COLOR_BGR2GRAY);

	//detect the lines in the gray image
	vector<Vec4f> lines;
	Ptr<LineSegmentDetector> detector = createLineSegmentDetector(LSD_REFINE_ADV);
	detector->detect(img_gray, lines);

	//draw the lines to the image
	for (int i = 0; i < lines.size(); i++){
		if (dist2Point(lines[i][0], lines[i][1], lines[i][2], lines[i][3])>20)
			line(img, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 1, CV_AA);
	}

	//show the image with the detected lines
	imwrite("linesImg.jpg", img);
	imshow("lsd", img);
	waitKey(0);

	return 0;
}