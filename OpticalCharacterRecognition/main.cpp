/*************************************************************************************************************************
 Copyright GEORGAKIS GEORGIOS, CHRISTODOULOU SAVVAS (c) 2013, ggeorgak@masonlive.gmu.edu, savvaschris@yahoo.com  

 Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted,
 provided that the above copyright notice and this permission notice appear in all copies.

 THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
 SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE 
 AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES 
 WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, 
 NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

*************************************************************************************************************************/

#include <stdio.h>
#include <iostream>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\ml\ml.hpp"
 
using namespace std; 
using namespace cv;

Mat GetBoundingBoxImage(Rect rect, Mat gray_image);
Rect boundingRect(vector<Point> & points);
vector<float> FindFeatures(Mat img);
Rect GetBoundingRect(Mat gray);
void Knn(Mat& trainData, Mat& trainClasses, Mat& testSamples, int k);
void NormalBayes(Mat& trainData, Mat& trainClasses, Mat& testSamples);
void SVMachine(Mat& trainData, Mat& trainClasses, Mat& testSamples);
int FindCenterMass(Mat img);
Mat PreProcessing(Mat src_image, int scale);


int main( int argc, char** argv ){
	int scale = 300;
	int trainCounter=0;
	int trainingNumber = 80, testNumber = 20, featureNumber = 5;
	Mat trainData (trainingNumber, featureNumber, CV_32FC1);
	Mat trainClasses (trainingNumber, 1, CV_32FC1);
	Mat testSamples(testNumber, featureNumber, CV_32FC1);
	Mat src_image;

	/*Collect the training data. out for is classes and in for is samples*/
	for (int Class=0; Class<=9; Class++){
		for (int i=0; i<=7; i++){
			String number, type;
			ostringstream conv_n, conv_t;
			conv_n << i;
			conv_t << Class;
			number = conv_n.str();
			type = conv_t.str();
			src_image = imread("samples/"+type+"/"+number+".jpg");

			/*Do the necessary preprocessing to the image*/
			Mat BBimageScaled = PreProcessing(src_image, scale);

			/*Extract the features of the preprocessed image*/
			vector<float> feat = FindFeatures(BBimageScaled);
			//cout << "Class:" << Class << "  " << feat.at(0) << "  " << feat.at(1) << "  " << feat.at(2) << "  " << feat.at(3) << "  " << feat.at(4) << endl;

			/*Fill the data from the training samples and label the classes*/
			trainClasses.at<float>(trainCounter) = Class;
			for (int w=0; w<trainData.cols; w++){
				trainData.at<float>(trainCounter, w) = feat.at(w);
			}
			trainCounter++;
		}
	}

	/*Collect testing data*/
	for (int samples=0; samples<testNumber; samples++){
		String test_sample;
		ostringstream conv2;
		conv2 << samples;
		test_sample = conv2.str();
		Mat test_image = imread("test/"+test_sample+".jpg");
		Mat process_test = PreProcessing(test_image, scale);
		vector<float> test_feat = FindFeatures(process_test);

		for (int w=0; w<testSamples.cols; w++)
				testSamples.at<float>(samples, w) = test_feat.at(w);
	}

	//cout << "Test features" << endl;
	//for (int i=0; i<testSamples.rows; i++){
	//	cout << "Test:" << i << "  " << testSamples.at<float>(i, 0) << "  " << testSamples.at<float>(i, 1) << "  " << testSamples.at<float>(i, 2) << "  " << testSamples.at<float>(i, 3) << "  " << testSamples.at<float>(i, 4) << endl;
	//}

	cout << endl;
	cout << "Test Sequence: 0  0  1  1  2  2  3  3  4  4  5  5  6  6  7  7  8  8  9  9" << endl;
	/*K-nearest neighbors algorithm*/
	Knn(trainData, trainClasses, testSamples, 1);

	/*Normal Bayes classifier*/
	NormalBayes(trainData, trainClasses, testSamples);

	/*Support Vector Machine*/
	SVMachine(trainData, trainClasses, testSamples);

	cout << endl;

	waitKey(0);
	return 0;

}

void SVMachine(Mat& trainData, Mat& trainClasses, Mat& testSamples){

	// Set up SVM's parameters
    //CvSVMParams params;
    //params.svm_type    = CvSVM::C_SVC;
    //params.kernel_type = CvSVM::LINEAR;
    //params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	
	CvSVMParams params;
	params.svm_type    = SVM::C_SVC;
	params.C              = 0.1;
	params.kernel_type = SVM::LINEAR;
	params.term_crit   = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-9);

    // Train the SVM
    CvSVM SVM;
	try{
		SVM.train(trainData, trainClasses, Mat(), Mat(), params);
	}
	catch(exception& e){
		cout<<e.what();
		
	}
	cout << "SVM result:    ";
	for (int i=0; i<testSamples.rows; i++){
		Mat sample_row = testSamples.row(i);
		float response = SVM.predict(sample_row);
		cout << response << "  ";
	}

}

void NormalBayes(Mat& trainData, Mat& trainClasses, Mat& testSamples){

	CvNormalBayesClassifier bayes(trainData, trainClasses);
	Mat res(testSamples.rows, 1, CV_32FC1);
	bayes.predict(testSamples, &res);
	cout << "Bayes Result: " << res << endl;
}

void Knn(Mat& trainData, Mat& trainClasses, Mat& testSamples, int k){

	CvKNearest knn(trainData, trainClasses, Mat(), false, k);
	Mat res(testSamples.rows, 1, CV_32FC1);
	knn.find_nearest(testSamples, k, &res);
	cout << "KNN Result:   " << res << endl;
}

vector<float> FindFeatures(Mat img){
	int value = 0, pixel_count = 0;
	vector<float> x, y, features;
	float diameter=0;

	for (int i=0; i<img.rows; i++){
		for (int j=0; j<img.cols; j++){
			value = (float)img.at<uchar>(i,j);
			if (value==0){ 
				x.push_back(i);
				y.push_back(j);
				pixel_count++;

				if (i==j) diameter++;
			}
		}
	}

	int sum_x = 0, sum_y = 0;
	for (int i=0; i<x.size(); i++){
		sum_x = sum_x + x.at(i);
		sum_y = sum_y + y.at(i);
	}
	float average_x = sum_x / x.size();
	float average_y = sum_y / x.size();

	int max = FindCenterMass(img);

	features.push_back(pixel_count);
	features.push_back(average_x);
	features.push_back(average_y);
	features.push_back(diameter);
	features.push_back(max);

	return features;
}

Mat PreProcessing(Mat src_image, int scale){
	Mat gray, gray_image;
	/*Gray image twice, one for contour and one for BB*/
	cvtColor(src_image, gray, CV_BGR2GRAY);
	cvtColor(src_image, gray_image, CV_BGR2GRAY);

	/*Blur images*/
	GaussianBlur(gray_image, gray_image, Size(15,15), 7);
	GaussianBlur(gray, gray, Size(15,15), 7);
	//imshow("blur", gray);

	/*Convert images to binary*/
	threshold(gray_image, gray_image, 180, 255, THRESH_BINARY);
	threshold(gray, gray, 180, 255, THRESH_BINARY);
	//imshow("binary", gray);

	/*Find bounding box for number with the use of contours*/
	Rect rect = GetBoundingRect(gray);

	/*Separate the image thats in the Bounding Box*/
	Mat BBimage = GetBoundingBoxImage(rect, gray_image);
	//imshow("Bounding Box image", BBimage);

	/*Draw the bounding box in the src image*/
	rectangle(src_image, rect, CV_RGB(255,0,0), 1);
	//imshow("image", src_image);

	/*Resize the image so that all numbers have the same size*/
	Mat BBimageScaled;
	resize(BBimage, BBimageScaled, Size(scale,scale), 0, 0, INTER_LINEAR);
	//imshow("Resized Image", BBimageScaled);

	return BBimageScaled;
}


Rect GetBoundingRect(Mat gray){

	vector<vector<Point>> v;
	findContours(gray, v, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	int area = 0;
	int idx;
	for(int i=0; i<v.size();i++) {
		if(area < v[i].size()) {
			idx = i; 
            area = v[i].size();
        }
	}

	Rect rect = boundingRect(v[idx-1]);
	/*Point pt1, pt2;
	pt1.x = rect.x;
	pt1.y = rect.y;
	pt2.x = rect.x + rect.width;
	pt2.y = rect.y + rect.height;*/
	return rect;
}

Mat GetBoundingBoxImage(Rect rect, Mat gray_image){
	//cout << "X: " << rect.x << "  Y: " << rect.y << endl;
	//cout << "Width: " << rect.width << "  Height: " << rect.height << endl;

	Mat new_image = Mat::zeros(rect.height, rect.width, CV_8UC1);
	//cout << "Rows: " << new_image.rows << "  Cols: " << new_image.cols << endl;

	int x_point = rect.x;
	int y_point = rect.y;	
	for (int i=0; i<new_image.rows; i++){
		for (int j=0; j<new_image.cols; j++){	
			new_image.at<uchar>(i, j) = (int)gray_image.at<uchar>(y_point, x_point);
			x_point++;
		}
		y_point++;
		x_point = rect.x;
	}
	return new_image;
}


Rect boundingRect (vector<Point> & points){
	// Points initialization
	Point min (numeric_limits<int>::max(), numeric_limits<int>::max() );
	Point max (numeric_limits<int>::min(), numeric_limits<int>::min() );
 
	// Getting bounding box coordinates
	for ( unsigned int i=0;i<points.size();i++ ) {
		if ( points[i].x > max.x ) max.x=points[i].x;
		if ( points[i].x < min.x ) min.x=points[i].x;
 
		if ( points[i].y > max.y ) max.y=points[i].y;
		if ( points[i].y < min.y ) min.y=points[i].y;
	}
 
	return Rect ( min.x,min.y,max.x-min.x+1,max.y-min.y+1 );
} 


int FindCenterMass(Mat img){
	try{
	float count[4] = {0,0,0,0};
	int quad = 0;
	for(int i=0; i<img.rows/2; i++)
	{
		for(int j=0; j<img.cols/2; j++)
		{
			if((int)img.at<uchar>(i,j)==0)
			{
				count[quad]++;
			}
		}
	}
	quad++;
	for(int i=img.rows/2; i<img.rows; i++)
	{
		for(int j=0; j<img.cols/2; j++)
		{
			if((int)img.at<uchar>(i,j)==0)
			{
				count[quad]++;
			}
		}
	}
	quad++;
	for(int i=0; i<img.rows/2; i++)
	{
		for(int j=img.cols/2; j<img.cols; j++)
		{
			if((int)img.at<uchar>(i,j)==0)
			{
				count[quad]++;
			}
		}
	}
	quad++;
	for(int i=img.rows/2; i<img.rows; i++)
	{
		for(int j=img.cols/2; j<img.cols; j++)
		{
			if((int)img.at<uchar>(i,j)==0)
			{
				count[quad]++;
			}
		}
	}

	int max=count[0];
	float index=0;
	for(int i=1;i<4;i++){
		if(count[i]>max){
			max=count[i];
			index = i;
		}
	}

	return index;

	}catch(exception& e){
		cout << e.what();
	}

	
}