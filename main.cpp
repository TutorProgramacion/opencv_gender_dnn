#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <iostream>

using namespace cv;
using namespace std;

void getFaceRect(dnn::Net net, cv::Mat& frame, vector<cv::Rect>& rc, double conf_threshold)
{
	int frameHeight = frame.rows;
	int frameWidth = frame.cols;
	double inScaleFactor = 1.0;
	Size size = Size(300, 300);
	Scalar meanVal = Scalar(104, 117, 123);

	cv::Mat inputBlob;
	cv::dnn::blobFromImage(frame, inputBlob, inScaleFactor, size, meanVal, true, false);

	net.setInput(inputBlob, "data");

	cv::Mat detection = net.forward("detection_out");
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > conf_threshold)
		{
			int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
			int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
			int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
			int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

			rc.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));		
		}
	}
}

void main() {

	auto net_file = "caffe-models/caffe/res10_300x300_ssd_iter_140000_fp16.caffemodel";
	auto net_desc = "caffe-models/caffe/res10_300x300_ssd_iter_140000_fp16.prototxt";
	
	auto net_file1 = "caffe-models/caffe/gender_net.caffemodel";
	auto net_desc1 = "caffe-models/caffe/gender_deploy.prototxt";

	auto net = dnn::readNetFromCaffe(net_desc, net_file);
	auto net_gender = dnn::readNetFromCaffe(net_desc1, net_file1);

	VideoCapture capture;
	capture.open(0);

	int64 counter = 0;
	double elapsed = 0, fps = 0;
	int64 start = getTickCount();

	vector<string> genderList = { "Hombre", "Mujer" };
	Scalar meanVal = Scalar(104, 117, 123);

	while (capture.isOpened()) {

		Mat frame;
		capture >> frame;

		if (!frame.empty()) {

			vector<cv::Rect> rcFace;
			getFaceRect(net, frame, rcFace, 0.6);
						
			for (auto roi_face : rcFace) {

				auto face = frame(roi_face);

				auto blob = dnn::blobFromImage(face, 1, Size(227, 227), meanVal, false);
				net_gender.setInput(blob);

				vector<float> genderPreds = net_gender.forward();

				int max_index_gender = std::distance(genderPreds.begin(), max_element(genderPreds.begin(), genderPreds.end()));
				string gender = genderList[max_index_gender];

				cv::rectangle(frame, roi_face, cv::Scalar(0, 255, 255), 1, LINE_AA);
				cv::putText(frame, gender, roi_face.br() - cv::Point(roi_face.width - 8, 8), FONT_HERSHEY_SIMPLEX, 0.60, cv::Scalar(0, 0, 255), 1, LINE_AA);
			}
		}

		int64 delta = getTickCount() - start;
		
		double freq = getTickFrequency();
		double elap = delta / freq;

		counter += 1;

		if (elap > 1) 
		{		
			fps = counter / elap;
			elapsed = elap / fps;
			counter = 0;
			start = getTickCount();
		}

		String text = cv::format("Elapsed time: %.3f seg - FPS: %.2f", elapsed, fps);
		
		int baseline = 0;
		Size size = cv::getTextSize(text, FONT_HERSHEY_PLAIN, 1.0, 1, &baseline);
		
		cv::rectangle(frame, cv::Rect(5, 5, size.width + 10, size.height + 10), Scalar(0, 128, 128), FILLED, LINE_AA);
		cv::putText(frame, text, Point(10, size.height + 10), FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 255, 255), 1, LINE_AA);

		cv::imshow("face", frame);

		if (waitKey(1) == 27) break;
	}

	capture.release();
	
}