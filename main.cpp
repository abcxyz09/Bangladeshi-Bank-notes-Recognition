#include <bits/stdc++.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <stdlib.h>
using namespace std;
using namespace cv;

// Global Variables
Size size(700,350);
char ImgNames[] = "/home/goutam/Desktop/banknote/trainImage.txt";
char queryName[] = "/home/goutam/Desktop/banknote/queryImgnames.txt";
string audioPath = "/home/goutam/Desktop/banknote/audio/";
string audio[] = {"10", "20", "50", "100", "500", "1000"};
string audioExt = ".wav";

int readImgNames( vector<string>& trainImgnames,char imgpath[] )
{
	trainImgnames.clear();
	ifstream file;
	file.open(imgpath);

	while ( !file.eof() )
	{
		string str;
		getline( file, str );
		if ( str.empty() )
		{
			break;
		}
		trainImgnames.push_back(str);
	}
	file.close();
	return 0;
}

int findTaka(string s)
{
	string note[] = {"10", "20", "50", "100", "500", "1000"};
	for (int i = 5; i >= 0; i--)
	{
		size_t found = s.find(note[i]);
		if (found != std::string::npos)
		{
			return i;
		}
	}
	return -1;
}

bool readTrainImgs(vector <Mat>& trainImages, vector<string>& trainImgNames )
{
	cout << "Reading the images..." << endl;

	readImgNames(trainImgNames,ImgNames);

	if ( trainImgNames.empty() )
	{
		cout << "Train image filenames can not be read." << endl << ">" << endl;
		return false;
	}
	int readImageCount = 0;
	for ( int i = 0; i < trainImgNames.size(); i++ )
	{
		string filename = trainImgNames[i];
		Mat img = imread( filename, CV_LOAD_IMAGE_GRAYSCALE );
		if ( img.empty() )
		{
			cout << "Train image " << filename << " can not be read." << endl;
			continue;
		}
		readImageCount++;
		Mat img1;
		resize(img,img1,size);
		trainImages.push_back( img );
	}
	if ( !readImageCount )
	{
		cout << "All train images can not be read." << endl << ">" << endl;
		return false;
	}
	else
	{
		cout << readImageCount << " train images were read." << endl;
	}
	return true;
}


bool createDetectorDescriptorMatcher( Ptr<FeatureDetector>& featureDetector,
                                      Ptr<DescriptorExtractor>& descriptorExtractor,
                                      Ptr<DescriptorMatcher>& descriptorMatcher )
{
	cout << "Creating feature detector, descriptor extractor and descriptor matcher ..." << endl;
	featureDetector = FeatureDetector::create("SURF");
	descriptorExtractor = DescriptorExtractor::create("SURF");
	descriptorMatcher = DescriptorMatcher::create("FlannBased");
	cout << "  FeatureDetector Type is SURF\n";
	cout << "  DescriptorExtractor Type is SURF\n";
	cout << "  DescriptorMatcher Type is FlannBased\n";
}



int detectKeypointsTrainImg(vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints,
                            Ptr<FeatureDetector> & featureDetector )
{
	cout << endl << " Extracting keypoints from images..." << endl;
	TickMeter tm;
	tm.start();
	featureDetector->detect( trainImages, trainKeypoints );
	tm.stop();
	double trainkeytime = tm.getTimeMilli();
	//cout << "Detecting KeyPoint Train Images = " << trainkeytime << " ms\n";
	return 0;
}


int processingQueryImg(Mat & queryImages, vector<KeyPoint> & queryKeypoints,
                       Mat & queryDescriptors , Ptr<FeatureDetector> & featureDetector,
                       Ptr<DescriptorExtractor> & descriptorExtractor )
{
	//cout << endl << " Extracting keypoints from query image..." << endl;
	TickMeter tm;
	tm.start();
	featureDetector->detect( queryImages, queryKeypoints );
	descriptorExtractor->compute( queryImages, queryKeypoints, queryDescriptors );
	tm.stop();
	double queryprocessingtime = tm.getTimeMilli();
	//cout << "Query Processing Time = " << queryprocessingtime << " ms\n";
	return 0;
}



void computeDescriptorTrainImgs(vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints, vector<Mat>& trainDescriptors,
                                Ptr<DescriptorExtractor> & descriptorExtractor )
{
	cout << "Computing descriptors for keypoints........" << endl;
	TickMeter tm;
	tm.start();
	descriptorExtractor->compute( trainImages, trainKeypoints, trainDescriptors );
	tm.stop();
	double computeTime = tm.getTimeMilli();
	cout << "Train Image Extracting Time = " << computeTime << " ms\n";
	int totalTrainDesc = 0;
	int counter = 0;
	for ( vector<Mat>::const_iterator tdIter = trainDescriptors.begin(); tdIter != trainDescriptors.end(); tdIter++ )
	{
		totalTrainDesc += tdIter->rows;
		counter++;
	}

	//cout << counter << " Total train descriptors count: " << totalTrainDesc << endl;
}


void trainDatabaseImg(vector<Mat>& trainDescriptors,Ptr<DescriptorMatcher>& descriptorMatcher)
{
	descriptorMatcher->add( trainDescriptors );
	descriptorMatcher->train();
}
void matchDescriptors(Mat& queryDescriptors, vector<Mat>& trainDescriptors,
                      vector<DMatch>& matches, Ptr<DescriptorMatcher>& descriptorMatcher )
{
	//cout << "Set train descriptors collection in the matcher and match query descriptors to them..." << endl;
	//TickMeter tm;

	//tm.start();
	descriptorMatcher->match( queryDescriptors, matches );
	//tm.stop();
	//double matchTime = tm.getTimeMilli();

	//CV_Assert( queryDescriptors.rows == (int)matches.size() || matches.empty() );

	//cout << "Number of matches: " << matches.size() << endl;
	//cout << "Build time: " << buildTime << " ms; Match time: " << matchTime << " ms" << endl;
}

int main(int argc, char const *argv[])
{

	cv::initModule_nonfree();
	vector <Mat> trainImgs, trainDescriptors;
	vector<string> trainImgNames;
	readTrainImgs(trainImgs, trainImgNames);


	Ptr<FeatureDetector> featureDetector;
	Ptr<DescriptorExtractor> descriptorExtractor;
	Ptr<DescriptorMatcher> descriptorMatcher;
	vector<vector<KeyPoint> >trainKeypoints;

    TickMeter tmm;
    tmm.start();
	createDetectorDescriptorMatcher(featureDetector, descriptorExtractor, descriptorMatcher);
	detectKeypointsTrainImg(trainImgs, trainKeypoints, featureDetector);
	computeDescriptorTrainImgs(trainImgs, trainKeypoints, trainDescriptors, descriptorExtractor);
	trainDatabaseImg(trainDescriptors,descriptorMatcher);
	tmm.stop();
	cout<<"Database Setup time: "<<tmm.getTimeMilli()/1000.0<<" s"<<endl;
	//vector<string> queryImagepath;
	//readImgNames(queryImagepath,queryName);
	for (int i = 0; i < 100 ; i++)
	{

		string ob,objpath1,objpath; //= queryImagepath[i];
		objpath1="/home/goutam/Desktop/banknote/img/query/query-";
		cout<<"Please Enter Query Image Name:";
		cin>>ob;
		if(ob=="q")
			return 0;
		objpath1=objpath1+ob+".jpg";
		//cout << "Ok" << endl;
		TickMeter tm;
		tm.start();
		Mat query = imread( objpath1, CV_LOAD_IMAGE_GRAYSCALE );
		Mat query1;
		resize(query,query1,size);
		if (query.empty())
		{
			cout << "Image can't read\n";
			cout << objpath << endl;
		}
		vector<KeyPoint> queryKeypoints;
		Mat queryDescriptors;
		vector<DMatch> matches, match2;
		processingQueryImg(query1, queryKeypoints, queryDescriptors, featureDetector, descriptorExtractor);
		matchDescriptors(queryDescriptors, trainDescriptors, matches, descriptorMatcher);
		int counter[50] = {0};
		double mi = 100.0;
		for (int j = 0; j < matches.size(); j++)
		{
			if (matches[j].distance < mi)
			{
				mi = matches[j].distance;
			}
			counter[matches[j].imgIdx]++;
			//cout << "Dist= " << matches[i].distance << " Img= " << matches[i].imgIdx << " ";
			//cout << "Query= " << matches[i].queryIdx << " Train= " << matches[i].trainIdx << endl;
		}
		for (int j = 0; j < matches.size(); j++)
		{
			if ( mi * 1.112233 > matches[j].distance )
			{
				match2.push_back(matches[j]);
			}
		}
		for (int j = 0; j < match2.size(); j++)
		{
			//counter[match2[j].imgIdx]++;
		}
		//cout << "Minimum Distances = " << mi << endl;
		/*for (int i = 0; i < 25; i++)
		{
			cout << counter[i] << " ";
		}*/
		cout << endl;
		int ma = -1, index = -1;
		for (int j = 0; j <= 24; j++)
		{
			if (ma < counter[j])
			{
				ma = counter[j];
				index = j;
			}
		}
		tm.stop();
		cout<<tm.getTimeMilli()<<" ms"<<endl;
		cout << objpath1 << endl;
		if (index != -1)
		{
			int taka = findTaka(trainImgNames[index]);

			if (taka >= 0)
			{
				string b = audioPath + audio[taka] + audioExt;
				//cout << b << endl;
				string cnt = "canberra-gtk-play -f " + b;
				system(cnt.c_str());
			}
		}
		else
		{
			cout << "Not Found";
		}
		cout << endl << endl;
	}
	return 0;
}
