#include "allheaderfile.hpp"



// Global Variable
string path = "/home/goutam/Desktop/project/img/";
string banknote[] = {"10-1", "10-2", "10-3", "10-4", "10-5", "10-6", "100-1",
                     "100-2", "100-3", "100-4", "100-5", "100-6", "1000-1", "1000-2", "1000-3", "1000-4",
                     "20-1", "20-2", "20-3", "20-4", "20-5", "20-6", "20-7",
                     "20-8", "50-1", "50-2", "50-3", "50-4", "50-5", "500-1",
                     "500-2", "500-3", "500-4",
                    };
int imgTemNum = 33;


int surfDetector(Mat img_1, Mat img_2, vector<DMatch> & matches, Mat &descriptors_1, Mat &descriptors_2)
{
	int minHessian = 400;

	SurfFeatureDetector detector( minHessian );

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector.detect( img_1, keypoints_1 );
	detector.detect( img_2, keypoints_2 );

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;


	extractor.compute( img_1, keypoints_1, descriptors_1 );
	extractor.compute( img_2, keypoints_2, descriptors_2 );

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	//std::vector< DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );
	return 0;
}

minMaxDistance findDistance(Mat descriptors_1, vector<DMatch> matches)
{
	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for ( int i = 0; i < descriptors_1.rows; i++ )
	{
		double dist = matches[i].distance;
		if ( dist < min_dist )
		{
			min_dist = dist;
		}
		if ( dist > max_dist )
		{
			max_dist = dist;
		}
	}
	minMaxDistance a;
	a.min = min_dist;
	a.max = max_dist;
	return a;
}
double goodMatches(Mat descriptors_1, vector<DMatch>matches, double min_dist)
{
	int counter = 0;
	double totalGoodMatch = 0.0;
	for ( int i = 0; i < descriptors_1.rows; i++ )
	{

		if ( matches[i].distance <= max(1.112233 * min_dist, 0.01) )
		{
			totalGoodMatch += matches[i].distance;
			counter++;
		}
	}
	//cout << "Counter = " << counter << endl;
	return totalGoodMatch / (double)counter;
}

string cvtString(string ss)
{
	string a = "";
	for (int i = 0; i < ss.size(); i++)
	{
		if (ss[i] == '-')
		{
			break;
		}
		a += ss[i];
	}
	a += " Taka";
	return a;
}
int main(int argc, char const *argv[])
{
	/*freopen("/home/jonyroy/Desktop/project/img/trainImage.txt", "w", stdout);
	for (int i = 1; i <= imgTemNum; i++)
	{
		string a = path + banknote[i - 1] + ".jpg";
		cout << a << endl;
	}*/
	double gm = 100.0;
	int ind = 0;
	Size size(1000, 600);
	string objpath = "/home/goutam/Desktop/project/img/obj.jpg";
	Mat objimg = imread( objpath, CV_LOAD_IMAGE_GRAYSCALE );
	//Mat objimg1;
	//resize(objimg, objimg1, size);
    cout<<"Processing........................"<<endl;
	for (int i = 0; i < imgTemNum; i++)
	{
		string scenepath = path + banknote[i] + ".jpg";
		Mat sceneimg = imread( scenepath, CV_LOAD_IMAGE_GRAYSCALE );
		//Mat sceneimg1;
		//resize(sceneimg, sceneimg1, size);
		vector<DMatch> matches;
		Mat descriptors_1, descriptors_2;
		surfDetector(objimg, sceneimg, matches, descriptors_1, descriptors_2);
		minMaxDistance a;
		a = findDistance(descriptors_1, matches);
		double m = goodMatches(descriptors_1, matches, a.min);
		cout << "Min=" << a.min << " Max=" << a.max <<  " " << cvtString(banknote[i]) << " Average= " << m << endl;

		if (a.min < gm)
		{
			gm = a.min;
			ind = i;
		}
	}
	string taka;
	taka = cvtString(banknote[ind]);
	cout << taka << endl;
	return 0;
}
