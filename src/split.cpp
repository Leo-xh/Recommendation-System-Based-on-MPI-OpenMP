#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <random>
#include <omp.h>

using namespace std;

double randomUniformDouble(unsigned int seed)
{
	static default_random_engine eng(seed);
	static uniform_real_distribution<double> randUniDou(0.0, 1.0);
	return randUniDou(eng);
}

void writeRatings(ofstream &file, map<int, map<int, double>> & ratings)
{
	file << "userId" << "," << "movieId" << "," << "rating" << endl;
	for (auto iter = ratings.begin(); iter != ratings.end(); ++iter)
	{
		for (auto subIter = iter->second.begin(); subIter != iter->second.end(); ++subIter) {
			file << iter->first << "," << subIter->first << "," << subIter->second << endl;
		}
	}
}

void split(int train, int test, string fileName, string trainFilename, string testFilename)
{
	map<int, map<int, double>> ratings;
	map<int, map<int, double>> trainRatings;
	map<int, map<int, double>> testRatings;
	double threshold = 1.0 * train / (train + test);
	stringstream ss;
	string fileLine;
	char tmpChar;
	int movieId;
	int userId;
	double rating;
	ifstream ratingfile;
	ofstream trainRatingsFile;
	ofstream testRatingsFile;
	trainRatingsFile.open(trainFilename, ios::out);
	testRatingsFile.open(testFilename, ios::out);
	ratingfile.open(fileName, ios::in);
	getline(ratingfile, fileLine); // header
	while (getline(ratingfile, fileLine)) {
		ss = stringstream(fileLine);
		ss >> userId >> tmpChar >> movieId >> tmpChar >> rating;
		ratings[userId][movieId] = rating;
	}
	for (auto iter = ratings.begin(); iter != ratings.end(); ++iter)
	{
		for (auto subIter = iter->second.begin(); subIter != iter->second.end(); ++subIter) {
			double randNum = randomUniformDouble(time(nullptr));
			if (randNum <= threshold)
			{
				trainRatings[iter->first][subIter->first] = subIter->second;
			} else {
				testRatings[iter->first][subIter->first] = subIter->second;
			}
		}
	}
	writeRatings(trainRatingsFile, trainRatings);
	writeRatings(testRatingsFile, testRatings);
	ratingfile.close();
	trainRatingsFile.close();
	testRatingsFile.close();
}

int main()
{
	split(8, 1, "D:\\dataset\\MoiveLens\\ml-latest-small\\ml-latest-small\\ratings.csv", "D:\\dataset\\MoiveLens\\ml-latest-small\\ml-latest-small\\trainRatings.csv", "D:\\dataset\\MoiveLens\\ml-latest-small\\ml-latest-small\\testRatings.csv");
	return 0;
}