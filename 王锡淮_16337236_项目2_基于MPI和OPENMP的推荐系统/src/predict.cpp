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
#include <vector>
#include <omp.h>

using namespace std;

struct movieAndPreference
{
	int movieId;
	int movieRank;
	bool operator <(const movieAndPreference & other)
	{
		return movieRank < other.movieRank;
	}
	movieAndPreference(int id, int rank):
		movieId(id), movieRank(rank) {}
};

vector<struct movieAndPreference> predict(string preferenceFileName, int numberOfPrediction, int userId)
{
	ifstream preferenceFile(preferenceFileName, ios::in);
	string fileLine;
	string header;
	vector<struct movieAndPreference> ret;
	getline(preferenceFile, header);
	for (int i = 0; i < userId; ++i)
	{
		getline(preferenceFile, fileLine);
	}
	stringstream headerStream(header);
	stringstream perferenceStream(fileLine);
	int movieId;
	char tmp;
	double rating;
	headerStream >> tmp;
	perferenceStream >> rating;
	perferenceStream >> tmp;
	while (!headerStream.eof() && !perferenceStream.eof())
	{
		headerStream >> movieId;
		perferenceStream >> rating;
		ret.push_back(movieAndPreference(movieId, rating));
		headerStream >> tmp;
		perferenceStream >> tmp;
	}
	sort(ret.begin(), ret.end());
	return vector<struct movieAndPreference>(ret.begin(), ret.begin() + numberOfPrediction);
}

void readMovieIdMap(string movieMapFileName, map<int, string> & movieIdMap)
{
	stringstream ss;
	string fileLine;
	ifstream movieMapFile(movieMapFileName, ios::in);
	if (!movieMapFile.is_open()) {
		cout << "error: movieMapFile can not be open.\n";
	}
	string movieName;
	char tmpChar;
	int movieId;
	getline(movieMapFile, fileLine); // header
	while (getline(movieMapFile, fileLine)) {
		fileLine = fileLine.substr(0, fileLine.rfind(","));
		// cout << fileLine << endl;
		ss = stringstream(fileLine);
		ss >> movieId >> tmpChar;
		getline(ss, movieName);
		movieIdMap[movieId] = movieName;
		// cout << movieId << ": " << movieName << endl;
	}
}

int main(int argc, char const *argv[])
{
	if (argc != 3) {
		cout << "Usage: executable [NumberOfPrediction] [UserId]" << endl;
		return 0;
	}
	// string movieMapFileName = "D:\\dataset\\MoiveLens\\ml-latest-small\\ml-latest-small\\movies.csv";
	string movieMapFileName = "/mnt/d/dataset/MoiveLens/ml-latest-small/ml-latest-small/movies.csv";
	// string preferenceFileName = "D:\\dataset\\MoiveLens\\ml-latest-small\\ml-latest-small\\preference.csv";
	string preferenceFileName = "/mnt/d/dataset/MoiveLens/ml-latest-small/ml-latest-small/preference.csv";
	int numberOfPrediction = atoi(argv[1]);
	int userId = atoi(argv[2]);
	auto preds = predict(preferenceFileName, numberOfPrediction, userId);
	map<int, string> movieIdMap;
	readMovieIdMap(movieMapFileName, movieIdMap);
	for (size_t i = 0; i < preds.size(); ++i)
	{
		cout << i + 1 << ": " << movieIdMap[preds[i].movieId] << endl;
		// << " movieId: " << preds[i].movieId << endl;
	}
	return 0;
}