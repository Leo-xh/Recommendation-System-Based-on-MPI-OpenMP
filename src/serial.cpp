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
#include <omp.h>


using namespace std;

// data structure
// rating is a map from movieId to ratings, for each user, movieMap is a map from movieId to movie name, movieIDMap is used to map raw id to sequence id.
void readDataset(string ratingFileName, string movieMappingFile, map<int, map<int, double>> & ratings, map<int, string> & movieMap, map<int, int> & movieIDMap, map<int, int> & rMovieIDMap)
{

	// reading movieMap
	stringstream ss;
	string fileLine;
	ifstream movieMapFile(movieMappingFile, ios::in);
	string movieName;
	char tmpChar;
	int movieId;
	int seqId = 0;
	getline(movieMapFile, fileLine); // header
	clock_t begin = clock();
	cout << "\treading movieMap...";
	while (getline(movieMapFile, fileLine)) {
		ss = stringstream(fileLine);
		ss >> movieId >> tmpChar >> movieName;
		movieIDMap[movieId] = seqId++;
		rMovieIDMap[movieIDMap[movieId]] = movieId;
		movieMap[movieId] = movieName;
	}
	cout << (clock() - begin) * 1.0 / CLOCKS_PER_SEC << "s" << endl;
	// cout << "movies: " << movieMap.size() << endl;

	// reading ratingFile
	ifstream ratingfile;
	int userId;
	double rating;
	ratingfile.open(ratingFileName, ios::in);
	getline(ratingfile, fileLine); // header
	begin = clock();
	cout << "\treading ratings...";
	while (getline(ratingfile, fileLine)) {
		ss = stringstream(fileLine);
		ss >> userId >> tmpChar >> movieId >> tmpChar >> rating;
		ratings[userId][movieId] = rating;
	}
	cout << (clock() - begin) * 1.0 / CLOCKS_PER_SEC << "s" << endl;
	// for (map<int, map<int, double>>::iterator iter = ratings.begin(); iter != ratings.end(); ++iter)
	// {
	// 	cout << "userId: " << iter->first << " rated movies: " << iter->second.size() << endl;
	// }
	// cout << "users: " << ratings.size() << endl;
	ratingfile.close();
	movieMapFile.close();
}

void calNeighAndCollab(map<int, map<int, double>> &ratings, map<int, int> & movieIDMap, int * neighbor, double ** collab, double ** weights, int sizeOfItems)
{
	cout << "\t calculating neighbors and collaboratives...";
	clock_t begin = clock();
	for (map<int, map<int, double>>::iterator iter = ratings.begin(); iter != ratings.end(); ++iter)
	{
		// printf("user: %d\n", iter->first);
		for (map<int, double>::iterator subIter = iter->second.begin(); subIter != iter->second.end(); ++subIter)
		{
			neighbor[movieIDMap[subIter->first]]++;
			for (map<int, double>::iterator subsIter = iter->second.begin(); subsIter != iter->second.end(); ++subsIter)
			{
				if (subsIter->first != subIter->first)
					collab[movieIDMap[subIter->first]][movieIDMap[subsIter->first]] += 1 / log(1 + iter->second.size());

			}
		}

	}
	cout << (clock() - begin) * 1.0 / CLOCKS_PER_SEC << "s" << endl;
	cout << "\t calculating weights...";
	begin = clock();
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < sizeOfItems; ++i)
	{
		for (int j = 0; j < sizeOfItems; ++j)
		{
			if (neighbor[i] != 0 && neighbor[j] != 0) {
				weights[i][j] = 1.0 * collab[i][j] / sqrt(neighbor[i] * neighbor[j]);
			} else {
				weights[i][j] = 0;
			}
		}
	}
	cout << (clock() - begin) * 1.0 / CLOCKS_PER_SEC << "s" << endl;
	cout << "\t normalizing weights...";
	begin = clock();
	double maxInLines[sizeOfItems];
	#pragma omp parallel for
	for (int i = 0; i < sizeOfItems; ++i)
	{
		maxInLines[i] = *max_element(weights[i], weights[i] + sizeOfItems);
	}

	#pragma omp parallel for collapse(2)
	for (int i = 0; i < sizeOfItems; ++i)
		for (int j = 0; j < sizeOfItems; ++j)
		{
			if (maxInLines[i] != 0)
				weights[i][j] /= maxInLines[i];
			else {
				weights[i][j] = 0;
			}
		}
	cout << (clock() - begin) * 1.0 / CLOCKS_PER_SEC << "s" << endl;
}

void saveWeights(string weightsFileName, map<int, int> & movieIDMap, double ** weights, int sizeOfItems)
{
	clock_t begin = clock();
	ofstream weightsFile(weightsFileName, ios::out);
	int rawIdOfItem[sizeOfItems], i = 0;
	if (weightsFile.is_open())
		cout << "Writing to file " << weightsFileName << endl;
	else {
		cout << "file " << weightsFileName << " can not be opened" << endl;
	}
	weightsFile << " " << ",";
	for (map<int, int>::iterator iter = movieIDMap.begin(); iter != movieIDMap.end(); ++iter)
	{
		weightsFile << iter->first << ",";
		rawIdOfItem[i++] = iter->first;
	}
	weightsFile << endl;
	for (int i = 0; i < sizeOfItems; ++i)
	{
		weightsFile << rawIdOfItem[i] << ",";
		for (int j = 0; j < sizeOfItems; ++j)
		{
			weightsFile << weights[i][j] << ",";
		}
		weightsFile << endl;
	}
	weightsFile.close();
	cout << (clock() - begin) * 1.0 / CLOCKS_PER_SEC << "s" << endl;
}



void calPreference(double** preference, double** weights, map<int, map<int, double>> &ratings, map<int, int> &rMovieIDMap, int k, int sizeOfItems)
{
	double maxKWeight[sizeOfItems][k];
	int maxKIndex[sizeOfItems][k];
	clock_t begin = clock();
	cout << "\t calculating K best neighbors...";
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < sizeOfItems; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			double * maxk  = max_element(weights[i], weights[i] + sizeOfItems);
			maxKWeight[i][j] = *maxk;
			maxKIndex[i][j] = maxk - weights[i];
			*maxk = -1;
		}
	}
	cout << (clock() - begin) * 1.0 / CLOCKS_PER_SEC << "s" << endl;
	cout << "\t calculating preferences...";
	begin = clock();
	for (map<int, map<int, double>>::iterator iter = ratings.begin(); iter != ratings.end(); ++iter)
	{
		#pragma omp parallel for collapse(2)
		for (int i = 0; i < sizeOfItems; ++i)
		{
			for (int j = 0; j < k; ++j)
			{
				if (iter->second.find(rMovieIDMap[maxKIndex[i][j]]) != iter->second.end()) {
					preference[iter->first][i] += maxKWeight[i][j] * iter->second[rMovieIDMap[maxKIndex[i][j]]];
				}
			}
		}
	}
	cout << (clock() - begin) * 1.0 / CLOCKS_PER_SEC << "s" << endl;
}

void savePreference(string preferenceFileName, map<int, int> & movieIDMap, map<int, int> & rMovieIDMap, map<int, map<int, double>> & ratings, double ** preference, int sizeOfUsers, int sizeOfItems)
{
	clock_t begin = clock();
	ofstream preferenceFile(preferenceFileName, ios::out);
	if (preferenceFile.is_open())
		cout << "Writing to file " << preferenceFileName << endl;
	else {
		cout << "file " << preferenceFileName << " can not be opened" << endl;
	}
	preferenceFile << " " << ",";
	for (map<int, int>::iterator iter = movieIDMap.begin(); iter != movieIDMap.end(); ++iter)
	{
		preferenceFile << iter->first << ",";
	}
	preferenceFile << endl;
	for (map<int, map<int, double>>::iterator iter = ratings.begin(); iter != ratings.end(); ++iter)
	{
		preferenceFile << iter->first << ",";
		for (int i = 0; i < sizeOfItems; ++i)
		{
			preferenceFile << preference[iter->first][i] << ",";
		}
		preferenceFile << endl;
	}
	cout << (clock() - begin) * 1.0 / CLOCKS_PER_SEC << "s" << endl;
}

int main(int argc, char const *argv[])
{
	if (argc != 3) {
		cout << "Usage: executable [k] [numOfThreads]" << endl;
		return 0;
	}
	int k = atoi(argv[1]);
	int numThreads = atoi(argv[2]);
	omp_set_num_threads(numThreads);
	map<int, map<int, double>> ratings;
	map<int, string> movieMap;
	map<int, int> movieIDMap;
	map<int, int> rMovieIDMap;
	cout << "reading files...\n";
	// readDataset("D:\\dataset\\MoiveLens\\ml-latest-small\\ml-latest-small\\trainRatings.csv", "D:\\dataset\\MoiveLens\\ml-latest-small\\ml-latest-small\\movies.csv", ratings, movieMap, movieIDMap, rMovieIDMap);
	readDataset("/mnt/d/dataset/MoiveLens/ml-latest-small/ml-latest-small/trainRatings.csv", "/mnt/d/dataset/MoiveLens/ml-latest-small/ml-latest-small/movies.csv", ratings, movieMap, movieIDMap, rMovieIDMap);
	//buffers
	int sizeOfUsers, sizeOfItems;
	sizeOfUsers = ratings.size();
	sizeOfItems = movieMap.size();
	int * neighbor = new int[sizeOfItems];
	double **preference = new double * [sizeOfUsers + 1], **weights = new double * [sizeOfItems], **collab = new double * [sizeOfItems];
	for (int i = 0; i < sizeOfItems; i++)
	{
		neighbor[i] = 0;
	}
	for (int i = 0; i < sizeOfItems; i++)
	{
		collab[i] = new double[sizeOfItems];
		for (int j = 0; j < sizeOfItems; ++j)
		{
			collab[i][j] = 0;
		}
	}
	for (int i = 0; i < sizeOfUsers + 1; i++)
	{
		preference[i] = new double[sizeOfItems];
		for (int j = 0; j < sizeOfItems; ++j)
		{
			preference[i][j] = 0;
		}
	}
	for (int i = 0; i < sizeOfItems; i++)
	{
		weights[i] = new double[sizeOfItems];
	}
	cout << "memory fine...\n";
	cout << "calculating weights...\n";
	calNeighAndCollab(ratings, movieIDMap, neighbor, collab, weights, sizeOfItems);
	cout << "saving weights...\n";
	saveWeights("/mnt/d/dataset/MoiveLens/ml-latest-small/ml-latest-small/weights.csv", movieIDMap, weights, sizeOfItems);
	cout << "calculating preference...\n";
	calPreference(preference, weights, ratings, rMovieIDMap, k, sizeOfItems);
	cout << "saving preference...\n";
	savePreference("/mnt/d/dataset/MoiveLens/ml-latest-small/ml-latest-small/preference.csv", movieIDMap, rMovieIDMap, ratings, preference, sizeOfUsers, sizeOfItems);

	delete [] neighbor;
	for (int i = 0; i < sizeOfItems; i++)
	{
		delete [] collab[i];
	}
	delete [] collab;
	for (int i = 0; i < sizeOfUsers; i++)
	{
		delete [] preference[i];
	}
	delete [] preference;
	for (int i = 0; i < sizeOfItems; i++)
	{
		delete [] weights[i];
	}
	delete [] weights;
	return 0;
}