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
#include <mpi.h>


using namespace std;

// data structure
// rating is a map from movieId to ratings, for each user, movieMap is a map from movieId to movie name, movieIDMap is used to map raw id to sequence id.
void readDataset(string ratingFileName, string movieMappingFile, map<int, map<int, double> > & ratings, map<int, string> & movieMap, map<int, int> & movieIDMap, map<int, int> & rMovieIDMap)
{

	// reading movieMap
	stringstream ss;
	string fileLine;
	ifstream movieMapFile(movieMappingFile);
	string movieName;
	char tmpChar;
	int movieId;
	int seqId = 0;
	getline(movieMapFile, fileLine); // header
	// clock_t begin = clock();
	// cout << "\treading movieMap...";
	while (getline(movieMapFile, fileLine)) {
		ss = stringstream(fileLine);
		ss >> movieId >> tmpChar >> movieName;
		movieIDMap[movieId] = seqId++;
		rMovieIDMap[movieIDMap[movieId]] = movieId;
		movieMap[movieId] = movieName;
	}
	// cout << (clock() - begin) * 1.0 / CLOCKS_PER_SEC << "s" << endl;
	// cout << "movies: " << movieMap.size() << endl;

	// reading ratingFile
	ifstream ratingfile;
	int userId;
	double rating;
	ratingfile.open(ratingFileName, ios::in);
	getline(ratingfile, fileLine); // header
	// cout << fileLine;
	// begin = clock();
	// cout << "\treading ratings...";
	while (getline(ratingfile, fileLine)) {
		ss = stringstream(fileLine);
		ss >> userId >> tmpChar >> movieId >> tmpChar >> rating;
		ratings[userId][movieId] = rating;
	}
	// cout << (clock() - begin) * 1.0 / CLOCKS_PER_SEC << "s" << endl;
	// for (map<int, map<int, double>>::iterator iter = ratings.begin(); iter != ratings.end(); ++iter)
	// {
	// 	cout << "userId: " << iter->first << " rated movies: " << iter->second.size() << endl;
	// }
	// cout << "users: " << ratings.size() << endl;
	ratingfile.close();
	movieMapFile.close();
}

void calNeighAndCollab(map<int, map<int, double>> &ratings, map<int, int> & movieIDMap, int * neighbor, double ** collab, double **weights, int sizeOfItems, int begin, int taskEachNode)
{
	// cout << "\t calculating neighbors and collaboratives...";
	// clock_t begin = clock();
	for (map<int, map<int, double>>::iterator iter = ratings.begin(); iter != ratings.end(); ++iter)
	{
		// printf("user: %d\n", iter->first);
		for (map<int, double>::iterator subIter = iter->second.begin(); subIter != iter->second.end(); ++subIter)
		{
			neighbor[movieIDMap[subIter->first]]++;
			if ((movieIDMap[subIter->first] >= begin) && (movieIDMap[subIter->first] < begin + taskEachNode))
			{
				for (map<int, double>::iterator subsIter = iter->second.begin(); subsIter != iter->second.end(); ++subsIter)
				{
					if (subsIter->first != subIter->first)
						collab[movieIDMap[subIter->first] - begin][movieIDMap[subsIter->first] - begin] += 1 / log(1 + iter->second.size());

				}
			}
		}

	}
	// cout << (clock() - begin) * 1.0 / CLOCKS_PER_SEC << "s" << endl;
	// cout << "\t calculating weights...";
	// begin = clock();
	// for (int i = 0; i < sizeOfItems; ++i)
	// {
	// 	cout << neighbor[i] << " ";
	// }
	// cout << endl;
	// #pragma omp parallel for collapse(2)
	for (int i = 0; i < taskEachNode; ++i)
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
	// cout << (clock() - begin) * 1.0 / CLOCKS_PER_SEC << "s" << endl;
	// cout << "\t normalizing weights...";
	// begin = clock();
	double maxInLines[taskEachNode];
	// #pragma omp parallel for
	for (int i = 0; i < taskEachNode; ++i)
	{
		maxInLines[i] = *max_element(weights[i], weights[i] + sizeOfItems);
		// printf("max in line %d is %f\n", i, maxInLines[i]);
	}

	// #pragma omp parallel for collapse(2)
	for (int i = 0; i < taskEachNode; ++i)
		for (int j = 0; j < sizeOfItems; ++j)
		{
			if (maxInLines[i] != 0)
				weights[i][j] /= maxInLines[i];
			else {
				weights[i][j] = 0;
			}
		}
	// cout << (clock() - begin) * 1.0 / CLOCKS_PER_SEC << "s" << endl;
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
	cout << "\t calculating " << k << " best neighbors...";
	// #pragma omp parallel for collapse(2)
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
		// #pragma omp parallel for collapse(2)
		for (int i = 0; i < sizeOfItems; ++i)
		{
			for (int j = 0; j < k; ++j)
			{
				if (iter->second.find(rMovieIDMap[maxKIndex[i][j]]) != iter->second.end()) {
					preference[iter->first][i] += maxKWeight[i][j] * 1;
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
	preferenceFile.close();
}

int main(int argc, char *argv[])
{
	// parameters about MPI
	int nodesNum, rank;
	// int argcTmp = 0;
	// char **argvTmp;
	// argvTmp[0] = new char[strlen(argv[0]) + 1];
	// strncpy(argvTmp[0], argv[0], strlen(argv[0]));
	// MPI_Init(&argcTmp, &argvTmp);
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nodesNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	char processorName[MPI_MAX_PROCESSOR_NAME];
	int nameLen;
	MPI_Get_processor_name(processorName, &nameLen);
	if (nodesNum <= 1) {
		cout << "At least 2 nodes are needed." << endl;
	}
	if (rank == 0)
	{
		if (argc != 3) {
			cout << "Usage: executable [k] [numOfThreads]" << endl;
			return 0;
		}
	}
	int k = atoi(argv[1]);
	int numThreads = atoi(argv[2]);
	if (rank == 0) {
		omp_set_num_threads(numThreads * 2);
	} else {
		omp_set_num_threads(numThreads);
	}
	map<int, map<int, double>> ratings;
	map<int, string> movieMap;
	map<int, int> movieIDMap;
	map<int, int> rMovieIDMap;
	// cout << "reading files...\n";
	readDataset("./trainRatings.csv", "./movies.csv", ratings, movieMap, movieIDMap, rMovieIDMap);
	// readDataset("/mnt/d/dataset/MoiveLens/ml-latest-small/ml-latest-small/trainRatings.csv", "/mnt/d/dataset/MoiveLens/ml-latest-small/ml-latest-small/movies.csv", ratings, movieMap, movieIDMap, rMovieIDMap);
	// readDataset("D:\\dataset\\MoiveLens\\ml-latest-small\\ml-latest-small\\trainRatings.csv", "D:\\dataset\\MoiveLens\\ml-latest-small\\ml-latest-small\\movies.csv", ratings, movieMap, movieIDMap, rMovieIDMap);
	//buffers
	int sizeOfUsers, sizeOfItems;
	sizeOfUsers = ratings.size();
	sizeOfItems = movieMap.size();
	int taskEachNode = ceil(1.0 * sizeOfItems / (nodesNum - 1));
	double **weightsBuffer = new double*[taskEachNode];
	if (weightsBuffer == nullptr) {
		cout << "Memory for buffer requirement denyed.\n";
	}
	for (int i = 0; i < taskEachNode; i++)
	{
		weightsBuffer[i] = new double[sizeOfItems];
	}

	// cout << "node " << rank << " ready.\n";
	printf("Rank %d in processer %s is on\n", rank, processorName);
	if (rank == 0)
	{

		double **preference = new double * [sizeOfUsers + 1], **weights = new double * [sizeOfItems], *recvBuff = new double [sizeOfItems];
		if (preference == nullptr || weights == nullptr)
		{
			cout << "Memory requirement in master denyed.\n";
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
		// calNeighAndCollab(ratings, movieIDMap, neighbor, collab, weights, sizeOfItems);
		cout << "\t distributing tasks... " << nodesNum - 1 << " nodes and " << taskEachNode << " items per node.\n";
		// in fact, here needs no distribution, nodes can do their job according to their ranks.
		// TODO a process bar here
		for (int i = 1; i < nodesNum; ++i)
		{
			cout << "\t collecting data from node " << i << " ...\n";
			// MPI_Send(nullptr, 0, MPI_DOUBLE, i, sizeOfItems, MPI_COMM_WORLD);
			{
				if (i != nodesNum - 1) {
					for (int j = 0; j < taskEachNode; ++j)
					{
						MPI_Status status;
						MPI_Recv(recvBuff, sizeOfItems, MPI_DOUBLE, i, (i - 1)*taskEachNode + j, MPI_COMM_WORLD, &status);
						// copy by line
						memcpy(weights + ((i - 1)*taskEachNode + j)*sizeOfItems, recvBuff, sizeOfItems);
					}
				} else {
					int lines = sizeOfItems - taskEachNode * (nodesNum - 2);
					for (int j = 0; j < lines; ++j)
					{
						MPI_Status status;
						MPI_Recv(recvBuff, sizeOfItems, MPI_DOUBLE, i, (i - 1)*taskEachNode + j, MPI_COMM_WORLD, &status);
						memcpy(weights + ((i - 1)*taskEachNode + j)*sizeOfItems, recvBuff, sizeOfItems);
					}
				}
			}
		}
		// for (int i = 0; i < sizeOfItems; ++i)
		// {
		// 	cout << "man in line " << i << " is " << *max_element(weights[i], weights[i] + sizeOfItems) << endl;
		// }
		cout << "saving weights...\n";
		// saveWeights("D:\\dataset\\MoiveLens\\ml-latest-small\\ml-latest-small\\weights.csv", movieIDMap, weights, sizeOfItems);
		// saveWeights("/mnt/d/dataset/MoiveLens/ml-latest-small/ml-latest-small/weights.csv", movieIDMap, weights, sizeOfItems);
		saveWeights("./weights.csv", movieIDMap, weights, sizeOfItems);
		cout << "calculating preference...\n";
		calPreference(preference, weights, ratings, rMovieIDMap, k, sizeOfItems);
		cout << "saving preference...\n";
		// savePreference("D:\\dataset\\MoiveLens\\ml-latest-small\\ml-latest-small\\preference.csv", movieIDMap, rMovieIDMap, ratings, preference, sizeOfUsers, sizeOfItems);
		// savePreference("/mnt/d/dataset/MoiveLens/ml-latest-small/ml-latest-small/preference.csv", movieIDMap, rMovieIDMap, ratings, preference, sizeOfUsers, sizeOfItems);
		savePreference("./preference.csv", movieIDMap, rMovieIDMap, ratings, preference, sizeOfUsers, sizeOfItems);

		// for (int i = 0; i < sizeOfUsers + 1; i++)
		// {
		// 	delete [] preference[i];
		// }
		// delete [] preference;
		// for (int i = 0; i < sizeOfItems; i++)
		// {
		// 	delete [] weights[i];
		// }
		// delete [] weights;
		// delete [] recvBuff;

	} else {

		// cout << "nodes here" << endl;

		int * neighbor = new int[sizeOfItems];
		double **collab = new double * [sizeOfItems];
		if (neighbor == nullptr || collab == nullptr)
		{
			cout << "Memory requirement in master denyed.\n";
		}
		for (int i = 0; i < sizeOfItems; i++)
		{
			neighbor[i] = 0;
		}
		for (int i = 0; i < taskEachNode; i++)
		{
			collab[i] = new double[sizeOfItems];
			for (int j = 0; j < sizeOfItems; ++j)
			{
				collab[i][j] = 0;
			}
		}

		// calculating weights according to rank
		clock_t begin = clock();
		printf("Node %d calculating weights...", rank);
		calNeighAndCollab(ratings, movieIDMap, neighbor, collab, weightsBuffer, sizeOfItems, taskEachNode * (rank - 1), taskEachNode);
		// sending weights to master
		printf(" %f s\n", (clock() - begin) / (1.0 * CLOCKS_PER_SEC));
		// for (int i = 0; i < 1; ++i)
		// {
		// 	for (int j = 0; j < sizeOfItems; ++j)
		// 	{
		// 		cout << weightsBuffer[i][j] << " ";
		// 	}
		// 	cout << endl;
		// }
		int lines = taskEachNode;
		if (rank == nodesNum - 1)
		{
			lines = sizeOfItems - taskEachNode * (nodesNum - 2);
		}
		// MPI_Recv(nullptr, 0, MPI_DOUBLE, 0, sizeOfItems, MPI_COMM_WORLD, nullptr);
		for (int i = 0; i < lines; ++i)
		{
			MPI_Send(weightsBuffer + i * sizeOfItems, sizeOfItems, MPI_DOUBLE, 0, i + (rank - 1)*taskEachNode, MPI_COMM_WORLD);
			if (i == 0) {
				printf("node %d in %s sending weights to master...\n", rank, processorName);
			}
		}
		// delete [] neighbor;
		// for (int i = 0; i < taskEachNode; i++)
		// {
		// 	delete [] collab[i];
		// }
		// delete [] collab;
		ratings.clear();
		movieIDMap.clear();
		movieMap.clear();
		rMovieIDMap.clear();
	}

	// for (int i = 0; i < taskEachNode; i++)
	// {
	// 	delete [] weightsBuffer[i];
	// }
	// delete [] weightsBuffer;
	MPI_Finalize();

	return 0;
}