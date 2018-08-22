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
	cout << fileLine;
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

void calNeighAndCollab(map<int, map<int, double>> &ratings, map<int, int> & rMovieIDMap, int * neighbor, double ** collab, double **weights, int sizeOfItems, int begin, int taskEachNode)
{

	// clock_t begin = clock();
	// calculating the averages that each user gives
	double aveUser[ratings.size()];
	for (map<int, map<int, double>>::iterator iter = ratings.begin(); iter != ratings.end(); ++iter)
	{
		aveUser[iter->first] = 0;
		for (map<int, double>::iterator subIter = iter->second.begin(); subIter != iter->second.end(); ++subIter)
		{
			aveUser[iter->first] += subIter->second;
		}
		aveUser[iter->first] /= iter->second.size();
	}
	// adjust cosine similarity
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	#pragma omp parallel for
	for (int i = begin; i < begin + taskEachNode; ++i)
	{
		double tmpSumI = 0;
		for (map<int, map<int, double>>::iterator iter = ratings.begin(); iter != ratings.end(); ++iter)
		{
			// printf("user: %d\n", iter->first);
			if (iter->second.count(rMovieIDMap[i]) > 0)
			{
				tmpSumI += pow(iter->second[rMovieIDMap[i]] - aveUser[iter->first], 2);
			}
		}
		#pragma omp parallel for
		for (int j = 0; j < sizeOfItems; ++j)
		{
			double tmpSumJ = 0;
			for (map<int, map<int, double>>::iterator iter = ratings.begin(); iter != ratings.end(); ++iter)
			{
				// printf("user: %d\n", iter->first);
				if (iter->second.count(rMovieIDMap[j]) > 0)
				{
					tmpSumJ += pow(iter->second[rMovieIDMap[j]] - aveUser[iter->first], 2);
				}
			}

			double dem = sqrt(tmpSumI * tmpSumJ);
			double num = 0;
			for (map<int, map<int, double>>::iterator iter = ratings.begin(); iter != ratings.end(); ++iter)
			{
				if (iter->second.count(rMovieIDMap[i]) > 0 && iter->second.count(rMovieIDMap[j]) > 0)
				{
					num += (iter->second[rMovieIDMap[i]] - aveUser[iter->first]) * (iter->second[rMovieIDMap[j]] - aveUser[iter->first]);
				}
			}
			weights[i - begin][j] = num / dem;
		}
	}


	// normlize the weights
	double maxInLines[taskEachNode];
	#pragma omp parallel for
	for (int i = 0; i < taskEachNode; ++i)
	{
		maxInLines[i] = *max_element(weights[i], weights[i] + sizeOfItems);
	}

	#pragma omp parallel for collapse(2)
	for (int i = 0; i < taskEachNode; ++i)
		for (int j = 0; j < sizeOfItems; ++j)
		{
			if (maxInLines[i] != 0)
				weights[i][j] /= maxInLines[i];
			else {
				weights[i][j] = 0;
			}
		}
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


void calAveRatingForItem(map<int, map<int, double>> & ratings, map<int, int> &movieIDMap, double * aveItem, int sizeOfItems)
// aveItem is initialized
{
	int countOfItemRatings[sizeOfItems];
	#pragma omp parallel for
	for (int i = 0; i < sizeOfItems; ++i)
	{
		countOfItemRatings[i] = 0;
	}
	for (map<int, map<int, double>>::iterator iter = ratings.begin(); iter != ratings.end(); ++iter)
	{
		for (map<int, double>::iterator subIter = iter->second.begin(); subIter != iter->second.end(); ++subIter)
		{
			aveItem[movieIDMap[subIter->first]] += subIter->second;
			countOfItemRatings[movieIDMap[subIter->first]]++;
		}
	}
	#pragma omp parallel for
	for (int i = 0; i < sizeOfItems; ++i)
	{
		aveItem[i] /= countOfItemRatings[i];
	}
}


void calPreference(double** preference, double** weights, double * aveItem, map<int, map<int, double>> &ratings, map<int, int> &rMovieIDMap, int k, int sizeOfItems)
{
	double maxKWeight[sizeOfItems][k];
	int maxKIndex[sizeOfItems][k];
	clock_t begin = clock();
	cout << "\t calculating " << k << " best neighbors...";
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
		#pragma omp parallel for
		for (int i = 0; i < sizeOfItems; ++i)
		{
			double num = 0;
			double dem = 0;
			for (int j = 0; j < k; ++j)
			{
				if (iter->second.find(rMovieIDMap[maxKIndex[i][j]]) != iter->second.end()) {
					num += maxKWeight[i][j] * (iter->second[rMovieIDMap[maxKIndex[i][j]]] - aveItem[i]);
					dem += abs(maxKWeight[i][j]);
				}
			}
			preference[iter->first][i] = aveItem[i] + num / dem;
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
	// jobs distributed strategy
	// 1 node is to calculate the average score each item get, others are to calculate the weights.

	// parameters about MPI
	int nodesNum, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nodesNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (nodesNum <= 2) {
		cout << "At least 3 nodes are needed." << endl;
		fflush(stdout);
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
	readDataset("/mnt/d/dataset/MoiveLens/ml-latest-small/ml-latest-small/trainRatings.csv", "/mnt/d/dataset/MoiveLens/ml-latest-small/ml-latest-small/movies.csv", ratings, movieMap, movieIDMap, rMovieIDMap);
	// readDataset("D:\\dataset\\MoiveLens\\ml-latest-small\\ml-latest-small\\trainRatings.csv", "D:\\dataset\\MoiveLens\\ml-latest-small\\ml-latest-small\\movies.csv", ratings, movieMap, movieIDMap, rMovieIDMap);
	//buffers
	int sizeOfUsers, sizeOfItems;
	sizeOfUsers = ratings.size();
	sizeOfItems = movieMap.size();
	int taskEachNode = ceil(1.0 * sizeOfItems / (nodesNum - 2));
	double **weightsBuffer = new double*[taskEachNode];
	if (weightsBuffer == nullptr) {
		cout << "Memory for buffer requirement denyed.\n";
	}
	for (int i = 0; i < taskEachNode; i++)
	{
		weightsBuffer[i] = new double[sizeOfItems];
	}

	if (rank == 0)
	{
		double **preference = new double * [sizeOfUsers + 1], **weights = new double * [sizeOfItems], *recvBuff = new double [sizeOfItems], *aveItem = new double[sizeOfItems];
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
		cout << "\t distributing tasks... node " << nodesNum - 1 << " calculating the average ratings of items with "  << nodesNum - 2 << " nodes calculating weights, " << taskEachNode << " items per node." << endl;
		MPI_Status status;
		cout << "\t collecting data from node " << nodesNum - 1 << " ...\n";
		MPI_Recv(aveItem, sizeOfItems, MPI_DOUBLE, nodesNum - 1, 0, MPI_COMM_WORLD, &status);
		// in fact, here needs no distribution, nodes can do their job according to their ranks.
		// TODO a process bar here
		for (int i = 1; i < nodesNum - 1; ++i)
		{
			cout << "\t collecting data from node " << i << " ...\n";
			// MPI_Send(nullptr, 0, MPI_DOUBLE, i, sizeOfItems, MPI_COMM_WORLD);
			{
				if (i < nodesNum - 2) {
					for (int j = 0; j < taskEachNode; ++j)
					{
						MPI_Recv(recvBuff, sizeOfItems, MPI_DOUBLE, i, (i - 1)*taskEachNode + j, MPI_COMM_WORLD, &status);
						// copy by line
						memcpy(weights + ((i - 1)*taskEachNode + j)*sizeOfItems, recvBuff, sizeOfItems);
					}
				} else {
					int lines = sizeOfItems - taskEachNode * (nodesNum - 3);
					for (int j = 0; j < lines; ++j)
					{
						MPI_Status status;
						MPI_Recv(recvBuff, sizeOfItems, MPI_DOUBLE, i, (i - 1)*taskEachNode + j, MPI_COMM_WORLD, &status);
						memcpy(weights + ((i - 1)*taskEachNode + j)*sizeOfItems, recvBuff, sizeOfItems);
					}
				}
			}
		}

		cout << "saving weights...\n";
		// saveWeights("D:\\dataset\\MoiveLens\\ml-latest-small\\ml-latest-small\\weights.csv", movieIDMap, weights, sizeOfItems);
		saveWeights("/mnt/d/dataset/MoiveLens/ml-latest-small/ml-latest-small/weights.csv", movieIDMap, weights, sizeOfItems);
		cout << "calculating preference...\n";
		calPreference(preference, weights, aveItem , ratings, rMovieIDMap, k, sizeOfItems);
		cout << "saving preference...\n";
		// savePreference("D:\\dataset\\MoiveLens\\ml-latest-small\\ml-latest-small\\preference.csv", movieIDMap, rMovieIDMap, ratings, preference, sizeOfUsers, sizeOfItems);
		savePreference("/mnt/d/dataset/MoiveLens/ml-latest-small/ml-latest-small/preference.csv", movieIDMap, rMovieIDMap, ratings, preference, sizeOfUsers, sizeOfItems);

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
	} else if (rank == nodesNum - 1) {
		double* aveItem = new double[sizeOfItems];
		for (int i = 0; i < sizeOfItems; ++i)
		{
			aveItem[i] = 0;
		}
		char processorName[MPI_MAX_PROCESSOR_NAME];
		int nameLen;
		MPI_Get_processor_name(processorName, &nameLen);
		printf("node %d in %s calculating average of item ratings...\n", rank, processorName);
		calAveRatingForItem(ratings, movieIDMap, aveItem, sizeOfItems);
		printf("node %d in %s sending average of item ratings to master...\n", rank, processorName);
		MPI_Send(aveItem, sizeOfItems, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
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
		calNeighAndCollab(ratings, rMovieIDMap, neighbor, collab, weightsBuffer, sizeOfItems, taskEachNode * (rank - 1), taskEachNode);
		// sending weights to master
		printf(" %f s\n", (clock() - begin) / (1.0 * CLOCKS_PER_SEC));

		char processorName[MPI_MAX_PROCESSOR_NAME];
		int nameLen;
		MPI_Get_processor_name(processorName, &nameLen);
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