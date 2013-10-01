
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <time.h>
#include <cmath>
#include <limits.h>

#define MAXHOPS 4
#define MAX_WAITING_TIME 420
#define BLOCK_LENGTH 512
#define END_OF_ARRAY 2147483647
#define BUFFER_LENGTH 50
#define AIRPORT_PATH "C:/Users/acer/Desktop/Semester 7/Project/AA_airports.txt" //"C:/Users/acer/Desktop/Semester 7/Project/Data/AA_airports.txt"

#define FLIGHT_PATH "C:/Users/acer/Desktop/Semester 7/Project/AA_data1.txt" //"C:/Users/acer/Desktop/Semester 7/Project/Data/OAGX_data_num_1.txt"

bool bool1 = true;
bool bool2 = false;
using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);

// for cuda error checking
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            return 1; \
        } \
    } while (0)


int ADJ_MATRIX_DIM;
__device__ int DEV_ADJ_MATRIX_DIM;

// FYP_BFS.cpp : Defines the entry point for the console application.
//

///////////////////Global Variables///////////////////
struct Flight{
	int flightNumber;
	int source;
	int destination;
	int arrivalTime;
	int departureTime;
	int price;
	string code;
};

vector<string> Airport_List;
vector<Flight> Flight_List;
vector<int>** AdjMatrix;

//////////////////////////////////////////////////////

//////////////////Data Read///////////////////////////
int readAirports(){
	
	ifstream myFile;
	myFile.open(AIRPORT_PATH);
	int numberOfAirports=0;
	if(myFile.is_open()){

		
		string line;

		cout<<"Reading Airports"<<endl;
		
		while(myFile.good()){
//------------------------------changed-------------------//			
			//myFile.ignore(256,' ');
			string s;
			myFile>>s;
			Airport_List.push_back(s);
			
			numberOfAirports++;
		}
	}
	myFile.close();
	ADJ_MATRIX_DIM = Airport_List.size();

	//cudaMemcpy(DEV_ADJ_MATRIX_DIM,&ADJ_MATRIX_DIM,sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(DEV_ADJ_MATRIX_DIM,&ADJ_MATRIX_DIM,sizeof(int),0,cudaMemcpyHostToDevice);
	cudaCheckErrors("Error copying adj matrix dim to device");
	cout<<Airport_List.size()<<" Airports Found"<<endl;

	return 1;
}


void readFlights(){

	//this is a bloody array of pointers
	AdjMatrix = new vector<int>*[Airport_List.size()];

	for(int i=0;i<Airport_List.size();i++){
		//thisi is a bloody array of vectors
		AdjMatrix[i] = new vector<int>[Airport_List.size()];
	}

	ifstream myFile;
	myFile.open(FLIGHT_PATH);

	int numOfFlights = 0;
	if(myFile.is_open()){
				
		string line;
		
		Flight tempFlight;
		while(myFile.good()){
//---------------------------------------changed----------------------------------------//			
			tempFlight.flightNumber= numOfFlights;
			/*myFile>>tempFlight.source;
			myFile>>tempFlight.destination;
			myFile>>tempFlight.departureTime;
			myFile>>tempFlight.arrivalTime;
			if(tempFlight.arrivalTime<tempFlight.departureTime) tempFlight.arrivalTime+=10080;
			myFile>>tempFlight.price;
			
			myFile>>tempFlight.code;*/

			myFile>>tempFlight.source;
			myFile>>tempFlight.destination;
			myFile>>tempFlight.price;
			myFile>>tempFlight.departureTime;
			myFile>>tempFlight.arrivalTime;
			if(tempFlight.arrivalTime<tempFlight.departureTime) tempFlight.arrivalTime+=10080;			
			
			myFile>>tempFlight.code;
						
			//add this flight to the adjmatrix;
			Flight_List.push_back(tempFlight);

			AdjMatrix[tempFlight.source][tempFlight.destination].push_back(tempFlight.flightNumber);

			numOfFlights++;

			if(numOfFlights%10000==0) cout<<"*";
		}
			
			cout<<endl;
	}
	myFile.close();
	
	cout<<Flight_List.size()<<" Flights Found"<<endl;
}

/////////////////////////////////////////////////////////////////////////////////////


struct route{
	vector<int> flights;
	int weight;
};



int initializeFlightListInDevice(Flight* &dev_flight_list){
	//allocate space for the flight list in cuda
	cudaMalloc((void**)&dev_flight_list, Flight_List.size()*sizeof(Flight));
	cudaCheckErrors("Failed to allocate memory to flight list");	
	cudaMemcpy(dev_flight_list,&Flight_List[0],Flight_List.size()*sizeof(Flight),cudaMemcpyHostToDevice);
	cudaCheckErrors("Failed to copy flight list");
	
	return 1;
}



int initializeAdjMatrixInDevice(int** &dev_adj_list, int ** &host_adj_vector){

	
	size_t size = ADJ_MATRIX_DIM*ADJ_MATRIX_DIM*sizeof(int*);

	

	//the vector in host that records the pointers in device memory
	host_adj_vector = (int **)malloc(size);
	
	//i indicates rows and j indicates columns of the adjacency matrix
	//allocate device memory for the boolean vector
	
	//allocate memory for each manhattan in device and store the pointer in memory
	for(int i=0;i<ADJ_MATRIX_DIM;i++){
		for(int j=0;j<ADJ_MATRIX_DIM;j++){
			
			cudaMalloc((void **)&host_adj_vector[i*ADJ_MATRIX_DIM+j],AdjMatrix[i][j].size()*sizeof(int));
			cudaCheckErrors("Failed to allocate memory to airport list manhattan:");
			cudaMemcpy(host_adj_vector[i*ADJ_MATRIX_DIM+j],&AdjMatrix[i][j][0],AdjMatrix[i][j].size()*sizeof(int),cudaMemcpyHostToDevice);
			cudaCheckErrors("Failed to copy data to airport list manhattan:");
			
		}
		if(i%100==0) cout<<"&";
	}	
	cout<<endl;

	cudaMalloc((void***)&dev_adj_list,size);
	cudaCheckErrors("Failed to allocate memory to pointer list in device");
	cudaMemcpy(dev_adj_list,host_adj_vector,size,cudaMemcpyHostToDevice);
	cudaCheckErrors("Failed to allocate data to pointer list in device");

	return 1;
}

int initializeBooleanMatrixInDevice(int* &boolean_matrix){
	int* host_bool_matrix= new int[ADJ_MATRIX_DIM*ADJ_MATRIX_DIM];
	for(int i=0;i<ADJ_MATRIX_DIM;i++){
		for(int j=0;j<ADJ_MATRIX_DIM;j++){
			host_bool_matrix[i*ADJ_MATRIX_DIM+j] =  (AdjMatrix[i][j].size() !=0);
		}
	}
	size_t size_bool =ADJ_MATRIX_DIM*ADJ_MATRIX_DIM*sizeof(int);
	cudaMalloc((void**)&boolean_matrix,size_bool);
	cudaCheckErrors("Failed to allocate memory to boolean adj matrix");
	cudaMemcpy(boolean_matrix,host_bool_matrix,size_bool,cudaMemcpyHostToDevice);
	cudaCheckErrors("Failed to move data to boolean adj matrix");
	delete(host_bool_matrix);

	return 1;
}

int initializeBuffer(int* &buffer){
	int* host_bool_buffer= new int[ADJ_MATRIX_DIM*ADJ_MATRIX_DIM];
	for(int i=0;i<ADJ_MATRIX_DIM;i++){
		for(int j=0;j<ADJ_MATRIX_DIM;j++){
			host_bool_buffer[i*ADJ_MATRIX_DIM+j] = false;
		}
	}
	size_t size_bool =ADJ_MATRIX_DIM*ADJ_MATRIX_DIM*sizeof(int);
	cudaMalloc((void**)&buffer,size_bool);
	cudaCheckErrors("Failed to allocate memory to boolean buffer");
	cudaMemcpy(buffer,host_bool_buffer,size_bool,cudaMemcpyHostToDevice);
	cudaCheckErrors("Failed to move data to boolean buffer");
	delete(host_bool_buffer);
	return 1;
}

__global__ void testBuffer(int* buffer,int* result, int size){
	int id =blockIdx.x*blockDim.x+threadIdx.x;
	if(id<size){
		if(buffer[id])
		result[id] = 1234345;
		else
			result[id] = 0;
	}
}

__global__ void testMatrix(int** devVector,int size, int* result, Flight* flights){
	//block dimension is the number of threads in a block. since blockid is zero based multiplying gets you somewhre close. 
	//to gt the correct position all u have to do then is to add the thread id
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	result[i] = 0;
	if(i<size*size && devVector[i]!= NULL )		
		result[i] = flights[devVector[i][0]].source;
}

__global__ void testMatrixBoolean(int* devMatrixBoolean,int size, int* result){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	result[i] = 0;
	//put 1 if a manhattan exists for the particular position
	if(i<size*size && devMatrixBoolean[i])		
		result[i] = 1;
}
//initialize buffer to end of array value so that as values are filled the array size will change, but will still be
//indicated by the first end of array value

//__global__ void initializeBuffer(bool* buffer, int size){
//	int id = blockIdx.x*blockDim.x+threadIdx.x;
//	if(id<size)
//		buffer[id] = false;
//}

//give enough threads to span the source row
//maximum id should be adj_matrix_dimension

__global__ void firstExpansion(int* buffer, int*dev_boolean_matrix, int source){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	//the source row. 
	
	//if(id<DEV_ADJ_MATRIX_DIM*DEV_ADJ_MATRIX_DIM){
	//	//if(dev_adj_matrix[DEV_ADJ_MATRIX_DIM*(source-1)+id]!=NULL){
	//	//	//set source to the precedant node list of each relevant airport
	//	//	buffer[id*BUFFER_LENGTH] = source;
	//	//}
	//}
}

//max id is number of airports
__global__ void expansion(int* dev_buffer,int* boolean_matrix, int* dev_source_vector,int matrix_dimension){
	
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	int row = (int) floor((double)id/matrix_dimension);
	int column = id%matrix_dimension;
	if(row<matrix_dimension && column<matrix_dimension){
		
		//for the source row if the matrix row column position has a manhattan set the buffer position to true
		dev_buffer[id] = (dev_source_vector[row] && boolean_matrix[id]);
		
		
	}
	__syncthreads();
	//set the source vector positions to zero by the first of each row
	if(row<matrix_dimension && column<matrix_dimension&& column==0) dev_source_vector[row]= 0;
	__syncthreads();
	
	if((row<matrix_dimension && column<matrix_dimension) && boolean_matrix[id]){
		dev_source_vector[column] = 1;
	}
}

//__global__ void copyNextSource(bool* next_source_array, bool* current_array, int size){
//	int id = blockDim.x*blockIdx.x+threadIdx.x;
//	if(id<size)
//		cudaMemcpy
//}

int main(int argc)
{
	
	readAirports();
	readFlights();
	
	int source = 344;
	int destination = 190;

	
	Flight* dev_flight_list;
	
	int** dev_adj_list;
	int* dev_adj_matrix_boolean;
	int** host_adj_vector;
	int* dev_level1;
	int* dev_level2;
	int* frames;
	//boolean array containing source airports in the next expansion
	int* dev_next_source_array;
	
	size_t matrixSize = ADJ_MATRIX_DIM*ADJ_MATRIX_DIM*sizeof(int);
	size_t bufferSize = ADJ_MATRIX_DIM*ADJ_MATRIX_DIM*sizeof(int);
		
	//add the flight array to GPU memory
	cout<<"Initializing Flights"<<endl;	
	initializeFlightListInDevice(dev_flight_list);
	cout<<"finished initializing FLights"<<endl;

	//add the adjacency matrix with manhattans to GPU
	cout<<"Initializing Matrix"<<endl;
	initializeAdjMatrixInDevice(dev_adj_list,host_adj_vector);	
	cout<<"Finished with adj matrix"<<endl;

	//add the boolean adjacency matrix (without manhattans) to GPU
	cout<<"Initializing Boolean Matrix"<<endl;
	initializeBooleanMatrixInDevice(dev_adj_matrix_boolean);	
	cout<<"Finished with boolean matrix"<<endl;

	//allocate memory for the 'next source array' in device
	cudaMalloc((void**)&dev_next_source_array,ADJ_MATRIX_DIM*sizeof(int));
	cudaCheckErrors("Failed to allocate memory to next source list");

	int* source_vector = new int [ADJ_MATRIX_DIM];

	//initialize the 'next source vector' with the source row of the adjacency matrix
	for(int i=0;i<ADJ_MATRIX_DIM;i++){
		source_vector[i] = AdjMatrix[source][i].size()!=0;
	}
	//intialize 'next source array' in device
	cudaMemcpy(dev_next_source_array,source_vector,ADJ_MATRIX_DIM*sizeof(int),cudaMemcpyHostToDevice);
	cudaCheckErrors("Failed to move data  to next source list");

	delete(source_vector);
	//////////////////////initialize the buffers for all the levels/////////////////
	cout<<"initializing Buffers"<<endl;

	initializeBuffer(dev_level1);
	initializeBuffer(dev_level2);
	
	cout<<"initialized buffers"<<endl;
	///////////////////////////////////////Interations///////////////////////////////////////
	
	int numBlocks = ceil((double)ADJ_MATRIX_DIM*ADJ_MATRIX_DIM/BLOCK_LENGTH);

	ofstream myFile;
	myFile.open("nextSource.txt");

	
	int* myArray = (int*) malloc(ADJ_MATRIX_DIM*sizeof(int));
	cudaMemcpy(myArray,dev_next_source_array,ADJ_MATRIX_DIM*sizeof(int),cudaMemcpyDeviceToHost);
	cudaCheckErrors("Failed to copy data from buffer array to host array");


	for(int i=0;i<ADJ_MATRIX_DIM;i++){
		//if(myArray[i]!= NULL)
			myFile<<myArray[i];
	}
	myFile<<endl<<endl;
	free(myArray);
	cout<<"moving into first expansion"<<endl;

	ofstream myFile2;
	myFile2.open("Frame1.txt");
	int* myArray2 = (int*)malloc(ADJ_MATRIX_DIM*ADJ_MATRIX_DIM*sizeof(int));

	   expansion<<<numBlocks,BLOCK_LENGTH>>>(dev_level1,dev_adj_matrix_boolean,dev_next_source_array,ADJ_MATRIX_DIM);
	 cudaThreadSynchronize();
	 cudaCheckErrors("Error occured in expansion");
	cout<<"finished expansion"<<endl;
	  cudaMemcpy(myArray2,dev_level1,ADJ_MATRIX_DIM*ADJ_MATRIX_DIM*sizeof(int),cudaMemcpyDeviceToHost);
	 cudaCheckErrors("Failed to retrieve memory from first frame");
	for(int i=0;i<ADJ_MATRIX_DIM*ADJ_MATRIX_DIM;i++){
		//if(myArray[i]!= NULL)
			myFile2<<myArray2[i];
	}

	myFile2<<endl<<endl;
	myFile2.close();

	free(myArray2);
	int* myArray1 = (int*)malloc(ADJ_MATRIX_DIM*sizeof(int));

	/* expansion<<<numBlocks,BLOCK_LENGTH>>>(dev_level1,dev_adj_matrix_boolean,dev_next_source_array,ADJ_MATRIX_DIM);
	 cudaThreadSynchronize();
	 cudaCheckErrors("Error occured in expansion");
	cout<<"finished expansion"<<endl;*/
	 cudaMemcpy(myArray1,dev_next_source_array,ADJ_MATRIX_DIM*sizeof(int),cudaMemcpyDeviceToHost);
	 cudaCheckErrors("Failed to retrieve memory from buffer array 1.2 to host");
	for(int i=0;i<ADJ_MATRIX_DIM;i++){
		//if(myArray[i]!= NULL)
			myFile<<myArray1[i];
	}
	myFile<<endl<<endl;
	free(myArray1);
	myFile.close();
	
	
	//cudaFree(dev_next_source_array);
	
	
	cudaFree(dev_level1);
	cudaFree(dev_level2);
	cudaFree(dev_flight_list);
	
	//for(int i=0;i<ADJ_MATRIX_DIM*ADJ_MATRIX_DIM;i++){
	//	//cout<<i<<endl;
	//	if(host_adj_vector[i] !=NULL)
	//		cudaFree(host_adj_vector[i]);
	//}
	cudaFree(dev_adj_list);
	free(host_adj_vector);
	
	
	
	return 0;
}

