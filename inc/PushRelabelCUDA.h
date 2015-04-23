
//#include "StereoUsingGraphCut.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <queue>

//typedef long long LL;
struct GCVtx;
struct GCEdge;

class PushRelabelCUDA
{
public:
	long long GetMaxFlow(GCVtx* vbuf, GCEdge* ebuf, int NoOfVertices, int NoOfEdges, int s, int t);
	int GetMaxFlow(GCVtx* h_vbuf, GCEdge* h_ebuf, int noOfVertices, int noOfEdges);
	PushRelabelCUDA(int N);
	~PushRelabelCUDA();

private:
	void InitializeEdge(GCVtx* h_vbuf,
						GCEdge* h_edgeBuf,
						int edgeIndex,
						int fromVertex,
						int toVertex,
						int weight/*Capacity*/);

	void CreateSourceAndSinkEdges(GCVtx* h_vbuf, 
							 GCEdge* h_edgeBuf, 
							 int noOfVertices, 
							 int noOfEdges);

	void InitializeGraph(GCVtx* vbuf, GCEdge* edgebuf, int s, int& excessTotal);

	void FreeDeviceMemory(GCVtx* d_vbuf, GCEdge* d_ebuf, int* d_edgeIndexArray);

	void GlobalRelabel(GCVtx* h_vbuf, GCEdge* h_ebuf, int noOfVertices, int& excessTotal);

	void ViolationCancellation(GCVtx* h_vbuf, GCEdge* h_ebuf, int noOfVertices);

	void BFSGlobalRelabel(GCVtx* h_vbuf, GCEdge* h_ebuf, int noOfVertices, int& excessTotal);

	void Push(int edgeIndex, GCVtx* from);	
	
	void CreateSourceAndSinkEdges(GCVtx* d_vbuf, 
										 GCEdge* d_edgeBuf, 
										 int noOfVertices, 
										 int noOfEdges, 
										 int s, 
										 int t);
	void InitializeSourceExcess(GCVtx* vbuf, GCEdge* ebuf, int s);
	void Gap(int k, int* prDist, unsigned int* countArray, int NumberOfVertices);
	void FindMinCut(int numVertices, GCVtx* vbuf, GCEdge* ebuf);
private:
	int N;
	std::vector<int> count;
	std::queue<GCVtx*> BFSQueue;
};
