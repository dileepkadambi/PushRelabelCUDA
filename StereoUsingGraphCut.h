#pragma once

#include <vector>
#include "sources\modules\legacy\src\precomp.hpp"

struct GCPREdge;
typedef long long LL;

typedef struct GCVtx
{
	//For Boykov Algorithm
    GCVtx *next;
    int parent;
    int first;
    int ts;
    int dist;
    short weight;
    uchar t; 

	//For BFS
	int visit;
	int isRelabeled;

	//For Push Relabel
	int excess;
	int active;
	int prdist;
	int newPrdist;

	//For CUDA
	int fArrayCounter;
	unsigned int height; //Not required
	unsigned int newHeight;//Not Required
	int index;
}
GCVtx;

typedef struct GCEdge
{
    GCVtx* dst;
    int next;
    int weight;

	//Added by Team 033
	int flow;
	int fArrayIndex;
	int nodeIndex;

}
GCEdge;

typedef struct GCFlowArrayStruct 
{
	int flowPushed;
	int edgeIndex;
}
GCFlowArrayStruct;