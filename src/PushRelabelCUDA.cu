#include "PushRelabelCUDA.h"
#include "StereoUsingGraphCut.h"
#define MAX_INT -1
#define MAX_NO_OF_NEIGHBOURS 8
#define K 1
#define KERNEL_CYCLES 200
#define THREAD_PER_BLOCK 1024

__device__ bool convergenceFlag = true;

__device__ int device_max(int a, int b)
{
	if(a > b)
		return a;
	return b;
}
__device__ int device_min(int a, int b)
{
	if(a < b)
		return a;
	else
		return b;
}

PushRelabelCUDA::PushRelabelCUDA(int N):count((2*N) + 2)
{

}

PushRelabelCUDA::~PushRelabelCUDA()
{
	
}

void PushRelabelCUDA::CreateSourceAndSinkEdges(GCVtx* h_vbuf, 
												GCEdge* h_edgeBuf, 
												int noOfVertices, 
												int noOfEdges)
{
	//Let the number of vertices include Source and Sink
	//source at NoOfVertices - 2 and Sink at NoOfVertices - 1

	GCVtx* source = &h_vbuf[noOfVertices -2];
	GCVtx* sink = &h_vbuf[noOfVertices - 1];

	source->first = 0;
	source->active = 1;
	source->height = noOfVertices;
	source->prdist = noOfVertices;
	source->newHeight =noOfVertices;
	source->index = noOfVertices - 2;
	source->excess = 0;
	source->fArrayCounter = -1;
	source->mark = 0;

	sink->first = 0;
	sink->newHeight = 0;
	sink->height = 0;
	sink->index = noOfVertices - 1;
	sink->active = 1;
	sink->excess = 0;
	sink->fArrayCounter = -1;
	sink->prdist = 0;
	sink->mark = 0;

	unsigned int j = 0;
	for(unsigned int i =0; i< noOfVertices - 2; i++)
	{
		//Initialize vertices for CUDA and initialize Edges
		h_vbuf[i].active = 0;
		h_vbuf[i].excess = 0;
		h_vbuf[i].prdist = 0;
		h_vbuf[i].index = i;
		h_vbuf[i].parent = 0;
		h_vbuf[i].height = 0;
		h_vbuf[i].mark = 0;
		
		if(h_vbuf[i].weight == 0)
		{
			h_vbuf[i].t = 0;
			continue;
		}
		else if(h_vbuf[i].weight > 0)
		{
			h_vbuf[i].t = 0;
			//Add Source Edge
			InitializeEdge(h_vbuf, h_edgeBuf, noOfEdges+j, noOfVertices - 2, i, h_vbuf[i].weight);
			//Add Reverse Edge
			InitializeEdge(h_vbuf, h_edgeBuf, noOfEdges+j+1, i, noOfVertices - 2, 0);
			j+=2;
		}
		else
		{
			h_vbuf[i].t = 1;
			//Add Sink Edges
			InitializeEdge(h_vbuf, h_edgeBuf, noOfEdges+j, i, noOfVertices - 1, -(h_vbuf[i].weight));
			//Add Reverse Edge
			InitializeEdge(h_vbuf, h_edgeBuf, noOfEdges+j+1, noOfVertices - 1, i, 0);
			j+=2;
		}
	}
}

void PushRelabelCUDA::InitializeEdge(GCVtx* h_vbuf,
					GCEdge* h_edgeBuf,
					int edgeIndex,
					int fromVertex,
					int toVertex,
					int weight/*Capacity*/)
{
	h_vbuf[fromVertex].fArrayCounter++;
	h_vbuf[toVertex].fArrayCounter++;

	//Set edge parameters
	h_edgeBuf[edgeIndex].next = h_vbuf[fromVertex].first;
	h_edgeBuf[edgeIndex].dst = &h_vbuf[toVertex];
	h_edgeBuf[edgeIndex].weight = weight;
	h_edgeBuf[edgeIndex].flow = 0;
	h_edgeBuf[edgeIndex].fArrayIndex = h_vbuf[toVertex].fArrayCounter;
	h_edgeBuf[edgeIndex].nodeIndex = h_edgeBuf[edgeIndex].dst->index; //h_vbuf[toVertex].index;
	h_vbuf[fromVertex].first = edgeIndex;
}

void PushRelabelCUDA::InitializeGraph(GCVtx* vbuf, GCEdge* edgebuf, int s, int &excessTotal)
{
	int next = vbuf[s].first;
	while(next != 0)
	{
		excessTotal += edgebuf[next].weight;
		//Making Cf = Capacity - flow = 0;
		edgebuf[next].flow = edgebuf[next].weight;
		edgebuf[next^1].flow = -(edgebuf[next].weight);
		edgebuf[next].dst->excess = edgebuf[next].weight;
        next = edgebuf[next].next;
	}
}

void PushRelabelCUDA::FreeDeviceMemory(GCVtx* d_vbuf, GCEdge* d_ebuf, int* d_edgeIndexArray)
{
	cudaFree(d_vbuf);
	cudaFree(d_ebuf);
	cudaFree(d_edgeIndexArray);
}

__global__ void PushRelabelKernel(GCVtx* d_vbuf, GCEdge* d_ebuf, int noOfVertices, int* d_EdgeIndexArray, int* d_flowArray)
{
	unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadId > noOfVertices - 1)
		return;

	int excess, srcHeight, dstHeight, indexOfMinHeightNode = -1, indexOfEdge = -1;
	int flowPushed;
	if(d_vbuf[threadId].excess > 0 && d_vbuf[threadId].prdist < (noOfVertices+2) ) /*Source and Sink are added here*/
	{
		excess = d_vbuf[threadId].excess;
		dstHeight = 2 * (noOfVertices + 2);		
		int next = d_vbuf[threadId].first;
		while(next != 0)
		{
			//Dileep Adding a new Condition that the the edge must have spare capacity too there in Hong Paper have to go through
			if(d_ebuf[next].weight - d_ebuf[next].flow > 0)
			{				
				srcHeight = d_vbuf[d_ebuf[next].nodeIndex].prdist;
				if(srcHeight < dstHeight)
				{
					//The min height node store
					indexOfMinHeightNode = d_ebuf[next].nodeIndex;
					indexOfEdge = next;
					dstHeight = srcHeight;
				}
			}
			next = d_ebuf[next].next;
		}

		//Dileep Add condition that dstHeight must have changed else don't change height
		if(d_vbuf[threadId].prdist > dstHeight)
		{
			//Push to the minimum height neighbor only 
			flowPushed = device_min(d_vbuf[threadId].excess, (d_ebuf[indexOfEdge].weight - d_ebuf[indexOfEdge].flow));
			atomicAdd(&d_ebuf[indexOfEdge].flow, flowPushed);
			atomicAdd(&d_ebuf[indexOfEdge^1].flow, -(flowPushed));
			atomicAdd(&d_vbuf[threadId].excess, -(flowPushed));
			atomicAdd(&d_vbuf[d_ebuf[indexOfEdge].nodeIndex].excess, flowPushed);
			
		}
		else
		{
			//Do Relabel
			d_vbuf[threadId].prdist = dstHeight + 1;
		}
	}
}

int PushRelabelCUDA::GetMaxFlow(GCVtx* h_vbuf, GCEdge* h_ebuf, int noOfVertices, int noOfEdges)
{
	//Add Source and Sink edges
	N = noOfVertices + 2;
	cudaError_t cudaStatus;
	unsigned int vbufSize = N;
	unsigned int ebufSize = (N -2) * 8 + 16;

	unsigned int blocksPerGrid, threadsPerBlock;
	threadsPerBlock = THREAD_PER_BLOCK;
	//The source and Sink node are not part of the kernel
	blocksPerGrid = (N-1) / threadsPerBlock + 1; // (int) ceil((double)(N-2)/(double)threadsPerBlock);

	//Device pointers for Vertex and edge and allocate memory in device
	GCVtx* d_vbuf;
	GCEdge* d_ebuf;
	int *d_EdgeIndexArray, *d_flowArray;

	cudaStatus = cudaMalloc((void**)&d_flowArray, vbufSize * sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_EdgeIndexArray, vbufSize * sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_ebuf, ebufSize * sizeof(GCEdge));

	cudaStatus = cudaMalloc((void**)&d_vbuf, vbufSize * sizeof(GCVtx));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        FreeDeviceMemory(d_vbuf, d_ebuf, d_EdgeIndexArray);
   }

	//Create Source and Sink Edges 
	CreateSourceAndSinkEdges(h_vbuf, 
							 h_ebuf, 
							 N, 
							 noOfEdges);

	//Initialize the Excess according to Algorithm given by Hong
	int totalExcess = 0;
	InitializeGraph(h_vbuf, h_ebuf, N - 2 /*Source*/, totalExcess);

	cudaStatus = cudaMemcpy(d_vbuf, h_vbuf, vbufSize * sizeof(GCVtx), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaMemcpy failed!");
        FreeDeviceMemory(d_vbuf, d_ebuf, d_EdgeIndexArray);
    }

	cudaStatus = cudaMemcpy(d_ebuf, h_ebuf, ebufSize * sizeof(GCEdge), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaMemcpy failed!");
        FreeDeviceMemory(d_vbuf, d_ebuf, d_EdgeIndexArray);
    }

	while(h_vbuf[N-2].excess + h_vbuf[N-1].excess < totalExcess)
	{
		cudaStatus = cudaMemcpy(d_vbuf, h_vbuf, vbufSize * sizeof(GCVtx), cudaMemcpyHostToDevice);
	    if (cudaStatus != cudaSuccess) 
		{
			fprintf(stderr, "cudaMemcpy failed!");
			FreeDeviceMemory(d_vbuf, d_ebuf, d_EdgeIndexArray);
		}

		cudaStatus = cudaMemcpy(d_ebuf, h_ebuf, ebufSize * sizeof(GCEdge), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) 
		{
			fprintf(stderr, "cudaMemcpy failed!");
			FreeDeviceMemory(d_vbuf, d_ebuf, d_EdgeIndexArray);
		}

		int iter = KERNEL_CYCLES;
		while(iter != 0)
		{
			//Launch Push Relabel Kernel
			PushRelabelKernel<<<blocksPerGrid, threadsPerBlock>>>(d_vbuf, d_ebuf, N-2, d_EdgeIndexArray, d_flowArray);
			iter=iter-1;
		}
		

		//Copy the Graph from device back to host to do Global Relbelling using BFS in Host
		cudaStatus = cudaMemcpy(h_vbuf, d_vbuf, vbufSize * sizeof(GCVtx), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) 
		{
			fprintf(stderr, "cudaMemcpy failed!");
			FreeDeviceMemory(d_vbuf, d_ebuf, d_EdgeIndexArray);
		}

		cudaStatus = cudaMemcpy(h_ebuf, d_ebuf, ebufSize * sizeof(GCEdge), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) 
		{
			fprintf(stderr, "cudaMemcpy failed!");
			FreeDeviceMemory(d_vbuf, d_ebuf, d_EdgeIndexArray);
		}
		printf("Source Excess %d and Sink Excess %d \n", h_vbuf[N-2].excess, h_vbuf[N-1].excess);
		//Call Global Relabel On Host
		GlobalRelabel(h_vbuf, h_ebuf, N, totalExcess);

	}

	FindMinCut(N, h_vbuf, h_ebuf);
	printf("\n Maxflow %d", h_vbuf[N-1].excess);
	cudaFree(d_vbuf);
	cudaFree(d_ebuf);
	cudaFree(d_EdgeIndexArray);
	cudaFree(d_flowArray);
	return h_vbuf[N-1].excess;
}

void PushRelabelCUDA::FindMinCut(int numVertices, GCVtx* h_vbuf, GCEdge* h_ebuf)
{
	//Algorithm: Do a BFS from the Sink 
	//All Edges from Sink Whose Sister Edge has Capacity - flow > 0 set the value of t to 1 i.e they belong to the sink.
	//All vertices with negative weights are connected to sink. Push them all  to the queue and do bfs from there.. 
	while(!BFSQueue.empty())
		BFSQueue.pop();

	for(int i=0; i<numVertices; i++)
	{
		h_vbuf[i].t = 0;
		h_vbuf[i].parent = 0;
	}

	//h_vbuf[numVertices-1].t = 0;
	//Searching from the sink
	BFSQueue.push(&h_vbuf[numVertices-1]);
	while(!BFSQueue.empty())
	{
		GCVtx* v = BFSQueue.front();
		BFSQueue.pop();

		for(int edgeIndex = v->first; edgeIndex != 0; edgeIndex = h_ebuf[edgeIndex].next) 
		{
            //int sisterEdgeIndex = edgeIndex ^ 1;
            GCVtx * nxt = h_ebuf[edgeIndex].dst;

            if(nxt->t == 0 && (h_ebuf[edgeIndex^1].weight - h_ebuf[edgeIndex^1].flow) > 0) 
			{
				//nxt->visit = 1;
                nxt->t = 1;
                nxt->parent = 1;
                BFSQueue.push(nxt);
            }
		}
	}
}

void PushRelabelCUDA::ViolationCancellation(GCVtx* h_vbuf, GCEdge* h_ebuf, int noOfVertices)
{
	//Go through all the edges and see if there is any vertex with excess and edge has spare capacity 
	//If height of the vertex is greater than destination this edge is a violating edge.
	//Saturate the violating edge
	//Push the capacity (weight - flow) to the destination

	for(unsigned int i=0; i<noOfVertices; i++)
	{
		int next = h_vbuf[i].first;
		while(next != 0)
		{
			if(h_vbuf[i].prdist > (h_ebuf[next].dst->prdist + 1))
			{
				h_vbuf[i].excess = h_vbuf[i].excess - (h_ebuf[next].weight - h_ebuf[next].flow);
				h_ebuf[next].dst->excess = h_ebuf[next].dst->excess + (h_ebuf[next].weight - h_ebuf[next].flow);
				h_ebuf[next^1].flow = h_ebuf[next^1].flow - (h_ebuf[next].weight - h_ebuf[next].flow);
				h_ebuf[next].flow = h_ebuf[next].weight;	
			}
			next = h_ebuf[next].next;
		}
	}
}

void PushRelabelCUDA::BFSGlobalRelabel(GCVtx* h_vbuf, GCEdge* h_ebuf, int noOfVertices, int& totalExcess)
{
	//Do a backward BFS from the Sink and relabel based on the level of the node in the tree till the sink 
	while(!BFSQueue.empty())
	{
		BFSQueue.pop();
	}

	for(unsigned int i=0; i<N; i++)
	{
		//Have to check this assignment of label to N if it helps in speeding up
		h_vbuf[i].prdist = N+1;
		h_vbuf[i].isRelabelled = 0;
		h_vbuf[i].visit = 0;
	}

	int next; 
	h_vbuf[noOfVertices-1].visit = 1;
	h_vbuf[noOfVertices-1].prdist = 0;
	GCVtx *dstVertex,*srcVertex; 

	BFSQueue.push(&h_vbuf[noOfVertices-1]);
	//Now access each vertex from the queue and go through its neighbours if it is not visited and Relabelled 
	while(!BFSQueue.empty())
	{
		srcVertex = BFSQueue.front(); BFSQueue.pop();
		next = srcVertex->first;
		while(next != 0)
		{
			dstVertex = h_ebuf[next].dst;
			if(dstVertex->visit != 1)
			{
				if((h_ebuf[next^1].weight - h_ebuf[next^1].flow) > 0)
				{
					dstVertex->prdist = srcVertex->prdist + 1;
					dstVertex->isRelabelled = 1;
					BFSQueue.push(dstVertex);
					dstVertex->visit = 1;
				}			
			}
			next = h_ebuf[next].next;
		}
	}

	//Go through all vertices and if they have not been relabelled or marked then and drain out their excess and reduce it from totalExcess
	//mark that vertex
	for(unsigned int i=0; i<noOfVertices-1; i++)
	{
		if(h_vbuf[i].isRelabelled == 0 && h_vbuf[i].mark == 0)
		{
			//Mark the vertex
			h_vbuf[i].mark = 1;
			//drain out the excess
			totalExcess = totalExcess - h_vbuf[i].excess;
		}
	}
}

void PushRelabelCUDA::GlobalRelabel(GCVtx* h_vbuf, GCEdge* h_ebuf, int noOfVertices, int& excessTotal)
{
	//First do violation cancellation no Of vertices contains source and Sink
	ViolationCancellation(h_vbuf, h_ebuf, noOfVertices);

	//Do Global relabel and change Excess total based on it 
	BFSGlobalRelabel(h_vbuf, h_ebuf, noOfVertices, excessTotal);
}

