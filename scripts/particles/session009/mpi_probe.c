#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv) {
 MPI_Init(&argc,&argv);
 int rank,n,ln;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&n);
 char host[MPI_MAX_PROCESSOR_NAME];MPI_Get_processor_name(host,&ln);
 const int count=65536;
 int *send=malloc((size_t)n*count*sizeof(int)),*recv=malloc((size_t)n*count*sizeof(int));
 for(int j=0;j<n;j++)for(int k=0;k<count;k++)send[j*count+k]=rank;
 MPI_Alltoall(send,count,MPI_INT,recv,count,MPI_INT,MPI_COMM_WORLD);
 int bad=0,total=0;
 for(int j=0;j<n;j++)for(int k=0;k<count;k++)if(recv[j*count+k]!=j)bad++;
 MPI_Allreduce(&bad,&total,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
 printf("rank=%d/%d host=%s alltoall_errors=%d\n",rank,n,host,bad);fflush(stdout);
 free(send);free(recv);MPI_Finalize();return total?1:0;
}
