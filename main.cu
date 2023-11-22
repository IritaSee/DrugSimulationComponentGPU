#include <cuda.h>
#include <cuda_runtime.h>

// #include "modules/drug_sim.hpp"
#include "modules/glob_funct.hpp"
#include "modules/glob_type.hpp"
#include "modules/gpu.cuh"
#include "modules/cipa_t.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <math.h>
#include <vector>
#include <sys/stat.h>

#define ENOUGH ((CHAR_BIT * sizeof(int) - 1) / 3 + 2)
char buffer[255];

unsigned int datapoint_size = 7000;
const unsigned int sample_limit = 10000;
double ic50[14 * sample_limit]; //temporary

clock_t START_TIMER;

clock_t tic();
void toc(clock_t start = START_TIMER);

clock_t tic()
{
    return START_TIMER = clock();
}

void toc(clock_t start)
{
    std::cout
        << "Elapsed time: "
        << (clock() - start) / (double)CLOCKS_PER_SEC << "s"
        << std::endl;
}
  
int gpu_check(unsigned int datasize){
    int num_gpus;
    float percent;
    int id;
    size_t free, total;
    cudaGetDeviceCount( &num_gpus );
    for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        cudaSetDevice( gpu_id );
        cudaGetDevice( &id );
        cudaMemGetInfo( &free, &total );
        percent = (free/(float)total);
        printf("GPU No %d\nFree Memory: %ld, Total Memory: %ld (%f percent free)\n", id,free,total,percent*100.0);
    }
    percent = 1.0-(datasize/(float)total);
    //// this code strangely gave out too small value, so i disable the safety switch for now

    // printf("The program uses GPU No %d and %f percent of its memory\n", id,percent*100.0);
    // printf("\n");
    // if (datasize<=free) {
    //   return 0;
    // }
    // else {
    //   return 1;
    // }


    return 0;
    
}

// since installing MPI in Windows
// is quite a hassle, don't bother
// to use it in Windows.
// #ifndef _WIN32
// 	#include <mpi.h>
// #endif

// constants to avoid magic values
// static const char *RESULT_FOLDER_PATH = "result";
// static const double CONTROL_CONC = 0.;


// get the IC50 data from file
drug_t get_IC50_data_from_file(const char* file_name);
// return error and message based on the IC50 data
int check_IC50_content(const drug_t* ic50, const param_t* p_param);

// define MPI data structure for qinward_t to be broadcasted
// #ifndef _WIN32
// MPI_Datatype create_mpi_qinward_t();
// #endif

// drug_t get_IC50_data_from_file(const char* file_name)
// {
//   FILE *fp_drugs;
//   drug_t ic50;
//   char *token, buffer[255];
//   row_data temp_array;
//   unsigned short idx;

//   if( (fp_drugs = fopen(file_name, "r")) == NULL){
//     printf("Cannot open file %s in %s at rank %d\n",
//       file_name, mympi::host_name, mympi::rank);
//     return ic50;
//   }

//   fgets(buffer, sizeof(buffer), fp_drugs); // skip header
//   while( fgets(buffer, sizeof(buffer), fp_drugs) != NULL )
//   { // begin line reading
//     token = strtok( buffer, "," );
//     idx = 0;
//     while( token != NULL )
//     { // begin data tokenizing
//       temp_array.data[idx++] = strtod(token, NULL);
//       token = strtok(NULL, ",");
//     } // end data tokenizing
//     ic50.push_back(temp_array);
//   } // end line reading

//   fclose(fp_drugs);
//   return ic50;
// }

int get_IC50_data_from_file(const char* file_name, double *ic50)
{
    /*
    a host function to take all samples from the file, assuming each sample has 14 features.

    it takes the file name, and an ic50 (already declared in 1D, everything become 1D)
    as a note, the data will be stored in 1D array, means this functions applies flatten.

    it returns 'how many samples were detected?' in integer.
    */
  FILE *fp_drugs;
//   drug_t ic50;
  char *token;
  
  unsigned short idx;

  if( (fp_drugs = fopen(file_name, "r")) == NULL){
    printf("Cannot open file %s\n",
      file_name);
    return 0;
  }
  idx = 0;
  int sample_size = 0;
  fgets(buffer, sizeof(buffer), fp_drugs); // skip header
  while( fgets(buffer, sizeof(buffer), fp_drugs) != NULL )
  { // begin line reading
    token = strtok( buffer, "," );
    while( token != NULL )
    { // begin data tokenizing
      ic50[idx++] = strtod(token, NULL);
      token = strtok(NULL, ",");
    } // end data tokenizing
    sample_size++;
  } // end line reading

  fclose(fp_drugs);
  return sample_size;
}


int check_IC50_content(const drug_t* ic50, const param_t* p_param)
{
	if(ic50->size() == 0){
		printf("Something problem with the IC50 file!\n");
		return 1;
	}
	else if(ic50->size() > 2000){
		printf( "Too much input! Maximum sample data is 2000!\n");
		return 2;
	}
	else if(p_param->pace_max < 750 && p_param->pace_max > 1000){
		printf("Make sure the maximum pace is around 750 to 1000!\n");
		return 3;
	}
	// else if(mympi::size > ic50->size()){
	// 	printf("%s\n%s\n",
  //               "Overflow of MPI Process!",
  //               "Make sure MPI Size is less than or equal the number of sample");
	// 	return 4;
	// }
	else{
		return 0;
	}
}

int main(int argc, char **argv)
{
	// enable real-time output in stdout
	setvbuf( stdout, NULL, _IONBF, 0 );
	
// #ifndef _WIN32
// 	MPI_Init( &argc, &argv );
// 	MPI_Comm_size( MPI_COMM_WORLD, &mympi::size );
// 	MPI_Comm_rank( MPI_COMM_WORLD, &mympi::rank );
// 	MPI_Get_processor_name(mympi::host_name, &mympi::host_name_len);
// #else
// 	mympi::size = 1;
// 	mympi::rank = 0;
// 	snprintf(mympi::host_name,sizeof(mympi::host_name),"%s","host");
// 	mympi::host_name_len = 4;
// #endif

// NEW CODE STARTS HERE //
    // mycuda *thread_id;
    // cudaMalloc(&thread_id, sizeof(mycuda));


    // input variables for cell simulation
    param_t *p_param, *d_p_param;
	  p_param = new param_t();
  	p_param->init();

    const double CONC = p_param->conc;

    double *d_ic50;
    double *d_ALGEBRAIC;
    double *d_CONSTANTS;
    double *d_RATES;
    double *d_STATES;

    double *d_STATES_RESULT;

    // double *time;
    // double *dt;
    // double *states;
    // double *ical;
    // double *inal;
    // double *cai_result;
    // double *ina;
    // double *ito;
    // double *ikr;
    // double *iks;
    // double *ik1;
    cipa_t *temp_result, *cipa_result;


    p_param->show_val();

    int num_of_constants = 146;
    int num_of_states = 41;
    int num_of_algebraic = 199;
    int num_of_rates = 41;

    // snprintf(buffer, sizeof(buffer),
    //   "./drugs/bepridil/IC50_samples.csv"
    //   // "./drugs/bepridil/IC50_optimal.csv"
    //   // "./IC50_samples.csv"
    //   );
    int sample_size = get_IC50_data_from_file(p_param->hill_file, ic50);
    if(sample_size == 0)
        printf("Something problem with the IC50 file!\n");
    // else if(sample_size > 2000)
    //     printf("Too much input! Maximum sample data is 2000!\n");
    printf("Sample size: %d\n",sample_size);
    cudaSetDevice(p_param->gpu_index);
    printf("preparing GPU memory space \n");
    cudaMalloc(&d_ALGEBRAIC, num_of_algebraic * sample_size * sizeof(double));
    cudaMalloc(&d_CONSTANTS, num_of_constants * sample_size * sizeof(double));
    cudaMalloc(&d_RATES, num_of_rates * sample_size * sizeof(double));
    cudaMalloc(&d_STATES, num_of_states * sample_size * sizeof(double));

    cudaMalloc(&d_p_param,  sizeof(param_t));

    // prep for 1 cycle plus a bit (7000 * sample_size)
    cudaMalloc(&temp_result, sample_size * sizeof(cipa_t));
    cudaMalloc(&cipa_result, sample_size * sizeof(cipa_t));

    // cudaMalloc(&time, sample_size * datapoint_size * sizeof(double)); 
    // cudaMalloc(&dt, sample_size * datapoint_size * sizeof(double)); 
    // cudaMalloc(&states, sample_size * datapoint_size * sizeof(double));
    // cudaMalloc(&ical, sample_size * datapoint_size * sizeof(double));
    // cudaMalloc(&inal, sample_size * datapoint_size * sizeof(double));
    // cudaMalloc(&cai_result, sample_size * datapoint_size * sizeof(double));
    // cudaMalloc(&ina, sample_size * datapoint_size * sizeof(double));
    // cudaMalloc(&ito, sample_size * datapoint_size * sizeof(double));
    // cudaMalloc(&ikr, sample_size * datapoint_size * sizeof(double));
    // cudaMalloc(&iks, sample_size * datapoint_size * sizeof(double));
    // cudaMalloc(&ik1, sample_size * datapoint_size * sizeof(double));
    cudaMalloc(&d_STATES_RESULT, num_of_states * sample_size * sizeof(double));

    printf("Copying sample files to GPU memory space \n");
    cudaMalloc(&d_ic50, sample_size * 14 * sizeof(double));
    
    cudaMemcpy(d_ic50, ic50, sample_size * 14 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_param, p_param, sizeof(param_t), cudaMemcpyHostToDevice);

    // // Get the maximum number of active blocks per multiprocessor
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, do_drug_sim_analytical, threadsPerBlock);

    // // Calculate the total number of blocks
    // int numTotalBlocks = numBlocks * cudaDeviceGetMultiprocessorCount();

    tic();
    printf("Timer started, doing simulation.... \n GPU Usage at this moment: \n");
    int thread;
    if (sample_size>=100){
      thread = 100;
    }
    else thread = sample_size;
    int block = int(ceil(sample_size/thread));
    // int block = (sample_size + thread - 1) / thread;
    if(gpu_check(15 * sample_size * datapoint_size * sizeof(double) + sizeof(param_t)) == 1){
      printf("GPU memory insufficient!\n");
      return 0;
    }
    printf("Sample size: %d\n",sample_size);
    cudaSetDevice(p_param->gpu_index);
    printf("\n   Configuration: \n\n\tblock\t||\tthread\n---------------------------------------\n  \t%d\t||\t%d\n\n\n", block,thread);
    // initscr();
    // printf("[____________________________________________________________________________________________________]  0.00 %% \n");

    kernel_DrugSimulation<<<block,thread>>>(d_ic50, d_CONSTANTS, d_STATES, d_RATES, d_ALGEBRAIC, 
                                              d_STATES_RESULT,
                                              // time, states, dt, cai_result,
                                              // ina, inal, 
                                              // ical, ito,
                                              // ikr, iks, 
                                              // ik1,
                                              sample_size,
                                              temp_result, cipa_result,
                                              d_p_param
                                              );
                                      //block per grid, threads per block
    // endwin();
    
    cudaDeviceSynchronize();
    

    printf("allocating memory for computation result in the CPU, malloc style \n");
    double *h_states;

    h_states = (double *)malloc(num_of_states * sample_size * sizeof(double));
    printf("...allocating for all states, all set!\n");

    ////// copy the data back to CPU, and write them into file ////////
    printf("copying the data back to the CPU \n");
    
    cudaMemcpy(h_states, d_STATES_RESULT, sample_size * num_of_states *  sizeof(double), cudaMemcpyDeviceToHost);

    FILE *writer;
    int check;
    bool folder_created = false;

    printf("writing to file... \n");
    // char sample_str[ENOUGH];
    char conc_str[ENOUGH];
    char filename[150] = "./result/state_only/";
    sprintf(conc_str, "%lf", CONC);
    strcat(filename,conc_str);
    // strcat(filename,"/");
      if (folder_created == false){
        check = mkdir(filename,0777);
        // check if directory is created or not
        if (!check){
          printf("Directory created\n");
          }
        else {
          printf("Unable to create directory\n");  
      }
      folder_created = true;
      }
      
    // strcat(filename,sample_str);
    strcat(filename,".csv");
    // sample loop
    for (int sample_id = 0; sample_id<sample_size; sample_id++){
      writer = fopen(filename,"a");
      // fprintf(writer, "Time,Vm,dVm/dt,Cai(x1.000.000)(milliM->picoM),INa(x1.000)(microA->picoA),INaL(x1.000)(microA->picoA),ICaL(x1.000)(microA->picoA),IKs(x1.000)(microA->picoA),IKr(x1.000)(microA->picoA),IK1(x1.000)(microA->picoA),Ito(x1.000)(microA->picoA)\n"); 
      for (int datapoint = 0; datapoint<num_of_states-1; datapoint++){
       // if (h_time[ sample_id + (datapoint * sample_size)] == 0.0) {continue;}

        fprintf(writer,"%lf,", // change this into string, or limit the decimal accuracy, so we can decrease filesize
        h_states[ (sample_id * num_of_states) + datapoint]

        );
      }
      fprintf(writer,"%lf\n", // write last data
        h_states[ (sample_id * num_of_states) + num_of_states-1]
        );

      fclose(writer);
    }
    toc();
    
    return 0;
	
}
