#include <cuda.h>
#include <cuda_runtime.h>

#include "modules/cipa_t.cuh"
#include "modules/drug_conc.hpp"
#include "modules/glob_funct.hpp"
#include "modules/glob_type.hpp"
#include "modules/gpu.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <math.h>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <vector>
namespace fs = std::filesystem;

#define ENOUGH ((CHAR_BIT * sizeof(int) - 1) / 3 + 3)
char buffer[255];

const unsigned int datapoint_size = 7500;
const unsigned int sample_limit = 10000;

clock_t START_TIMER;

clock_t tic();
void toc(clock_t start = START_TIMER);

clock_t tic() {
    return START_TIMER = clock();
}

void toc(clock_t start) {
    std::cout
        << "Elapsed time: "
        << (clock() - start) / (double)CLOCKS_PER_SEC << "s"
        << std::endl;
}

int gpu_check(unsigned int datasize) {
    int num_gpus;
    float percent;
    int id;
    size_t free, total;
    cudaGetDeviceCount(&num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        cudaSetDevice(gpu_id);
        cudaGetDevice(&id);
        cudaMemGetInfo(&free, &total);
        percent = (free / (float)total);
        printf("GPU No %d\nFree Memory: %ld, Total Memory: %ld (%f percent free)\n", id, free, total, percent * 100.0);
    }
    percent = 1.0 - (datasize / (float)total);

    return 0;
}

drug_t get_IC50_data_from_file(const char *file_name);
// return error and message based on the IC50 data
int check_IC50_content(const drug_t *ic50, const param_t *p_param);

void freeingGPUMemory(double *d_ALGEBRAIC, double *d_CONSTANTS, double *d_RATES, double *d_STATES, param_t *d_p_param, cipa_t *temp_result, cipa_t *cipa_result, double *d_STATES_RESULT, double *d_ic50);

void prepGPUMemory(double *&d_ALGEBRAIC, int num_of_algebraic, int sample_size, double *&d_CONSTANTS, int num_of_constants, double *&d_RATES, int num_of_rates, double *&d_STATES, int num_of_states, param_t *&d_p_param, cipa_t *&temp_result, cipa_t *&cipa_result, double *&d_STATES_RESULT, double *&d_ic50, double *ic50, param_t *p_param);

int get_IC50_data_from_file(const char *file_name, double *ic50) {
    /*
    a host function to take all samples from the file, assuming each sample has 14 features.

    it takes the file name, and an ic50 (already declared in 1D, everything become 1D)
    as a note, the data will be stored in 1D array, means this functions applies flatten.

    it returns 'how many samples were detected?' in integer.
    */
    FILE *fp_drugs;
    //   drug_t ic50;
    char *token;
    char buffer_ic50[255];
    unsigned int idx;

    if ((fp_drugs = fopen(file_name, "r")) == NULL) {
        printf("Cannot open file %s\n",
               file_name);
        return 0;
    }
    idx = 0;
    int sample_size = 0;
    fgets(buffer_ic50, sizeof(buffer_ic50), fp_drugs);                  // skip header
    while (fgets(buffer_ic50, sizeof(buffer_ic50), fp_drugs) != NULL) { // begin line reading
        token = strtok(buffer_ic50, ",");
        while (token != NULL) { // begin data tokenizing
            ic50[idx++] = strtod(token, NULL);
            token = strtok(NULL, ",");
        } // end data tokenizing
        sample_size++;
    } // end line reading

    fclose(fp_drugs);
    return sample_size;
}

int get_cvar_data_from_file(const char *file_name, unsigned int limit, double *cvar) {
    // buffer for writing in snprintf() function
    char buffer_cvar[255];
    FILE *fp_cvar;
    // cvar_t cvar;
    char *token;
    // std::array<double,18> temp_array;
    unsigned int idx;

    if ((fp_cvar = fopen(file_name, "r")) == NULL) {
        printf("Cannot open file %s\n",
               file_name);
    }
    idx = 0;
    int sample_size = 0;
    fgets(buffer_cvar, sizeof(buffer_cvar), fp_cvar);                                             // skip header
    while ((fgets(buffer_cvar, sizeof(buffer_cvar), fp_cvar) != NULL) && (sample_size < limit)) { // begin line reading
        token = strtok(buffer_cvar, ",");
        while (token != NULL) { // begin data tokenizing
            cvar[idx++] = strtod(token, NULL);
            // printf("%lf\n",cvar[idx]);
            token = strtok(NULL, ",");
        } // end data tokenizing
        // printf("\n");
        sample_size++;
        // cvar.push_back(temp_array);
    } // end line reading

    fclose(fp_cvar);
    return sample_size;
}

int check_IC50_content(const drug_t *ic50, const param_t *p_param) {
    if (ic50->size() == 0) {
        printf("Something problem with the IC50 file!\n");
        return 1;
    } else if (ic50->size() > 2000) {
        printf("Too much input! Maximum sample data is 2000!\n");
        return 2;
    } else if (p_param->pace_max < 750 && p_param->pace_max > 1000) {
        printf("Make sure the maximum pace is around 750 to 1000!\n");
        return 3;
    } else {
        return 0;
    }
}

int main(int argc, char **argv) {
    // enable real-time output in stdout
    setvbuf(stdout, NULL, _IONBF, 0);

    // NEW CODE STARTS HERE //
    // mycuda *thread_id;
    // cudaMalloc(&thread_id, sizeof(mycuda));

    // input variables for cell simulation
    param_t *t_param;
    t_param = new param_t();
    t_param->init();
    edison_assign_params(argc, argv, t_param);
    char drug_dir[1024];
    strcpy(drug_dir, t_param->hill_file);

    // TODO: Automation 3. check file inside folder
    for (const auto &entry : fs::directory_iterator(drug_dir)) {
        param_t *p_param, *d_p_param;
        p_param = new param_t();
        p_param->init();
        edison_assign_params(argc, argv, p_param);

        std::filesystem::directory_entry dir_entry = entry;
        std::string entry_str = dir_entry.path().string();
        std::cout << entry_str << std::endl;
        std::regex pattern("/([a-zA-Z]+)");
        std::smatch match;
        std::regex_search(entry_str, match, pattern);

        // TODO: Automation 2. create drug_name and conc
        strcpy(p_param->drug_name, match[1].str().c_str());
        strcpy(p_param->hill_file, drug_dir);
        strcat(p_param->hill_file, match[1].str().c_str());
        strcat(p_param->hill_file, "/IC50_samples.csv");

        for (int cmax = 1; cmax <= 4; cmax++) {
            p_param->conc = getValue(drugConcentration, match[1].str()) * cmax;
            p_param->show_val();

            double *ic50; // temporary
            double *cvar;

            ic50 = (double *)malloc(14 * sample_limit * sizeof(double));

            const double CONC = p_param->conc;

            double *d_ic50;
            double *d_cvar;
            double *d_ALGEBRAIC;
            double *d_CONSTANTS;
            double *d_RATES;
            double *d_STATES;
            double *d_STATES_RESULT;

            cipa_t *temp_result, *cipa_result;

            int num_of_constants = 146;
            int num_of_states = 41;
            int num_of_algebraic = 199;
            int num_of_rates = 41;

            int sample_size = get_IC50_data_from_file(p_param->hill_file, ic50);
            if (sample_size == 0)
                printf("Something problem with the IC50 file!\n");
            // else if(sample_size > 2000)
            //     printf("Too much input! Maximum sample data is 2000!\n");
            printf("Sample size: %d\n", sample_size);
            printf("Set GPU Number: %d\n", p_param->gpu_index);

            cudaSetDevice(p_param->gpu_index);

            if (p_param->is_cvar == true) {
                int cvar_sample = get_cvar_data_from_file(p_param->cvar_file, sample_size, cvar);
                printf("Reading: %d Conductance Variability samples\n", cvar_sample);
            }

            prepGPUMemory(d_ALGEBRAIC, num_of_algebraic, sample_size, d_CONSTANTS, num_of_constants, d_RATES, num_of_rates, d_STATES, num_of_states, d_p_param, temp_result, cipa_result, d_STATES_RESULT, d_ic50, ic50, p_param);

            tic();
            printf("Timer started, doing simulation.... \n\n\nGPU Usage at this moment: \n");
            int thread;
            if (sample_size >= 100) {
                thread = 100;
            } else
                thread = sample_size;
            int block = int(ceil(sample_size / thread));
            // int block = (sample_size + thread - 1) / thread;
            if (gpu_check(15 * sample_size * datapoint_size * sizeof(double) + sizeof(param_t)) == 1) {
                printf("GPU memory insufficient!\n");
                return 0;
            }
            printf("Sample size: %d\n", sample_size);
            cudaSetDevice(p_param->gpu_index);
            printf("\n   Configuration: \n\n\tblock\t||\tthread\n---------------------------------------\n  \t%d\t||\t%d\n\n\n", block, thread);
            // initscr();
            // printf("[____________________________________________________________________________________________________]  0.00 %% \n");

            kernel_DrugSimulation<<<block, thread>>>(d_ic50, d_cvar, d_CONSTANTS, d_STATES, d_RATES, d_ALGEBRAIC,
                                                     d_STATES_RESULT,
                                                     sample_size,
                                                     temp_result, cipa_result,
                                                     d_p_param);
            // block per grid, threads per block
            // endwin();

            cudaDeviceSynchronize();

            printf("allocating memory for computation result in the CPU, malloc style \n");
            double *h_states;
            cipa_t *h_cipa_result;

            h_states = (double *)malloc(num_of_states * sample_size * sizeof(double));
            h_cipa_result = (cipa_t *)malloc(sample_size * sizeof(cipa_t));
            printf("...allocating for all states, all set!\n");

            ////// copy the data back to CPU, and write them into file ////////
            printf("copying the data back to the CPU \n");

            cudaMemcpy(h_states, d_STATES_RESULT, sample_size * num_of_states * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_cipa_result, cipa_result, sample_size * sizeof(cipa_t), cudaMemcpyDeviceToHost);

            // TODO: Automation 4. Free up GPU memory
            freeingGPUMemory(d_ALGEBRAIC, d_CONSTANTS, d_RATES, d_STATES, 
                            d_p_param, temp_result, cipa_result, d_STATES_RESULT, d_ic50);

            FILE *writer;
            int check;
            bool folder_created = false;

            printf("writing to file... \n");
            char conc_str[ENOUGH];
            char filename[500] = "./result/";
            char dvmdt_file[500];
            sprintf(conc_str, "%.2lf", CONC);
            strcat(filename, p_param->drug_name);
            strcat(filename, "_");
            strcat(filename, conc_str);
            strcat(filename, "/");
            if (folder_created == false) {
                check = mkdir(filename, 0777);
                // check if directory is created or not
                if (!check) {
                    printf("Directory created\n");
                } else {
                    printf("Unable to create directory\n");
                }
                folder_created = true;
            }

            // strcat(filename,conc_str);
            strcpy(dvmdt_file, filename);
            strcat(filename, "_state_only.csv");
            // sample loop
            writer = fopen(filename, "w");
            fprintf(writer, "V,CaMKt,cass,nai,nass,ki,kss,cansr,cajsr,cai,m,hf,hs,j,hsp,jp,mL,hL,hLp,a,iF,iS,ap,iFp,iSp,d,ff,fs,fcaf,fcas,jca,ffp,fcafp,nca,xrf,xrs,xs1,xs2,xk1,Jrelnp,Jrelp,\n");
            for (int sample_id = 0; sample_id < sample_size; sample_id++) {

                // fprintf(writer,"%d,",sample_id);
                for (int datapoint = 0; datapoint < num_of_states - 1; datapoint++) {
                    // if (h_time[ sample_id + (datapoint * sample_size)] == 0.0) {continue;}
                    fprintf(writer, "%lf,", // change this into string, or limit the decimal accuracy, so we can decrease filesize
                            h_states[(sample_id * num_of_states) + datapoint]);
                }
                fprintf(writer, "%lf\n", // write last data
                        h_states[(sample_id * num_of_states) + num_of_states - 1]

                        // 22.00
                );
            }
            fclose(writer);

            // dvmdt file
            strcat(dvmdt_file, "_dvmdt.csv");
            writer = fopen(dvmdt_file, "w");
            fprintf(writer, "Sample,dVm/dt\n");
            for (int sample_id = 0; sample_id < sample_size; sample_id++) {

                fprintf(writer, "%d,%lf\n", // write last data
                        sample_id,
                        h_cipa_result[sample_id].dvmdt_repol);
            }
            fclose(writer);

            toc();
        }
    }
    return 0;
}

void prepingGPUMemory(double *&d_ALGEBRAIC, int num_of_algebraic, int sample_size, double *&d_CONSTANTS, int num_of_constants, double *&d_RATES, int num_of_rates, double *&d_STATES, int num_of_states, param_t *&d_p_param, cipa_t *&temp_result, cipa_t *&cipa_result, double *&d_STATES_RESULT, double *&d_ic50, double *ic50, param_t *p_param) {
    printf("preparing GPU memory space \n");
    cudaMalloc(&d_ALGEBRAIC, num_of_algebraic * sample_size * sizeof(double));
    cudaMalloc(&d_CONSTANTS, num_of_constants * sample_size * sizeof(double));
    cudaMalloc(&d_RATES, num_of_rates * sample_size * sizeof(double));
    cudaMalloc(&d_STATES, num_of_states * sample_size * sizeof(double));

    cudaMalloc(&d_p_param, sizeof(param_t));

    // prep for 1 cycle plus a bit (7000 * sample_size)
    cudaMalloc(&temp_result, sample_size * sizeof(cipa_t));
    cudaMalloc(&cipa_result, sample_size * sizeof(cipa_t));

    cudaMalloc(&d_STATES_RESULT, num_of_states * sample_size * sizeof(double));

    printf("Copying sample files to GPU memory space \n");
    cudaMalloc(&d_ic50, sample_size * 14 * sizeof(double));
    // cudaMalloc(&d_cvar, sample_size * 18 * sizeof(double));

    cudaMemcpy(d_ic50, ic50, sample_size * 14 * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_cvar, cvar, sample_size * 18 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_param, p_param, sizeof(param_t), cudaMemcpyHostToDevice);
}

void freeingGPUMemory(double *d_ALGEBRAIC, double *d_CONSTANTS, double *d_RATES, double *d_STATES, param_t *d_p_param, cipa_t *temp_result, cipa_t *cipa_result, double *d_STATES_RESULT, double *d_ic50) {
    cudaFree(d_ALGEBRAIC);
    cudaFree(d_CONSTANTS);
    cudaFree(d_RATES);
    cudaFree(d_STATES);
    cudaFree(d_p_param);
    cudaFree(temp_result);
    cudaFree(cipa_result);
    cudaFree(d_STATES_RESULT);
    cudaFree(d_ic50);
}
