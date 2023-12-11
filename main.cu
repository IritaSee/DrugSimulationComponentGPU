#include <cuda.h>
#include <cuda_runtime.h>

#include "modules/cipa_t.cuh"
#include "modules/drug_conc.hpp"
#include "modules/glob_funct.hpp"
#include "modules/glob_type.hpp"
#include "modules/gpu.cuh"

#include <algorithm>
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
#include <unordered_map>
#include <vector>
namespace fs = std::filesystem;

#define ENOUGH ((CHAR_BIT * sizeof(int) - 1) / 3 + 3)
char buffer[255];

const unsigned int datapoint_size = 7500;
const unsigned int sample_limit = 10000;

// Custom comparison function for C-style strings
struct CStringCompare {
    bool operator()(const char *lhs, const char *rhs) const {
        return std::strcmp(lhs, rhs) < 0;
    }
};

// Function to get unique C-style strings from an array
// const char** getUniqueStrings(const char** stringArray, size_t arraySize, size_t& uniqueSize) {
//     // Use std::set with custom comparison function to get unique values
//     std::set<const char*, CStringCompare> uniqueStringsSet(stringArray, stringArray + arraySize);

//     // Create a new array to store unique strings
//     const char** uniqueArray = new const char*[uniqueStringsSet.size()];
//     size_t index = 0;

//     // Copy unique strings to the new array
//     for (const auto& element : uniqueStringsSet) {
//         uniqueArray[index++] = element;
//     }

//     // Update the output parameter with the size of the unique array
//     uniqueSize = uniqueStringsSet.size();

//     return uniqueArray;
// }

// Function to create a bag of words with insertion order from an array of C-style strings
std::map<std::string, int> createBagOfWords(char *textArray[], size_t arraySize, std::vector<std::string> &insertionOrder) {
    std::map<std::string, int> bagOfWords;

    for (size_t i = 0; i < arraySize; ++i) {
        // Copy the string to a mutable buffer
        char buffer[32]; // Adjust the size as needed
        strcpy(buffer, textArray[i]);
        insertionOrder.push_back(buffer);
    }

    return bagOfWords;
}

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

void addDrugData(char*** arrayOfStrings, int& size, const char newString[]) {
    // Allocate memory for a new array with increased size
    char** newArray = new char*[size + 1];

    // Copy existing strings to the new array
    for (int i = 0; i < size; ++i) {
        newArray[i] = new char[strlen((*arrayOfStrings)[i]) + 1];
        strcpy(newArray[i], (*arrayOfStrings)[i]);
        delete[] (*arrayOfStrings)[i]; // Deallocate memory for old strings
    }

    // Allocate memory for the new string and copy it
    newArray[size] = new char[strlen(newString) + 1];
    strcpy(newArray[size], newString);

    // Deallocate memory for the old array
    delete[] *arrayOfStrings;

    // Update the pointer to point to the new array
    *arrayOfStrings = newArray;

    // Increment the size
    ++size;
}

void prepingGPUMemory(double *&d_ALGEBRAIC, int num_of_algebraic, int sample_size, double *&d_CONSTANTS, int num_of_constants, double *&d_RATES, int num_of_rates, double *&d_STATES, int num_of_states, param_t *&d_p_param, cipa_t *&temp_result, cipa_t *&cipa_result, double *&d_STATES_RESULT, double *&d_ic50, double *ic50, double *&d_conc, double *conc, param_t *p_param) {
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
    cudaMalloc(&d_conc, sample_size * sizeof(double));

    cudaMemcpy(d_ic50, ic50, sample_size * 14 * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_cvar, cvar, sample_size * 18 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conc, conc, sample_size * sizeof(double), cudaMemcpyHostToDevice);
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

// TODO: NewFile 3. Create new function that takes several params
int get_IC50_data_from_file(const char *file_name, double *ic50, double *conc, char **drug_name) {
    /*
    a host function to take all samples from the file, assuming each sample has 14 features.

    it takes the file name, and an ic50 (already declared in 1D, everything become 1D)
    as a note, the data will be stored in 1D array, means this functions applies flatten.

    it returns 'how many samples were detected?' in integer.
    */
    FILE *fp_drugs;
    //   drug_t ic50;
    char *token;
    char tmp_drug_name[32];
    char buffer_ic50[255];
    unsigned int idx_ic50, idx_conc;
    int drugsize = 0;

    if ((fp_drugs = fopen(file_name, "r")) == NULL) {
        printf("Cannot open file %s\n",
               file_name);
        return 0;
    }
    idx_ic50 = 0;
    idx_conc = 0;
    int sample_size = 0;
    fgets(buffer_ic50, sizeof(buffer_ic50), fp_drugs);                  // skip header
    while (fgets(buffer_ic50, sizeof(buffer_ic50), fp_drugs) != NULL) { // begin line reading
        /*
        TODO: Extracting token from file
        1. take token for each file
        2. check the first token to drug_name, if already exist in array, then skip it
        3. check the second token to conc
        */

        token = strtok(buffer_ic50, ",");
        printf("%s\n", token); // testingAuto
        strcpy(tmp_drug_name, token);
        token = strtok(NULL, ",");
        printf("%s\n", token); // testingAuto
        strcat(tmp_drug_name, "_");
        strcat(tmp_drug_name, token);
        
        printf("%s\n", tmp_drug_name); // testingAuto
        addDrugData(&drug_name, drugsize, tmp_drug_name);
        conc[idx_conc++] = strtod(token, NULL);
        token = strtok(NULL, ",");
        // Check if there is wrong in here
        while (token != NULL) { // begin data tokenizing
            ic50[idx_ic50++] = strtod(token, NULL);
            printf("%s\n", token); // testingAuto
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

drug_t get_IC50_data_from_file(const char *file_name);
// return error and message based on the IC50 data
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
    /* TODO: Creating new init state that takes new file format
     * 1. Set the mechanism to iterate over file inside folder
     * 2. Take value of the csv file and put it inside pointer
     * 3. Create a filename based on first and second column similarity
     * 4.
     *
     */
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
        std::regex pattern("/([a-zA-Z0-9_\.]+)\.csv");
        std::smatch match;
        std::regex_search(entry_str, match, pattern);

        // TODO: Automation 2. create drug_name and conc

        // TODO: NewFile 2. disable drug name for now since the file name is inside it
        // strcpy(p_param->drug_name, match[1].str().c_str());
        strcpy(p_param->hill_file, entry_str.c_str());
        // strcat(p_param->hill_file, ".csv");
        // strcat(p_param->hill_file, "/IC50_samples.csv");

        // TODO: NewFile 3. getvalue from source is unnecessary
        // p_param->conc = getValue(drugConcentration, match[1].str()) * cmax;
        // p_param->show_val();

        double *ic50; // temporary
        double *cvar;
        double *conc;
        char **drug_name = nullptr;

        ic50 = (double *)malloc(14 * sample_limit * sizeof(double));
        conc = (double *)malloc(sample_limit * sizeof(double));

        double *d_ic50;
        double *d_conc;
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

        printf("%s\n", p_param->hill_file); // testingAuto
        int sample_size = get_IC50_data_from_file(p_param->hill_file, ic50, conc, drug_name);
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

        prepingGPUMemory(d_ALGEBRAIC, num_of_algebraic, sample_size, d_CONSTANTS, num_of_constants, d_RATES, num_of_rates, d_STATES, num_of_states, d_p_param, temp_result, cipa_result, d_STATES_RESULT, d_ic50, ic50, d_conc, conc, p_param);

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

        kernel_DrugSimulation<<<block, thread>>>(d_ic50, d_cvar, d_conc, d_CONSTANTS, d_STATES, d_RATES, d_ALGEBRAIC,
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
        printf("Successfully reach here!!");
        // TODO: Automation 4. Free up GPU memory
        freeingGPUMemory(d_ALGEBRAIC, d_CONSTANTS, d_RATES, d_STATES,
                         d_p_param, temp_result, cipa_result, d_STATES_RESULT, d_ic50);

        FILE *writer;
        int check;
        bool folder_created = false;

        // TODO: writing to several files
        printf("writing to file... \n");
        char filename[500] = "./result/init_";
        char dvmdt_file[500];
        strcat(filename, match[1].str().c_str());
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
    return 0;
}