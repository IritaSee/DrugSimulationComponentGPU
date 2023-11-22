// #include "cellmodels/enums/enum_Ohara_Rudy_2011.hpp"
#include "../cellmodels/Ohara_Rudy_2011.hpp"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "glob_funct.hpp"
#include "glob_type.hpp"
#include "gpu_glob_type.cuh"
#include "gpu.cuh"


/*
all kernel function has been moved. Unlike the previous GPU code, now we seperate everything into each modules.
all modules here are optimised for GPU and slightly different than the original component based code
differences are related to GPU offset calculations
*/

__device__ void kernel_DoDrugSim(double *d_ic50, double *d_CONSTANTS, double *d_STATES, double *d_RATES, double *d_ALGEBRAIC, 
                                        double *d_STATES_RESULT, 
                                      //  double *time, double *states, double *out_dt,  double *cai_result, 
                                      //  double *ina, double *inal,
                                      //  double *ical, double *ito,
                                      //  double *ikr, double *iks, 
                                      //  double *ik1,
                                       double *tcurr, double *dt, unsigned short sample_id, unsigned int sample_size,
                                       cipa_t *temp_result, cipa_t *cipa_result,
                                       param_t *p_param
                                       )
    {
    
    unsigned int input_counter = 0;

    int num_of_constants = 146;
    int num_of_states = 41;
    int num_of_algebraic = 199;
    int num_of_rates = 41;


    // INIT STARTS
    temp_result[sample_id].qnet_ap = 0.;
    temp_result[sample_id].qnet4_ap = 0.;
    temp_result[sample_id].inal_auc_ap = 0.;
    temp_result[sample_id].ical_auc_ap = 0.;
    
    temp_result[sample_id].qnet_cl = 0.;
    temp_result[sample_id].qnet4_cl = 0.;
    temp_result[sample_id].inal_auc_cl = 0.;
    temp_result[sample_id].ical_auc_cl = 0.;
    
    temp_result[sample_id].dvmdt_repol = -999;
    temp_result[sample_id].vm_peak = -999;
    temp_result[sample_id].vm_valley = d_STATES[(sample_id * num_of_states) +V];

    cipa_result[sample_id].qnet_ap = 0.;
    cipa_result[sample_id].qnet4_ap = 0.;
    cipa_result[sample_id].inal_auc_ap = 0.;
    cipa_result[sample_id].ical_auc_ap = 0.;
    
    cipa_result[sample_id].qnet_cl = 0.;
    cipa_result[sample_id].qnet4_cl = 0.;
    cipa_result[sample_id].inal_auc_cl = 0.;
    cipa_result[sample_id].ical_auc_cl = 0.;
    
    cipa_result[sample_id].dvmdt_repol = -999;
    cipa_result[sample_id].vm_peak = -999;
    cipa_result[sample_id].vm_valley = d_STATES[(sample_id * num_of_states) +V];
    // INIT ENDS
    bool is_peak = false;
    // to search max dvmdt repol

    tcurr[sample_id] = 0.000001;
    dt[sample_id] = p_param->dt;
    double tmax;
    double max_time_step = 1.0, time_point = 25.0;
    double dt_set;

    int cipa_datapoint = 0;

    // bool writen = false;

    // files for storing results
    // time-series result
    // FILE *fp_vm, *fp_inet, *fp_gate;

    // features
    // double inet, qnet;

    // looping counter
    // unsigned short idx;
  
    // simulation parameters
    // double dtw = 2.0;
    // const char *drug_name = "bepridil";
    // const double bcl = 2000; // bcl is basic cycle length
    const double bcl = p_param->bcl;
    
    // const double inet_vm_threshold = p_param->inet_vm_threshold;
    // const unsigned short pace_max = 300;
    // const unsigned short pace_max = 1000;
    const unsigned short pace_max = p_param->pace_max;
    // const unsigned short celltype = 0.;
    // const unsigned short last_pace_print = 3;
    const unsigned short last_drug_check_pace = p_param->find_steepest_start;
    // const unsigned int print_freq = (1./dt) * dtw;
    // unsigned short pace_count = 0;
    // unsigned short pace_steepest = 0;
    double conc = p_param->conc; //mmol
    double type = p_param->celltype;
    double epsilon = 10E-14;
    // double top_dvmdt = -999.0;

    // eligible AP shape means the Vm_peak > 0.
    bool is_eligible_AP;
    // Vm value at 30% repol, 50% repol, and 90% repol, respectively.
    double vm_repol30, vm_repol50, vm_repol90;
    double t_peak_capture = 0.0;
    unsigned short pace_steepest = 0;

    bool init_states_captured = false;

    // qnet_ap/inet_ap values
	  double inet_ap, qnet_ap, inet4_ap, qnet4_ap, inet_cl, qnet_cl, inet4_cl, qnet4_cl;
	  double inal_auc_ap, ical_auc_ap,inal_auc_cl, ical_auc_cl;
    // qinward_cl;

    // char buffer[255];

    // static const int CALCIUM_SCALING = 1000000;
	  // static const int CURRENT_SCALING = 1000;

    // printf("Core %d:\n",sample_id);
    initConsts(d_CONSTANTS, d_STATES, type, conc, d_ic50, d_cvar, p_param->is_dutta, p_param->is_cvar, sample_id);
    

    applyDrugEffect(d_CONSTANTS, conc, d_ic50, epsilon, sample_id);

    d_CONSTANTS[BCL + (sample_id * num_of_constants)] = bcl;

    // generate file for time-series output

    tmax = pace_max * bcl;
    int pace_count = 0;
    
  
    // printf("%d,%lf,%lf,%lf,%lf\n", sample_id, dt[sample_id], tcurr[sample_id], d_STATES[V + (sample_id * num_of_states)],d_RATES[V + (sample_id * num_of_rates)]);
    // printf("%lf,%lf,%lf,%lf,%lf\n", d_ic50[0 + (14*sample_id)], d_ic50[1+ (14*sample_id)], d_ic50[2+ (14*sample_id)], d_ic50[3+ (14*sample_id)], d_ic50[4+ (14*sample_id)]);

    while (tcurr[sample_id]<tmax)
    {
        computeRates(tcurr[sample_id], d_CONSTANTS, d_RATES, d_STATES, d_ALGEBRAIC, sample_id); 
        
        dt_set = set_time_step( tcurr[sample_id], time_point, max_time_step, 
        d_CONSTANTS, 
        d_RATES, 
        d_STATES, 
        d_ALGEBRAIC, 
        sample_id); 
        
        // printf("tcurr at core %d: %lf\n",sample_id,tcurr[sample_id]);
        if (floor((tcurr[sample_id] + dt_set) / bcl) == floor(tcurr[sample_id] / bcl)) { 
          dt[sample_id] = dt_set;
          // printf("dt : %lf\n",dt_set);
          // it goes in here, but it does not, you know, adds the pace, 
        }
        else{
          dt[sample_id] = (floor(tcurr[sample_id] / bcl) + 1) * bcl - tcurr[sample_id];

          // new part starts
          if( is_eligible_AP && pace_count >= pace_max-last_drug_check_pace) {
            temp_result[sample_id].qnet_ap = qnet_ap;
            temp_result[sample_id].qnet4_ap = qnet4_ap;
            temp_result[sample_id].inal_auc_ap = inal_auc_ap;
            temp_result[sample_id].ical_auc_ap = ical_auc_ap;
            temp_result[sample_id].qnet_cl = qnet_cl;
            temp_result[sample_id].qnet4_cl = qnet4_cl;
            temp_result[sample_id].inal_auc_cl = inal_auc_cl;
            temp_result[sample_id].ical_auc_cl = ical_auc_cl;
            // fprintf(fp_vmdebug, "%hu,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf\n", pace_count,t_peak_capture,temp_result.vm_peak,vm_repol30,vm_repol50,vm_repol90,temp_result.dvmdt_repol);
            // replace result with steeper repolarization AP or first pace from the last 250 paces
            // if( temp_result->dvmdt_repol > cipa_result.dvmdt_repol ) {
            //   pace_steepest = pace_count;
            //   cipa_result = temp_result;
            //   }
            if( temp_result[sample_id].dvmdt_repol > cipa_result[sample_id].dvmdt_repol ) {
              pace_steepest = pace_count;
              // printf("Steepest pace updated: %d dvmdt_repol: %lf\n",pace_steepest,temp_result[sample_id].dvmdt_repol);
              // cipa_result = temp_result;
              cipa_result[sample_id].qnet_ap = temp_result[sample_id].qnet_ap;
              cipa_result[sample_id].qnet4_ap = temp_result[sample_id].qnet4_ap;
              cipa_result[sample_id].inal_auc_ap = temp_result[sample_id].inal_auc_ap;
              cipa_result[sample_id].ical_auc_ap = temp_result[sample_id].ical_auc_ap;
              
              cipa_result[sample_id].qnet_cl = temp_result[sample_id].qnet_cl;
              cipa_result[sample_id].qnet4_cl = temp_result[sample_id].qnet4_cl;
              cipa_result[sample_id].inal_auc_cl = temp_result[sample_id].inal_auc_cl;
              cipa_result[sample_id].ical_auc_cl = temp_result[sample_id].ical_auc_cl;
              
              cipa_result[sample_id].dvmdt_repol = temp_result[sample_id].dvmdt_repol;
              cipa_result[sample_id].vm_peak = temp_result[sample_id].vm_peak;
              cipa_result[sample_id].vm_valley = d_STATES[(sample_id * num_of_states) +V];
              is_peak = true;
              init_states_captured = false;
              }
            else{
              is_peak = false;
            }
          };
          inet_ap = 0.;
          qnet_ap = 0.;
          inet4_ap = 0.;
          qnet4_ap = 0.;
          inal_auc_ap = 0.;
          ical_auc_ap = 0.;
          inet_cl = 0.;
          qnet_cl = 0.;
          inet4_cl = 0.;
          qnet4_cl = 0.;
          inal_auc_cl = 0.;
          ical_auc_cl = 0.;
          t_peak_capture = 0.;

          // temp_result->init( p_cell->STATES[V]);	
          temp_result[sample_id].qnet_ap = 0.;
          temp_result[sample_id].qnet4_ap = 0.;
          temp_result[sample_id].inal_auc_ap = 0.;
          temp_result[sample_id].ical_auc_ap = 0.;
          
          temp_result[sample_id].qnet_cl = 0.;
          temp_result[sample_id].qnet4_cl = 0.;
          temp_result[sample_id].inal_auc_cl = 0.;
          temp_result[sample_id].ical_auc_cl = 0.;
          
          temp_result[sample_id].dvmdt_repol = -999;
          temp_result[sample_id].vm_peak = -999;
          temp_result[sample_id].vm_valley = d_STATES[(sample_id * num_of_states) +V];
          // end of init

          pace_count++;
          input_counter = 0; // at first, we reset the input counter since we re gonna only take one, but I remember we don't have this kind of thing previously, so do we need this still?
          cipa_datapoint = 0; // new pace? reset variables related to saving the values,
              
          is_eligible_AP = false;
          // new part ends
           if(sample_id == 1000 || sample_id == 2000 || sample_id == 3000 || sample_id == 4000 || sample_id == 5000 || sample_id == 6000 || sample_id == 7000 || sample_id == 8000 || sample_id == 9000 ){
            printf("core: %d pace count: %d t: %lf, steepest: %d, dvmdt_repol: %lf\n",sample_id,pace_count, tcurr[sample_id], pace_steepest, cipa_result[sample_id].dvmdt_repol);
          }
          // printf("core: %d pace count: %d t: %lf, steepest: %d, dvmdt_repol: %lf, t_peak: %lf\n",sample_id,pace_count, tcurr[sample_id], pace_steepest, cipa_result[sample_id].dvmdt_repol,t_peak_capture);
          // writen = false;
        }
        

        //// progress bar starts ////

        // if(sample_id==0 && pace_count%10==0 && pace_count>99 && !writen){
        // // printf("Calculating... watching core 0: %.2lf %% done\n",(tcurr[sample_id]/tmax)*100.0);
        // printf("[");
        // for (cnt=0; cnt<pace_count/10;cnt++){
        //   printf("=");
        // }
        // for (cnt=pace_count/10; cnt<pace_max/10;cnt++){
        //   printf("_");
        // }
        // printf("] %.2lf %% \n",(tcurr[sample_id]/tmax)*100.0);
        // //mvaddch(0,pace_count,'=');
        // //refresh();
        // //system("clear");
        // writen = true;
        // }

        // //// progress bar ends ////

        solveAnalytical(d_CONSTANTS, d_STATES, d_ALGEBRAIC, d_RATES,  dt[sample_id], sample_id);
        // tcurr[sample_id] = tcurr[sample_id] + dt[sample_id];
        // __syncthreads();
        // printf("solved analytical\n"); 
        // it goes here, so it means, basically, floor((tcurr[sample_id] + dt_set) / bcl) == floor(tcurr[sample_id] / bcl) is always true

        // begin the last 250 pace operations

        if (pace_count >= pace_max-last_drug_check_pace)
        {
          // printf("last 250 ops\n");
			    // Find peak vm around 2 msecs and  40 msecs after stimulation
			    // and when the sodium current reach 0
          // new codes start here
          // printf("a: %d, b: %d, c: %d, eligible ap: %d\n",
          // tcurr[sample_id] > ((d_CONSTANTS[(sample_id * num_of_constants) +BCL]*pace_count)+(d_CONSTANTS[(sample_id * num_of_constants) +stim_start]+2)),
          // tcurr[sample_id] < ((d_CONSTANTS[(sample_id * num_of_constants) +BCL]*pace_count)+(d_CONSTANTS[(sample_id * num_of_constants) +stim_start]+10)),
          // abs(d_ALGEBRAIC[(sample_id * num_of_algebraic) +INa]) < 1,
          // is_eligible_AP
          // );
          
			    if( tcurr[sample_id] > ((d_CONSTANTS[(sample_id * num_of_constants) +BCL]*pace_count)+(d_CONSTANTS[(sample_id * num_of_constants) +stim_start]+2)) && 
				      tcurr[sample_id] < ((d_CONSTANTS[(sample_id * num_of_constants) +BCL]*pace_count)+(d_CONSTANTS[(sample_id * num_of_constants) +stim_start]+10)) && 
				      abs(d_ALGEBRAIC[(sample_id * num_of_algebraic) +INa]) < 1)
          {
            // printf("check 1\n");
            if( d_STATES[(sample_id * num_of_states) +V] > temp_result[sample_id].vm_peak )
            {
              temp_result[sample_id].vm_peak = d_STATES[(sample_id * num_of_states) +V];
              if(temp_result[sample_id].vm_peak > 0)
              {
                vm_repol30 = temp_result[sample_id].vm_peak - (0.3 * (temp_result[sample_id].vm_peak - temp_result[sample_id].vm_valley));
                vm_repol50 = temp_result[sample_id].vm_peak - (0.5 * (temp_result[sample_id].vm_peak - temp_result[sample_id].vm_valley));
                vm_repol90 = temp_result[sample_id].vm_peak - (0.9 * (temp_result[sample_id].vm_peak - temp_result[sample_id].vm_valley));
                is_eligible_AP = true;
                t_peak_capture = tcurr[sample_id];
                // printf("check 2\n");
              }
              else is_eligible_AP = false;
            }
			    }
			    else if( tcurr[sample_id] > ((d_CONSTANTS[(sample_id * num_of_constants) +BCL]*pace_count)+(d_CONSTANTS[(sample_id * num_of_constants) +stim_start]+10)) && is_eligible_AP )
          {
            // printf("check 3\n");
            // printf("rates: %lf, dvmdt_repol: %lf\n states: %lf vm30: %lf, vm90: %lf\n",
            // d_RATES[(sample_id * num_of_rates) +V],
            // temp_result->dvmdt_repol, 
            // d_STATES[(sample_id * num_of_states) +V],
            // vm_repol30,
            // vm_repol90
            // );
				    if( d_RATES[(sample_id * num_of_rates) +V] > temp_result[sample_id].dvmdt_repol &&
					      d_STATES[(sample_id * num_of_states) +V] <= vm_repol30 &&
					      d_STATES[(sample_id * num_of_states) +V] >= vm_repol90 )
              {
					      temp_result[sample_id].dvmdt_repol = d_RATES[(sample_id * num_of_rates) +V];
                // printf("check 4\n");
				      }
          }
			    // calculate AP shape
			    if(is_eligible_AP && d_STATES[(sample_id * num_of_states) +V] > vm_repol90)
          {
            // printf("check 5 (eligible)\n");
          // inet_ap/qnet_ap under APD.
          inet_ap = (d_ALGEBRAIC[(sample_id * num_of_algebraic) +INaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +ICaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +Ito]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IKr]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IKs]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IK1]);
          inet4_ap = (d_ALGEBRAIC[(sample_id * num_of_algebraic) +INaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +ICaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IKr]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +INa]);
          qnet_ap += (inet_ap * dt[sample_id])/1000.;
          qnet4_ap += (inet4_ap * dt[sample_id])/1000.;
          inal_auc_ap += (d_ALGEBRAIC[(sample_id * num_of_algebraic) +INaL]*dt[sample_id]);
          ical_auc_ap += (d_ALGEBRAIC[(sample_id * num_of_algebraic) +ICaL]*dt[sample_id]);
			    }
          // inet_ap/qnet_ap under Cycle Length
          inet_cl = (d_ALGEBRAIC[(sample_id * num_of_algebraic) +INaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +ICaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +Ito]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IKr]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IKs]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IK1]);
          inet4_cl = (d_ALGEBRAIC[(sample_id * num_of_algebraic) +INaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +ICaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IKr]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +INa]);
          qnet_cl += (inet_cl * dt[sample_id])/1000.;
          qnet4_cl += (inet4_cl * dt[sample_id])/1000.;
          inal_auc_cl += (d_ALGEBRAIC[(sample_id * num_of_algebraic) +INaL]*dt[sample_id]);
          ical_auc_cl += (d_ALGEBRAIC[(sample_id * num_of_algebraic) +ICaL]*dt[sample_id]);

          // save temporary result -> ALL TEMP RESULTS IN, TEMP RESULT != WRITTEN RESULT
          if((pace_count >= pace_max-last_drug_check_pace) && (is_peak == true) && (pace_count<pace_max) )
          {
            // printf("input_counter: %d\n",input_counter);
            // datapoint_at_this_moment = tcurr[sample_id] - (pace_count * bcl);
            temp_result[sample_id].cai_data[cipa_datapoint] =  d_STATES[(sample_id * num_of_states) +cai] ;
            temp_result[sample_id].cai_time[cipa_datapoint] =  tcurr[sample_id];

            temp_result[sample_id].vm_data[cipa_datapoint] = d_STATES[(sample_id * num_of_states) +V];
            temp_result[sample_id].vm_time[cipa_datapoint] = tcurr[sample_id];

            temp_result[sample_id].dvmdt_data[cipa_datapoint] = d_RATES[(sample_id * num_of_rates) +V];
            temp_result[sample_id].dvmdt_time[cipa_datapoint] = tcurr[sample_id];

            if(init_states_captured == false){
              for(int counter=0; counter<num_of_states; counter++){
                d_STATES_RESULT[(sample_id * num_of_states) + counter] = d_STATES[(sample_id * num_of_states) + counter];
              }
              init_states_captured = true;
            }

            // time series result

            // time[input_counter + sample_id] = tcurr[sample_id];
            // states[input_counter + sample_id] = d_STATES[V + (sample_id * num_of_states)];
            
            // out_dt[input_counter + sample_id] = d_RATES[V + (sample_id * num_of_states)];

            
            // cai_result[input_counter + sample_id] = d_ALGEBRAIC[cai + (sample_id * num_of_algebraic)];

            // ina[input_counter + sample_id] = d_ALGEBRAIC[INa + (sample_id * num_of_algebraic)] ;
            // inal[input_counter + sample_id] = d_ALGEBRAIC[INaL + (sample_id * num_of_algebraic)] ;

            // ical[input_counter + sample_id] = d_ALGEBRAIC[ICaL + (sample_id * num_of_algebraic)] ;
            // ito[input_counter + sample_id] = d_ALGEBRAIC[Ito + (sample_id * num_of_algebraic)] ;

            // ikr[input_counter + sample_id] = d_ALGEBRAIC[IKr + (sample_id * num_of_algebraic)] ;
            // iks[input_counter + sample_id] = d_ALGEBRAIC[IKs + (sample_id * num_of_algebraic)] ;

            // ik1[input_counter + sample_id] = d_ALGEBRAIC[IK1 + (sample_id * num_of_algebraic)] ;

            input_counter = input_counter + sample_size;
            cipa_datapoint = cipa_datapoint + 1; // this causes the resource usage got so mega and crashed in running

           
             } // temporary guard ends here

		    } // end the last 250 pace operations
        tcurr[sample_id] = tcurr[sample_id] + dt[sample_id];
        //printf("t after addition: %lf\n", tcurr[sample_id]);
       
    } // while loop ends here 
    // __syncthreads();
}



__global__ void kernel_DrugSimulation(double *d_ic50, double *d_CONSTANTS, double *d_STATES, double *d_RATES, double *d_ALGEBRAIC, 
                                      double *d_STATES_RESULT, 
                                      // double *time, double *states, double *out_dt,  double *cai_result, 
                                      // double *ina, double *inal, 
                                      // double *ical, double *ito,
                                      // double *ikr, double *iks,
                                      // double *ik1,
                                      unsigned int sample_size,
                                      cipa_t *temp_result, cipa_t *cipa_result,
                                      param_t *p_param
                                      )
  {
    unsigned short thread_id;
    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    double time_for_each_sample[10000];
    double dt_for_each_sample[10000];
    // cipa_t temp_per_sample[2000];
    // cipa_t cipa_per_sample[2000];
    // printf("in\n");
    // printf("Calculating %d\n",thread_id);
    kernel_DoDrugSim(d_ic50, d_CONSTANTS, d_STATES, d_RATES, d_ALGEBRAIC, 
                          d_STATES_RESULT, 
                          // time, states, out_dt, cai_result,
                          // ina, inal, 
                          // ical, ito,
                          // ikr, iks, 
                          // ik1,
                          time_for_each_sample, dt_for_each_sample, thread_id, sample_size,
                          temp_result, cipa_result,
                          p_param
                          );
                          // __syncthreads();
    // printf("Calculation for core %d done\n",sample_id);
  }