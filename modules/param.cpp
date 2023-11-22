#include "param.hpp"

#include <cstdio>
#include "glob_funct.hpp"

void param_t::init()
{
  simulation_mode = 0;
  // max_samples = 10000;
  is_dutta = true;
  gpu_index = 0;
  
  is_cvar = false;
  bcl = 2000.;
  pace_max = 10;

  find_steepest_start = 5;

  celltype = 0.;
  dt = 0.005;
  // dt = 0.1;

  conc = 99.0;
  // dt_write = 2.0;
  // inet_vm_threshold = -88.0;
  
  snprintf(hill_file, sizeof(hill_file), "%s", "./drugs/bepridil/IC50_samples.csv");
  snprintf(cvar_file, sizeof(cvar_file), "%s", "./drugs/10000_pop.csv");
  snprintf(drug_name, sizeof(drug_name), "%s", "bepridil");
  // snprintf(concs, sizeof(concs), "%s", "99.0");
}

void param_t::show_val()
{
  //change this to printf somehow
  mpi_printf( 0, "%s -- %s\n", "Simulation mode", simulation_mode ? "full-pace" : "sample-based" );
  mpi_printf( 0, "%s -- %s\n", "Hill File", hill_file );
  mpi_printf( 0, "%s -- %hu\n", "Celltype", celltype);
  mpi_printf( 0, "%s -- %s\n", "Is_Dutta", is_dutta ? "true" : "false" );
  mpi_printf( 0, "%s -- %s\n", "Is_Cvar", is_cvar ? "true" : "false" );
  mpi_printf( 0, "%s -- %lf\n", "Basic_Cycle_Length", bcl);
  mpi_printf( 0, "%s -- %d\n", "GPU_Index", gpu_index);
  mpi_printf( 0, "%s -- %hu\n", "Number_of_Pacing", pace_max);
  mpi_printf( 0, "%s -- %hu\n", "Pace_Find_Steepest", find_steepest_start);
  mpi_printf( 0, "%s -- %lf\n", "Time_Step", dt);
  mpi_printf( 0, "%s -- %s\n", "Drug_Name", drug_name);
  mpi_printf( 0, "%s -- %lf\n\n\n", "Concentrations", conc);
}
