#ifndef PARAM_HPP
#define PARAM_HPP

struct param_t
{
  unsigned short simulation_mode; // toggle between sample-based or full-pace simulations
  bool is_dutta; // TRUE if using Dutta scaling
  unsigned short gpu_index;
  bool is_print_graph; // TRUE if we want to print graph
  bool is_using_output; // TRUE if using last output file
  bool is_cvar;
  double bcl; // basic cycle length
  // unsigned int max_samples;
  unsigned short pace_max; // maximum pace
  unsigned short find_steepest_start;
  unsigned short celltype;  // cell types
  double dt;        // time step
  double dt_write;  // writing step
  double inet_vm_threshold; // Vm threshold for calculating inet
  char hill_file[1024];
  char cvar_file[1024];
  char drug_name[100];
  char concs[100];
  float conc;
  void init();
  void show_val();
};

#endif
