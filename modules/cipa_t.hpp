#ifndef CIPA_T_HPP
#define CIPA_T_HPP

#include <map>
#include <string>

using std::multimap;
using std::string;

struct cipa_t{
  double qnet_ap;
  double qnet4_ap;
  double inal_auc_ap;
  double ical_auc_ap;
  double qnet_cl;
  double qnet4_cl;
  double inal_auc_cl;
  double ical_auc_cl;
  
  double dvmdt_repol;
  double vm_peak;
  double vm_valley;
  multimap<double, double> vm_data;
  multimap<double, double> dvmdt_data;
  multimap<double, double> cai_data;
  multimap<double, string> ires_data;
  
  multimap<double, string> inet_data;
  multimap<double, string> qnet_data;
  multimap<double, string> inet4_data;
  multimap<double, string> qnet4_data;
  
  multimap<double, string> time_series_data;

  cipa_t();
  cipa_t( const cipa_t &source );
  cipa_t& operator=(const cipa_t & source);
  void copy(const cipa_t &source);
  void init(const double vm_val);
  void clear_time_result();


};


#endif
