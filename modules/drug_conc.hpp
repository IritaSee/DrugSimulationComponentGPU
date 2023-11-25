#ifndef DRUG_CONC_HPP
#define DRUG_CONC_HPP

#include <unordered_map>
#include <string>

// Declare the dictionary function template
float getValue(const std::unordered_map<std::string, float>& drugConc, const std::string& key, float defaultValue = 0.0);

// Declare the dictionary extern
extern std::unordered_map<std::string, float> drugConcentration;
#endif  // DRUG_CONC_HPP