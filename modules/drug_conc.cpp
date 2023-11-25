#include "drug_conc.hpp"

float getValue(const std::unordered_map<std::string, float>& drugConc, const std::string& key, float defaultValue) {
    auto it = drugConc.find(key);
    if (it != drugConc.end()) {
        return it->second;
    }
    return defaultValue;
}

// Instantiate the dictionary with keys and values
std::unordered_map<std::string, float> drugConcentration = {
    {"azimilide", 70.0f},
    {"bepridil", 33.0f},
    {"disopyramide", 742.0f},
    {"dofetilide", 2.0f},
    {"ibutilide", 100.0f},
    {"quinidine", 3237.0f},
    {"sotalol", 14690.0f},
    {"vandetanib", 255.0f},
    {"astemizole", 0.26f},
    {"chlorpromazine", 38.0f},
    {"cisapride", 2.6f},
    {"clarithromycin", 1206.0f},
    {"clozapine", 71.0f},
    {"domperidone", 19.0f},
    {"droperidol", 6.3f},
    {"ondansetron", 139.0f},
    {"pimozide", 0.431f},
    {"risperidone", 1.81f},
    {"terfenadine", 4.0f},
    {"diltiazem", 122.0f},
    {"loratadine", 0.45f},
    {"metoprolol", 1800.0f},
    {"mexiletine", 4129.0f},
    {"nifedipine", 7.7f},
    {"nitrendipine", 3.02f},
    {"ranolazine", 1948.2f},
    {"tamoxifen", 21.0f},
    {"verapamil", 81.0f}
};