/***
Copyright (c) UChicago Argonne, LLC. All rights reserved.
See LICENSE file.
***/
// Created by Barbara Frosik

#ifndef bridge_hpp
#define bridge_hpp

#include "vector"
#include "string"
#include "common.h"

class Manager;

class Bridge
{
private:
    Manager *mgr;
    
public:
    Bridge();

    int StartCalcWithGuess(int device, std::vector<float> data_buffer_r, std::vector<float> guess_buffer_r, std::vector<float> guess_buffer_i, std::vector<int> dim, const std::string & config);

    int StartCalcWithGuessSupport(int device, std::vector<float> data_buffer_r, std::vector<float> guess_buffer_r, std::vector<float> guess_buffer_i, std::vector<int> support_buffer, std::vector<int> dim, const std::string & config, int start_coh);

    int StartCalcWithGuessSupportCoh(int device, std::vector<float> data_buffer_r, std::vector<float> guess_buffer_r, std::vector<float> guess_buffer_i, std::vector<int> support_buffer, std::vector<int> dim, std::vector<float> coh_buffer, std::vector<int> coh_dim, const std::string & config);

    int StartCalc(int device, std::vector<float> data_buffer_r, std::vector<int> dim, std::string const & config, int start_coh);

    std::vector<d_type> GetReciprocalR();
    std::vector<d_type> GetReciprocalI();
    std::vector<float> GetSupportV();
    std::vector<d_type> GetCoherenceV();
    std::vector<d_type> GetImageR();
    std::vector<d_type> GetImageI();
    std::vector<d_type> GetErrors();
    std::vector<int> GetFlowV();
    std::vector<int> GetIterFlowV();

    void Cleanup();
};


#endif /* bridge_hpp */
