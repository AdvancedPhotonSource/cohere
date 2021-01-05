/***
Copyright (c) UChicago Argonne, LLC. All rights reserved.
See LICENSE file.
***/
// Created by Barbara Frosik

#include "stdio.h"
#include "vector"
#include "support.hpp"
#include "parameters.hpp"
#include "util.hpp"

Support::Support(const af::dim4 data_dim, Params *parameters, af::array support)
{
    params = parameters;
    threshold = params->GetSupportThreshold();
    sigma = params->GetSupportSigma();
    algorithm = params->GetSupportAlg();
    af::array ones = constant(1, Utils::Int2Dim4(params->GetSupportArea()), u32);
    if (Utils::IsNullArray(support))
    {
        support_array = Utils::PadAround(ones, data_dim, 0);
    }
    else
    {
        support_array = support;
    }
    update_iter = -1;

/*    if (algorithm == ALGORITHM_GAUSS)
    {
        int alpha = 1;
        d_type *sigmas = new d_type[nD];
        for (int i=0; i<nD; i++)
        {
            sigmas[i] = data_dim[i]/(2.0*af::Pi*sigma);
        } 
        distribution = Utils::GaussDistribution(params->GetNdim(), data_dim, sigmas, alpha);
    } 
 */   
}


Support::~Support()
{
    distribution = af::array();
    support_array = af::array();
//    init_support_array = af::array();
}

void Support::UpdateAmp(const af::array ds_image, d_type sig, int iter)
{
    
    if (sig != last_sigma)
    {
        distribution = GetDistribution(ds_image.dims(), sig);
    }

    //printf("updating support\n");
    af::array convag = GaussConvFft(abs(ds_image));
    d_type max_convag = af::max<d_type>(convag);
    convag = convag/max_convag;
    support_array = (convag >= threshold);

    last_sigma = sig;
    update_iter = iter;
}

void Support::UpdatePhase(const af::array ds_image, int iter)
{
        //printf("phase trigger\n");
        af::array phase = atan2(imag(ds_image), real(ds_image));
        af::array phase_condition = ((phase > params->GetPhaseMin()) && (phase < params->GetPhaseMax()));
        support_array *= phase_condition;
//    printf("support sum %f\n", sum<d_type>(support_array));
}

int Support::GetTriggerAlgorithm()
{
    return algorithm;
}

float Support::GetThreshold()
{
    return threshold;
}

af::array Support::GetSupportArray()
{
    return support_array;
}

af::array Support::GetDistribution(const af::dim4 data_dim, d_type sigma)
{
    int alpha = 1;
    uint nD = params->GetNdim();
    d_type *sigmas = new d_type[nD];
    for (uint i=0; i<nD; i++)
    {
        sigmas[i] = data_dim[i]/(2.0*af::Pi*sigma);
    }
    af::array dist = Utils::GaussDistribution(params->GetNdim(), data_dim, sigmas, alpha);
    return dist;
}


af::array Support::GaussConvFft(af::array ds_image_abs)
{
    d_type image_sum = sum<d_type>(ds_image_abs);
    af::array shifted = Utils::ifftshift(ds_image_abs);
    af::array rs_amplitudes = Utils::fft(shifted, params->GetNdim());
    af::array rs_amplitudes_cent = Utils::ifftshift(rs_amplitudes);
    af::array distribution = GetDistribution (ds_image_abs.dims(), 1.0);
    af::array amp_dist = rs_amplitudes_cent * distribution;
    shifted = Utils::ifftshift(amp_dist);
    af::array convag_compl = Utils::ifft(shifted, params->GetNdim());
    af::array convag = (Utils::ifftshift(convag_compl));
    convag = real(convag);
    convag(convag < 0) = 0;
    d_type correction = image_sum/sum<d_type>(convag);
    convag *= correction;
    return convag;
}

