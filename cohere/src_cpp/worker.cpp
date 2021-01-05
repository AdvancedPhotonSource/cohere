/***
Copyright (c) UChicago Argonne, LLC. All rights reserved.
See LICENSE file.
***/
// Created by Barbara Frosik

#include "fstream"
#include "unistd.h"
#include "stdio.h"
#include "worker.hpp"
#include "cstdio"
#include "cstdlib"
#include "math.h"
#include "vector"
#include "map"
#include "parameters.hpp"
#include "support.hpp"
#include "pcdi.hpp"
#include "state.hpp"
#include "common.h"
#include "util.hpp"
#include "resolution.hpp"


Reconstruction::Reconstruction(af::array image_data, af::array guess, Params* parameters, af::array support_array, af::array coherence_array)
{
    num_points = 0;
    norm_data = 0;
    current_iteration = 0;
    aver_iter = 0;
    current_error = 0.0;
    data = image_data;
    ds_image = guess;
    params = parameters;
    for (int i = 0; i < params->GetNumberIterations(); i++)
    {
         std::vector<fp> v;
         iter_flow.push_back(v);
    }
    state = new State(params);
    support = new Support(data.dims(), params, support_array);
    
    if (params->IsPcdi())
    {
        partialCoherence = new PartialCoherence(params, coherence_array);
    }
    else
    {
        partialCoherence = NULL;
    }
    if (params->IsResolution())
    {
        resolution = new Resolution(params);
    }
    else
    {
        resolution = NULL;
    }
}

Reconstruction::~Reconstruction()
{
    delete state;
    delete support;
    if (partialCoherence != NULL)
    {
        delete partialCoherence;
    }
    if (resolution != NULL)
    {
        delete resolution;
    }
    delete params;

    if (&data == &iter_data)
    {
        data = af::array();
    }
    else
    {
        data = af::array();
        iter_data = af::array();
    }
    ds_image = af::array();
    ds_image_raw = af::array();
    rs_amplitudes = af::array();

    aver_v.clear();
    support_vector.clear();
    coherence_vector.clear();
    iter_flow.clear();
    algorithm_map.clear();
    Gc();
}

void Reconstruction::Init(bool first)
{
    std::map<const char*, fp> flow_ptr_map;
    flow_ptr_map["NextIter"] = &Reconstruction::NextIter;
    flow_ptr_map["ResolutionTrigger"] =  &Reconstruction::ResolutionTrigger;
    flow_ptr_map["ShrinkWrapTrigger"] = &Reconstruction::ShrinkWrapTrigger;
    flow_ptr_map["PhaseTrigger"] = &Reconstruction::PhaseTrigger;
    flow_ptr_map["ToReciprocal"] = &Reconstruction::ToReciprocal;
    flow_ptr_map["PcdiTrigger"] = &Reconstruction::PcdiTrigger;
    flow_ptr_map["Pcdi"] = &Reconstruction::Pcdi;
    flow_ptr_map["NoPcdi"] = &Reconstruction::NoPcdi;
    flow_ptr_map["Gc"] = &Reconstruction::Gc;
    flow_ptr_map["SetPcdiPrevious"] = &Reconstruction::SetPcdiPrevious;
    flow_ptr_map["ToDirect"] = &Reconstruction::ToDirect;
    flow_ptr_map["Er"] = &Reconstruction::ModulusConstrainEr;
    flow_ptr_map["Hio"] = &Reconstruction::ModulusConstrainHio;
    flow_ptr_map["Twin"] = &Reconstruction::Twin;
    flow_ptr_map["Average"] = &Reconstruction::Average;
    flow_ptr_map["Prog"] = &Reconstruction::Progress;


    std::vector<int> used_flow_seq = params->GetUsedFlowSeq();
    std::vector<int> flow_array = params->GetFlowArray();
    int num_iter = params->GetNumberIterations();

    for (uint i = 0; i < used_flow_seq.size(); i++)
    {
        int func_order = used_flow_seq[i];
        int offset = i * num_iter;
        for (int j=0; j < num_iter; j++)
        {
            if (flow_array[offset + j])
            {
                fp func_ptr = flow_ptr_map[flow_def[func_order].func_name];
                iter_flow[j].push_back(func_ptr);
            }
        }
    }

    // initialize other components
    state->Init();
    if (partialCoherence != NULL)
    {
         partialCoherence->Init(data);
    }
   
    norm_data = GetNorm(data);
    num_points = data.elements();
    if (first)
    {
	// multiply the rs_amplitudes by max element of data array and the norm
        d_type max_data = af::max<d_type>(data);
        ds_image *= max_data * GetNorm(ds_image);

        // the next two lines are for testing it sets initial guess to initial support
        // af::array temp = support->GetSupportArray();
        // ds_image  = complex(temp.as((af_dtype) dtype_traits<d_type>::ctype), 0.0).as(c64);

        ds_image *= support->GetSupportArray();
     //   printf("initial image norm %f\n", GetNorm(ds_image));
    }

}

int Reconstruction::Iterate()
{
    while (state->Next())
    {
        current_iteration = state->GetCurrentIteration();
        if (access("stopfile", F_OK) == 0)
        {
            remove("stopfile");
            return -1;
        }
        if (anyTrue<bool>(isNaN(ds_image)))
        {
            printf("the image array has NaN element, quiting this reconstruction process\n");
            return -3;
        }
        for (uint i=0; i<iter_flow[current_iteration].size(); i++ )
        {
            (this->*iter_flow[current_iteration][i])();
        }
    }

    if (aver_v.size() > 0)
    {
        af::array aver_a(ds_image.dims(), &aver_v[0]);
        af::array ratio = Utils::GetRatio(aver_a, abs(ds_image));
        ds_image *= ratio/aver_iter;
    }
    ds_image *= support->GetSupportArray();
    VectorizeSupport();
    if (partialCoherence != NULL)
    {
        VectorizeCoherence();
    }
    return 0;
}

void Reconstruction::NextIter()
{
    iter_data = data;
    sig = params->GetSupportSigma();
//     printf("NextIter %d\n", (uint)(getpid()));
}

void Reconstruction::ResolutionTrigger()
{
    iter_data = resolution->GetIterData(current_iteration, data.copy());
    sig = resolution->GetIterSigma(current_iteration);
   //  printf("ResolutionTrigger %d\n", (uint)(getpid()));
}

void Reconstruction::ShrinkWrapTrigger()
{
    support->UpdateAmp(ds_image.copy(), sig, current_iteration);
  //   printf("SupportTrigger, support norm %fl\n", GetNorm(support->GetSupportArray()));
}

void Reconstruction::PhaseTrigger()
{
    support->UpdatePhase(ds_image.copy(), current_iteration);
   //  printf("PhaseTrigger %d\n", (uint)(getpid()));
}

void Reconstruction::ToReciprocal()
{
    rs_amplitudes = Utils::ifft(ds_image, params->GetNdim())*num_points;
 //   printf("ToReciprocal, rs_amplitudes norm %fl\n", GetNorm(rs_amplitudes));
}

void Reconstruction::PcdiTrigger()
{

    af::array abs_amplitudes = abs(rs_amplitudes).copy();
    partialCoherence->UpdatePartialCoherence(abs_amplitudes);
 //    printf("PcdiTrigger %d\n", (uint)(getpid()));
}

void Reconstruction::Pcdi()
{
    af::array abs_amplitudes = abs(rs_amplitudes).copy();
    af::array converged = partialCoherence->ApplyPartialCoherence(abs_amplitudes);
    af::array ratio = Utils::GetRatio(iter_data, abs(converged));
    current_error =  GetNorm(abs(converged)(converged > 0)-iter_data(converged > 0))/GetNorm(iter_data);
    state->RecordError(current_error);
    rs_amplitudes *= ratio;
 //    printf("Pcdi %d\n", (uint)(getpid()));
}

void Reconstruction::NoPcdi()
{
    af::array ratio = Utils::GetRatio(iter_data, abs(rs_amplitudes));
    current_error = GetNorm(abs(rs_amplitudes)(rs_amplitudes > 0)-iter_data(rs_amplitudes > 0))/GetNorm(iter_data);
    state->RecordError(current_error);
    rs_amplitudes *= ratio;
//     printf("NoPcdi, rs_amplitudes after correction %fl \n", GetNorm(rs_amplitudes));
}

void Reconstruction::Gc()
{
    af::deviceGC();
}

void Reconstruction::SetPcdiPrevious()
{
    partialCoherence->SetPrevious(abs(rs_amplitudes));
  //   printf("SetPcdiPrevious\n");
}

void Reconstruction::ToDirect()
{
    ds_image_raw = Utils::fft(rs_amplitudes, params->GetNdim())/num_points;
//     printf("ToDirect, ds_image_raw norm %fl\n", GetNorm(ds_image_raw));
}

void Reconstruction::Twin()
{
    dim4 dims = data.dims();
    std::vector<int> twin_halves = params->GetTwinHalves();
    af::array temp = constant(0, dims, u32);
    int x_start = (twin_halves[0] == 0) ? 0 : dims[0]/2;
    int x_end = (twin_halves[0] == 0) ? dims[0]/2 -1 : dims[0]-1;
    int y_start = (twin_halves[1] == 0) ? 0 : dims[1]/2;
    int y_end = (twin_halves[1] == 0) ? dims[1]/2 -1 : dims[1]-1;
    temp( af::seq(x_start, x_end), af::seq(y_start, y_end), span, span) = 1;
    ds_image = ds_image * temp;
//     printf("Twin\n");
}

void Reconstruction::Average()
{
    aver_iter++;
    af::array abs_image = abs(ds_image).copy();
    d_type *image_v = abs_image.host<d_type>();
    std::vector<d_type> v(image_v, image_v + ds_image.elements());
    if (aver_v.size() == 0)
    {
        for (uint i = 0; i < v.size(); i++)
        {
            aver_v.push_back(v[i]);
        }
    }
    else
    {
        for (uint i = 0; i < v.size(); i++)
        {
            aver_v[i] += v[i];
        }
    }
    delete [] image_v;
 //   printf("Average\n");
}

void Reconstruction::Progress()
{
    printf("------- current iteration %i, error %f -------\n", current_iteration, current_error);
}

void Reconstruction::ModulusConstrainEr()
{
//    printf("er\n");
    af::array support_array = support->GetSupportArray();
    ds_image = ds_image_raw * support_array;
//    printf("er, image norm after support %fl %d\n", GetNorm(ds_image), (uint)(getpid()));
}

void Reconstruction::ModulusConstrainHio()
{
//    printf("hio\n");
    af::array support_array = support->GetSupportArray();
    af::array adjusted_calc_image = ds_image_raw * params->GetBeta();
    af::array combined_image = ds_image - adjusted_calc_image;
    ds_image = ds_image_raw * support_array + combined_image * !support_array;
//  printf("hio, image norm after support %fl\n",GetNorm(ds_image));
}

double Reconstruction::GetNorm(af::array arr)
{
    return sum<d_type>(pow(abs(arr), 2));
}

void Reconstruction::VectorizeSupport()
{
    af::array a = support->GetSupportArray().as(f32);
    float *support_v = a.host<float>();
    std::vector<float> v(support_v, support_v + a.elements());
    support_vector = v;
    delete [] support_v;
 }

void Reconstruction::VectorizeCoherence()
{
    // get the partial coherence as double, so it will work for float and double data types
    af::array a = partialCoherence->GetKernelArray().copy();
    d_type *coherence_v = a.host<d_type>();
    std::vector<d_type> v(coherence_v, coherence_v + a.elements());
    coherence_vector = v;
    delete [] coherence_v;
}

af::array Reconstruction::GetImage()
{
    return ds_image;
}

af::array Reconstruction::GetSupportArray()
{
    return support->GetSupportArray();
}

af::array Reconstruction::GetCoherenceArray()
{
    return partialCoherence->GetKernelArray();
}

std::vector<d_type> Reconstruction::GetErrors()
{
    return state->GetErrors();
}

std::vector<float> Reconstruction::GetSupportVector()
{
    return support_vector;
}

std::vector<d_type> Reconstruction::GetCoherenceVector()
{
    return coherence_vector;
}

std::vector<d_type> Reconstruction::GetCoherenceVectorR()
{
    return coherence_vector;
}

std::vector<d_type> Reconstruction::GetCoherenceVectorI()
{
    return coherence_vector;
}

af::array Reconstruction::GetReciprocal()
{
    return rs_amplitudes;
}

std::vector<int> Reconstruction::GetFlowVector()
{
    return params->GetUsedFlowSeq();
}

std::vector<int> Reconstruction::GetIterFlowVector()
{
    return params->GetFlowArray();
}
