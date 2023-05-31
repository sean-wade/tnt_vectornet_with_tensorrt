/*
 * @Author: zhanghao
 * @LastEditTime: 2023-04-18 15:53:03
 * @FilePath: /cpp/test.cpp
 * @LastEditors: zhanghao
 * @Description:
 */
#include <chrono>
#include "tnt.h"
#include "data.h"

int main2()
{
    TNTOptions options;
    options.engine_path  = "../tnt.engine";
    options.weights_path = "../tnt.wts";
    options.ues_fp16     = false;

    TrajfeatureInputData input_data;
    input_data.feats_num     = feats_num1;
    input_data.cluster_num   = cluster_num1;
    input_data.candidate_num = candidate_num1;

    input_data.feature          = feature1;
    input_data.cluster          = cluster1;
    input_data.cluster_count    = cluster_count1;
    input_data.id_embedding     = id_embedding1;
    input_data.candidate_points = candidate1;

    TNT tnt_net;
    tnt_net.Init(options);

    // For precision compare.
    TNTPredictTrajs pred_data;
    tnt_net.Process(input_data, pred_data);
    pred_data.print();

    printf("==============\n\n\n");

    TrajfeatureInputData input_data2;
    input_data2.feats_num     = feats_num2;
    input_data2.cluster_num   = cluster_num2;
    input_data2.candidate_num = candidate_num2;

    input_data2.feature          = feature2;
    input_data2.cluster          = cluster2;
    input_data2.cluster_count    = cluster_count2;
    input_data2.id_embedding     = id_embedding2;
    input_data2.candidate_points = candidate2;

    // For precision compare.
    TNTPredictTrajs pred_data2;
    tnt_net.Process(input_data2, pred_data2);
    pred_data2.print();

    printf("==============\n\n\n");

    // For precision compare.
    TNTPredictTrajs pred_data3;
    tnt_net.Process(input_data, pred_data3);
    pred_data3.print();

    // // For timing, because first time is slow, doesnot count.
    // int  loop_time = 1000;
    // auto start     = std::chrono::system_clock::now();
    // for (int k = 0; k < loop_time; k++)
    // {
    //     TNTPredictTrajs pred_data2;
    //     tnt_net.Process(input_data, pred_data2);
    // }
    // auto  end   = std::chrono::system_clock::now();
    // float mills = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "\n[INFO]: Time taken by execution: [" << mills << "] ms, "
    //           << "average per excute time: [" << mills / loop_time << "] ms." << std::endl;

    return 0;
}

int main()
{
    main2();
    // main2();
}
