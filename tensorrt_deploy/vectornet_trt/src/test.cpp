/*
 * @Author: zhanghao
 * @LastEditTime: 2023-03-24 16:48:29
 * @FilePath: /vectornetx/test.cpp
 * @LastEditors: zhanghao
 * @Description:
 */
#include <chrono>
#include "vectornet.h"

int main()
{
    VectorNetOptions options;
    options.engine_path  = "/home/zhanghao/code/master/6_DEPLOY/vectornetx/data/vectornet.engine";
    options.weights_path = "/home/zhanghao/code/master/6_DEPLOY/vectornetx/data/vectornet.wts";
    options.ues_fp16     = true;

    TrajfeatureInputData input_data;
    input_data.feats_num   = 109;
    input_data.cluster_num = 7;

    float feature[INPUT_CHANNEL * input_data.feats_num] = {
        7.436824589967727661e-02,  -1.142864322662353516e+01, 3.525845706462860107e-03,  6.032838821411132812e-01,
        0.000000000000000000e+00,  0.000000000000000000e+00,  7.789409160614013672e-02,  -1.082535934448242188e+01,
        1.645401120185852051e-03,  6.033983230590820312e-01,  1.000000000000000000e+00,  0.000000000000000000e+00,
        7.953949272632598877e-02,  -1.022196102142333984e+01, -1.331865787506103516e-04, 6.031322479248046875e-01,
        2.000000000000000000e+00,  0.000000000000000000e+00,  7.940630614757537842e-02,  -9.618828773498535156e+00,
        -1.426056027412414551e-03, 6.027584075927734375e-01,  3.000000000000000000e+00,  0.000000000000000000e+00,
        7.798025012016296387e-02,  -9.016070365905761719e+00, -2.937324345111846924e-03, 6.034059524536132812e-01,
        4.000000000000000000e+00,  0.000000000000000000e+00,  7.504292577505111694e-02,  -8.412664413452148438e+00,
        -3.392755985260009766e-03, 6.036572456359863281e-01,  5.000000000000000000e+00,  0.000000000000000000e+00,
        7.165016978979110718e-02,  -7.809007167816162109e+00, -5.414791405200958252e-03, 6.033630371093750000e-01,
        6.000000000000000000e+00,  0.000000000000000000e+00,  6.623537838459014893e-02,  -7.205644130706787109e+00,
        -6.986964493989944458e-03, 6.030435562133789062e-01,  7.000000000000000000e+00,  0.000000000000000000e+00,
        5.924841389060020447e-02,  -6.602600574493408203e+00, -7.487453520298004150e-03, 6.025609970092773438e-01,
        8.000000000000000000e+00,  0.000000000000000000e+00,  5.176096037030220032e-02,  -6.000039577484130859e+00,
        -7.645368576049804688e-03, 6.014323234558105469e-01,  9.000000000000000000e+00,  0.000000000000000000e+00,
        4.411559179425239563e-02,  -5.398607254028320312e+00, -7.742304354906082153e-03, 6.013598442077636719e-01,
        1.000000000000000000e+01,  0.000000000000000000e+00,  3.637328743934631348e-02,  -4.797247409820556641e+00,
        -8.678562939167022705e-03, 6.014933586120605469e-01,  1.100000000000000000e+01,  0.000000000000000000e+00,
        2.769472450017929077e-02,  -4.195754051208496094e+00, -9.098321199417114258e-03, 6.010248661041259766e-01,
        1.200000000000000000e+01,  0.000000000000000000e+00,  1.859640330076217651e-02,  -3.594729185104370117e+00,
        -8.032076992094516754e-03, 6.006155014038085938e-01,  1.300000000000000000e+01,  0.000000000000000000e+00,
        1.056432630866765976e-02,  -2.994113683700561523e+00, -5.823119543492794037e-03, 6.002724170684814453e-01,
        1.400000000000000000e+01,  0.000000000000000000e+00,  4.741206765174865723e-03,  -2.393841266632080078e+00,
        -4.741271026432514191e-03, 5.994819402694702148e-01,  1.500000000000000000e+01,  0.000000000000000000e+00,
        -6.420223286340842606e-08, -1.794359326362609863e+00, -2.670313930138945580e-03, 5.984537601470947266e-01,
        1.600000000000000000e+01,  0.000000000000000000e+00,  -2.670378191396594048e-03, -1.195905566215515137e+00,
        -4.446196835488080978e-04, 5.979884266853332520e-01,  1.700000000000000000e+01,  0.000000000000000000e+00,
        -3.114997874945402145e-03, -5.979171395301818848e-01, 3.114375285804271698e-03,  5.979187488555908203e-01,
        1.800000000000000000e+01,  0.000000000000000000e+00,  7.361537456512451172e+00,  -7.120077514648437500e+01,
        2.050399780273437500e-03,  5.243911743164062500e-01,  4.000000000000000000e+00,  1.000000000000000000e+00,
        7.363587856292724609e+00,  -7.067638397216796875e+01, 6.593227386474609375e-02,  5.465545654296875000e-01,
        5.000000000000000000e+00,  1.000000000000000000e+00,  7.429520130157470703e+00,  -7.012982940673828125e+01,
        2.786636352539062500e-02,  5.838241577148437500e-01,  6.000000000000000000e+00,  1.000000000000000000e+00,
        7.457386493682861328e+00,  -6.954600524902343750e+01, 3.253793716430664062e-02,  5.011520385742187500e-01,
        7.000000000000000000e+00,  1.000000000000000000e+00,  7.489924430847167969e+00,  -6.904485321044921875e+01,
        -1.276969909667968750e-03, 5.055618286132812500e-01,  8.000000000000000000e+00,  1.000000000000000000e+00,
        7.488647460937500000e+00,  -6.853929138183593750e+01, -8.541107177734375000e-03, 5.379028320312500000e-01,
        9.000000000000000000e+00,  1.000000000000000000e+00,  7.480106353759765625e+00,  -6.800138854980468750e+01,
        2.901124954223632812e-02,  5.734024047851562500e-01,  1.000000000000000000e+01,  1.000000000000000000e+00,
        7.509117603302001953e+00,  -6.742798614501953125e+01, 1.401281356811523438e-02,  5.776824951171875000e-01,
        1.100000000000000000e+01,  1.000000000000000000e+00,  7.523130416870117188e+00,  -6.685030364990234375e+01,
        1.982831954956054688e-02,  4.652938842773437500e-01,  1.200000000000000000e+01,  1.000000000000000000e+00,
        7.542958736419677734e+00,  -6.638500976562500000e+01, 4.219770431518554688e-02,  5.215072631835937500e-01,
        1.300000000000000000e+01,  1.000000000000000000e+00,  7.585156440734863281e+00,  -6.586350250244140625e+01,
        -1.670026779174804688e-02, 5.227890014648437500e-01,  1.400000000000000000e+01,  1.000000000000000000e+00,
        7.568456172943115234e+00,  -6.534071350097656250e+01, 1.123476028442382812e-02,  5.368270874023437500e-01,
        1.500000000000000000e+01,  1.000000000000000000e+00,  7.579690933227539062e+00,  -6.480388641357421875e+01,
        -1.076698303222656250e-03, 5.442047119140625000e-01,  1.600000000000000000e+01,  1.000000000000000000e+00,
        7.578614234924316406e+00,  -6.425968170166015625e+01, -9.348392486572265625e-03, 5.420494079589843750e-01,
        1.700000000000000000e+01,  1.000000000000000000e+00,  7.569265842437744141e+00,  -6.371763229370117188e+01,
        1.344537734985351562e-02,  5.592384338378906250e-01,  1.800000000000000000e+01,  1.000000000000000000e+00,
        -3.837791919708251953e+00, 1.237205600738525391e+01,  -1.185034438967704773e-01, 1.704231739044189453e+00,
        0.000000000000000000e+00,  2.000000000000000000e+00,  -3.886562585830688477e+00, 1.421294975280761719e+01,
        2.096232213079929352e-02,  1.977554917335510254e+00,  0.000000000000000000e+00,  2.000000000000000000e+00,
        -3.950538873672485352e+00, 1.619766807556152344e+01,  -1.489150673151016235e-01, 1.991883039474487305e+00,
        0.000000000000000000e+00,  2.000000000000000000e+00,  -4.047121524810791016e+00, 1.819177055358886719e+01,
        -4.424993693828582764e-02, 1.996319532394409180e+00,  0.000000000000000000e+00,  2.000000000000000000e+00,
        -4.106105327606201172e+00, 2.018779945373535156e+01,  -7.371773570775985718e-02, 1.995741724967956543e+00,
        0.000000000000000000e+00,  2.000000000000000000e+00,  -4.133986473083496094e+00, 2.217730140686035156e+01,
        1.795570738613605499e-02,  1.983259558677673340e+00,  0.000000000000000000e+00,  2.000000000000000000e+00,
        -4.127092838287353516e+00, 2.416782188415527344e+01,  -4.169063176959753036e-03, 1.997781872749328613e+00,
        0.000000000000000000e+00,  2.000000000000000000e+00,  -4.091072559356689453e+00, 2.616449737548828125e+01,
        7.620998471975326538e-02,  1.995570778846740723e+00,  0.000000000000000000e+00,  2.000000000000000000e+00,
        -4.008279800415039062e+00, 2.816065406799316406e+01,  8.937565982341766357e-02,  1.996743083000183105e+00,
        0.000000000000000000e+00,  2.000000000000000000e+00,  -3.930078029632568359e+00, 3.015811157226562500e+01,
        6.702778488397598267e-02,  1.998169422149658203e+00,  0.000000000000000000e+00,  2.000000000000000000e+00,
        -3.837462186813354492e+00, 3.215513229370117188e+01,  1.182038411498069763e-01,  1.995875120162963867e+00,
        0.000000000000000000e+00,  2.000000000000000000e+00,  -3.718083143234252930e+00, 3.415118026733398438e+01,
        1.205540224909782410e-01,  1.996220946311950684e+00,  0.000000000000000000e+00,  2.000000000000000000e+00,
        -3.608697414398193359e+00, 3.614749145507812500e+01,  9.821779280900955200e-02,  1.996397972106933594e+00,
        0.000000000000000000e+00,  2.000000000000000000e+00,  -3.511837482452392578e+00, 3.814450836181640625e+01,
        9.550180286169052124e-02,  1.997635960578918457e+00,  0.000000000000000000e+00,  2.000000000000000000e+00,
        -3.406197547912597656e+00, 4.014163589477539062e+01,  1.157784014940261841e-01,  1.996623158454895020e+00,
        0.000000000000000000e+00,  2.000000000000000000e+00,  -3.286454677581787109e+00, 4.213786697387695312e+01,
        1.237070485949516296e-01,  1.995835661888122559e+00,  0.000000000000000000e+00,  2.000000000000000000e+00,
        -3.152158498764038086e+00, 4.413314056396484375e+01,  1.448853164911270142e-01,  1.994714379310607910e+00,
        0.000000000000000000e+00,  2.000000000000000000e+00,  1.935915946960449219e+00,  1.222680473327636719e+01,
        1.346894949674606323e-01,  1.894992113113403320e+00,  0.000000000000000000e+00,  3.000000000000000000e+00,
        2.046184778213500977e+00,  1.417021846771240234e+01,  8.584836870431900024e-02,  1.991836071014404297e+00,
        0.000000000000000000e+00,  3.000000000000000000e+00,  2.162996768951416016e+00,  1.616250991821289062e+01,
        1.477753818035125732e-01,  1.992747306823730469e+00,  0.000000000000000000e+00,  3.000000000000000000e+00,
        2.304739713668823242e+00,  1.815561294555664062e+01,  1.357106119394302368e-01,  1.993457674980163574e+00,
        0.000000000000000000e+00,  3.000000000000000000e+00,  2.450719356536865234e+00,  2.014867782592773438e+01,
        1.562487632036209106e-01,  1.992670893669128418e+00,  0.000000000000000000e+00,  3.000000000000000000e+00,
        2.592387676239013672e+00,  2.214142990112304688e+01,  1.270879060029983521e-01,  1.992835640907287598e+00,
        0.000000000000000000e+00,  3.000000000000000000e+00,  2.730113983154296875e+00,  2.413462066650390625e+01,
        1.483646035194396973e-01,  1.993544936180114746e+00,  0.000000000000000000e+00,  3.000000000000000000e+00,
        2.876693010330200195e+00,  2.612789154052734375e+01,  1.447931975126266479e-01,  1.992998838424682617e+00,
        0.000000000000000000e+00,  3.000000000000000000e+00,  3.018285512924194336e+00,  2.812138557434082031e+01,
        1.383920907974243164e-01,  1.993989109992980957e+00,  0.000000000000000000e+00,  3.000000000000000000e+00,
        3.159373044967651367e+00,  3.011531257629394531e+01,  1.437827497720718384e-01,  1.993865013122558594e+00,
        0.000000000000000000e+00,  3.000000000000000000e+00,  3.310741901397705078e+00,  3.210858154296875000e+01,
        1.589552611112594604e-01,  1.992672204971313477e+00,  0.000000000000000000e+00,  3.000000000000000000e+00,
        3.459377050399780273e+00,  3.410206985473632812e+01,  1.383149027824401855e-01,  1.994303822517395020e+00,
        0.000000000000000000e+00,  3.000000000000000000e+00,  3.612912893295288086e+00,  3.609537506103515625e+01,
        1.687566190958023071e-01,  1.992310047149658203e+00,  0.000000000000000000e+00,  3.000000000000000000e+00,
        3.777738809585571289e+00,  3.808800888061523438e+01,  1.608956158161163330e-01,  1.992955088615417480e+00,
        0.000000000000000000e+00,  3.000000000000000000e+00,  3.938525915145874023e+00,  4.008096694946289062e+01,
        1.606784462928771973e-01,  1.992960572242736816e+00,  0.000000000000000000e+00,  3.000000000000000000e+00,
        4.095366001129150391e+00,  4.207425689697265625e+01,  1.530021727085113525e-01,  1.993621587753295898e+00,
        0.000000000000000000e+00,  3.000000000000000000e+00,  4.260129451751708984e+00,  4.406697845458984375e+01,
        1.765242815017700195e-01,  1.991816401481628418e+00,  0.000000000000000000e+00,  3.000000000000000000e+00,
        6.517323017120361328e+00,  1.423272228240966797e+01,  1.056644693017005920e-01,  1.952507972717285156e+00,
        0.000000000000000000e+00,  4.000000000000000000e+00,  6.670034885406494141e+00,  1.620129776000976562e+01,
        1.997596770524978638e-01,  1.984644532203674316e+00,  0.000000000000000000e+00,  4.000000000000000000e+00,
        6.832424163818359375e+00,  1.818934822082519531e+01,  1.250188946723937988e-01,  1.991455912590026855e+00,
        0.000000000000000000e+00,  4.000000000000000000e+00,  7.030147552490234375e+00,  2.017398452758789062e+01,
        2.704281508922576904e-01,  1.977817058563232422e+00,  0.000000000000000000e+00,  4.000000000000000000e+00,
        7.164125442504882812e+00,  2.215261077880859375e+01,  -2.472935942932963371e-03, 1.979436159133911133e+00,
        0.000000000000000000e+00,  4.000000000000000000e+00,  7.256398200988769531e+00,  2.413741302490234375e+01,
        1.870183944702148438e-01,  1.990166544914245605e+00,  0.000000000000000000e+00,  4.000000000000000000e+00,
        7.364072799682617188e+00,  2.613063621520996094e+01,  2.833077684044837952e-02,  1.996281504631042480e+00,
        0.000000000000000000e+00,  4.000000000000000000e+00,  7.443918704986572266e+00,  2.812641716003417969e+01,
        1.313607990741729736e-01,  1.995278120040893555e+00,  0.000000000000000000e+00,  4.000000000000000000e+00,
        7.572989463806152344e+00,  3.012179756164550781e+01,  1.267810910940170288e-01,  1.995484471321105957e+00,
        0.000000000000000000e+00,  4.000000000000000000e+00,  7.701957225799560547e+00,  3.211704635620117188e+01,
        1.311546713113784790e-01,  1.995011210441589355e+00,  0.000000000000000000e+00,  4.000000000000000000e+00,
        7.842288970947265625e+00,  3.411135482788085938e+01,  1.495084315538406372e-01,  1.993606805801391602e+00,
        0.000000000000000000e+00,  4.000000000000000000e+00,  8.016213417053222656e+00,  3.610265350341796875e+01,
        1.983410269021987915e-01,  1.988987445831298828e+00,  0.000000000000000000e+00,  4.000000000000000000e+00,
        8.218122482299804688e+00,  3.809153747558593750e+01,  2.054767310619354248e-01,  1.988786816596984863e+00,
        0.000000000000000000e+00,  4.000000000000000000e+00,  8.415946960449218750e+00,  4.008115386962890625e+01,
        1.901725828647613525e-01,  1.990441799163818359e+00,  0.000000000000000000e+00,  4.000000000000000000e+00,
        8.628990173339843750e+00,  4.206940460205078125e+01,  2.359134256839752197e-01,  1.986064195632934570e+00,
        0.000000000000000000e+00,  4.000000000000000000e+00,  8.849534988403320312e+00,  4.405715179443359375e+01,
        2.051756829023361206e-01,  1.989425897598266602e+00,  0.000000000000000000e+00,  4.000000000000000000e+00,
        -6.685617446899414062e+00, 1.746151733398437500e+01,  -8.465433865785598755e-02, 1.993393778800964355e+00,
        0.000000000000000000e+00,  5.000000000000000000e+00,  -6.771605014801025391e+00, 1.945569801330566406e+01,
        -8.732049167156219482e-02, 1.994968056678771973e+00,  0.000000000000000000e+00,  5.000000000000000000e+00,
        -6.949454307556152344e+00, 2.144436836242675781e+01,  -2.683780491352081299e-01, 1.982372999191284180e+00,
        0.000000000000000000e+00,  5.000000000000000000e+00,  -6.978304862976074219e+00, 2.341926956176757812e+01,
        2.106766551733016968e-01,  1.967429041862487793e+00,  0.000000000000000000e+00,  5.000000000000000000e+00,
        -6.884036064147949219e+00, 2.540054512023925781e+01,  -2.213949337601661682e-02, 1.995122432708740234e+00,
        0.000000000000000000e+00,  5.000000000000000000e+00,  -6.811784744262695312e+00, 2.739352798461914062e+01,
        1.666428595781326294e-01,  1.990842938423156738e+00,  0.000000000000000000e+00,  5.000000000000000000e+00,
        -6.704393386840820312e+00, 2.938816642761230469e+01,  4.813972115516662598e-02,  1.998434185981750488e+00,
        0.000000000000000000e+00,  5.000000000000000000e+00,  -6.617659091949462891e+00, 3.138487243652343750e+01,
        1.253286451101303101e-01,  1.994978189468383789e+00,  0.000000000000000000e+00,  5.000000000000000000e+00,
        -6.493432044982910156e+00, 3.337998580932617188e+01,  1.231250762939453125e-01,  1.995249986648559570e+00,
        0.000000000000000000e+00,  5.000000000000000000e+00,  -6.361684799194335938e+00, 3.537453460693359375e+01,
        1.403697282075881958e-01,  1.993846774101257324e+00,  0.000000000000000000e+00,  5.000000000000000000e+00,
        -6.268584251403808594e+00, 3.737096405029296875e+01,  4.583103582262992859e-02,  1.999010205268859863e+00,
        0.000000000000000000e+00,  5.000000000000000000e+00,  -6.200242996215820312e+00, 3.936912918090820312e+01,
        9.085240960121154785e-02,  1.997324705123901367e+00,  0.000000000000000000e+00,  5.000000000000000000e+00,
        -6.107946872711181641e+00, 4.136639022827148438e+01,  9.373934566974639893e-02,  1.997190594673156738e+00,
        0.000000000000000000e+00,  5.000000000000000000e+00,  -6.005339622497558594e+00, 4.336305236816406250e+01,
        1.114751696586608887e-01,  1.996139883995056152e+00,  0.000000000000000000e+00,  5.000000000000000000e+00,
        -9.426095008850097656e+00, 2.386907005310058594e+01,  1.060331091284751892e-01,  1.974369525909423828e+00,
        0.000000000000000000e+00,  6.000000000000000000e+00,  -9.375787734985351562e+00, 2.585077095031738281e+01,
        -5.418063607066869736e-03, 1.989032149314880371e+00,  0.000000000000000000e+00,  6.000000000000000000e+00,
        -9.312583923339843750e+00, 2.784189224243164062e+01,  1.318266540765762329e-01,  1.993209719657897949e+00,
        0.000000000000000000e+00,  6.000000000000000000e+00,  -9.243625640869140625e+00, 2.983792686462402344e+01,
        6.089696194976568222e-03,  1.998858690261840820e+00,  0.000000000000000000e+00,  6.000000000000000000e+00,
        -9.169966697692871094e+00, 3.183437156677246094e+01,  1.412270665168762207e-01,  1.994030117988586426e+00,
        0.000000000000000000e+00,  6.000000000000000000e+00,  -9.049769401550292969e+00, 3.382954025268554688e+01,
        9.916866570711135864e-02,  1.996309399604797363e+00,  0.000000000000000000e+00,  6.000000000000000000e+00,
        -8.935756683349609375e+00, 3.582468032836914062e+01,  1.288564652204513550e-01,  1.993972301483154297e+00,
        0.000000000000000000e+00,  6.000000000000000000e+00,  -8.867085456848144531e+00, 3.782122802734375000e+01,
        8.485795930027961731e-03,  1.999121665954589844e+00,  0.000000000000000000e+00,  6.000000000000000000e+00,
        -8.815942764282226562e+00, 3.981941604614257812e+01,  9.379958361387252808e-02,  1.997253179550170898e+00,
        0.000000000000000000e+00,  6.000000000000000000e+00,  -8.736066818237304688e+00, 4.181710052490234375e+01,
        6.595162302255630493e-02,  1.998115301132202148e+00,  0.000000000000000000e+00,  6.000000000000000000e+00,
        -8.655145645141601562e+00, 4.381479263305664062e+01,  9.589059650897979736e-02,  1.997271180152893066e+00,
        0.000000000000000000e+00,  6.000000000000000000e+00};
    float cluster[input_data.feats_num] = {
        0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00,
        0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00,
        0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00,
        0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00,
        1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
        1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00,
        1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 2.000000000000000000e+00,
        2.000000000000000000e+00, 2.000000000000000000e+00, 2.000000000000000000e+00, 2.000000000000000000e+00, 2.000000000000000000e+00,
        2.000000000000000000e+00, 2.000000000000000000e+00, 2.000000000000000000e+00, 2.000000000000000000e+00, 2.000000000000000000e+00,
        2.000000000000000000e+00, 2.000000000000000000e+00, 2.000000000000000000e+00, 2.000000000000000000e+00, 2.000000000000000000e+00,
        2.000000000000000000e+00, 3.000000000000000000e+00, 3.000000000000000000e+00, 3.000000000000000000e+00, 3.000000000000000000e+00,
        3.000000000000000000e+00, 3.000000000000000000e+00, 3.000000000000000000e+00, 3.000000000000000000e+00, 3.000000000000000000e+00,
        3.000000000000000000e+00, 3.000000000000000000e+00, 3.000000000000000000e+00, 3.000000000000000000e+00, 3.000000000000000000e+00,
        3.000000000000000000e+00, 3.000000000000000000e+00, 3.000000000000000000e+00, 4.000000000000000000e+00, 4.000000000000000000e+00,
        4.000000000000000000e+00, 4.000000000000000000e+00, 4.000000000000000000e+00, 4.000000000000000000e+00, 4.000000000000000000e+00,
        4.000000000000000000e+00, 4.000000000000000000e+00, 4.000000000000000000e+00, 4.000000000000000000e+00, 4.000000000000000000e+00,
        4.000000000000000000e+00, 4.000000000000000000e+00, 4.000000000000000000e+00, 4.000000000000000000e+00, 5.000000000000000000e+00,
        5.000000000000000000e+00, 5.000000000000000000e+00, 5.000000000000000000e+00, 5.000000000000000000e+00, 5.000000000000000000e+00,
        5.000000000000000000e+00, 5.000000000000000000e+00, 5.000000000000000000e+00, 5.000000000000000000e+00, 5.000000000000000000e+00,
        5.000000000000000000e+00, 5.000000000000000000e+00, 5.000000000000000000e+00, 6.000000000000000000e+00, 6.000000000000000000e+00,
        6.000000000000000000e+00, 6.000000000000000000e+00, 6.000000000000000000e+00, 6.000000000000000000e+00, 6.000000000000000000e+00,
        6.000000000000000000e+00, 6.000000000000000000e+00, 6.000000000000000000e+00, 6.000000000000000000e+00};
    float cluster_count[input_data.cluster_num] = {
        1.900000000000000000e+01,
        1.500000000000000000e+01,
        1.700000000000000000e+01,
        1.700000000000000000e+01,
        1.600000000000000000e+01,
        1.400000000000000000e+01,
        1.100000000000000000e+01};
    float id_embedding[input_data.cluster_num * 2] = {
        -3.114997874945402145e-03,
        -1.142864322662353516e+01,
        7.361537456512451172e+00,
        -7.120077514648437500e+01,
        -4.133986473083496094e+00,
        1.237205600738525391e+01,
        1.935915946960449219e+00,
        1.222680473327636719e+01,
        6.517323017120361328e+00,
        1.423272228240966797e+01,
        -6.978304862976074219e+00,
        1.746151733398437500e+01,
        -9.426095008850097656e+00,
        2.386907005310058594e+01};

    // float feature[INPUT_CHANNEL * input_data.feats_num] = {
    //     -0.3801, -0.1300, 1.1666,  -1.1327, 0.6438,  0.6729,  -1.1299, -2.2857, 0.1849,  0.0493,  -0.4179, -0.5331,
    //     0.7467,  -1.0006, 1.4848,  0.2771,  0.1393,  -0.9162, -1.7744, 0.8850,  -1.6748, 1.3581,  -0.4987, -0.7244,
    //     0.7941,  -0.4109, -0.3446, -0.5246, -0.8153, -0.5685, 1.9105,  -0.1069, 0.7214,  0.5255,  0.3654,  -0.3434,
    //     0.7163,  -0.6460, 1.9680,  0.8964,  0.3845,  3.4347,  -2.6291, -0.9330, 0.6411,  0.9983,  0.6731,  0.9110,
    //     -2.0634, -0.5751, 1.4070,  0.5285,  -0.1171, -0.1863, 2.1200,  1.3745,  0.9763,  -0.1193, -0.3343, -1.5933};
    // float cluster[input_data.feats_num]            = {0, 1, 1, 2, 2, 3, 3, 3, 3, 4};
    // float cluster_count[input_data.cluster_num]    = {1, 2, 2, 4, 1};
    // float id_embedding[input_data.cluster_num * 2] = {-0.3330, -0.7534, 1.1834, 0.6447, -1.1398, 0.5933, 1.5586, 1.0459, 0.2039, 1.0544};

    input_data.feature       = feature;
    input_data.cluster       = cluster;
    input_data.cluster_count = cluster_count;
    input_data.id_embedding  = id_embedding;

    VectorNet vectornet;
    vectornet.Init(options);

    // For precision compare.
    TrajPredictData pred_data;
    vectornet.Process(input_data, pred_data);
    for (int i = 0; i < pred_data.predict_points.size(); i++)
    {
        if (i % 2 != 0)
            printf("%f\n", pred_data.predict_points[i]);
        else
            printf("%f,", pred_data.predict_points[i]);
    }

    // For timing, because first time is slow, doesnot count.
    auto start = std::chrono::system_clock::now();
    for (int k = 0; k < 1000; k++)
    {
        TrajPredictData pred_data2;
        vectornet.Process(input_data, pred_data2);
    }
    auto end = std::chrono::system_clock::now();
    std::cout << "\n[INFO]: Time taken by execution: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
              << std::endl;

    return 0;
}

/*
This code output            Pytorch output
-0.000751,0.598084        [-0.0008,  0.5981],
0.002925,0.618462         [ 0.0029,  0.6184],
0.002822,0.607659         [ 0.0028,  0.6076],
0.003941,0.613367         [ 0.0039,  0.6133],
0.004445,0.619251         [ 0.0044,  0.6192],
0.005261,0.606338         [ 0.0053,  0.6063],
0.004013,0.606201         [ 0.0040,  0.6062],
0.001496,0.615865         [ 0.0015,  0.6158],
0.002494,0.603205         [ 0.0025,  0.6032],
0.001168,0.611155         [ 0.0012,  0.6111],
0.002630,0.602360         [ 0.0026,  0.6023],
0.002455,0.606869         [ 0.0025,  0.6068],
0.003524,0.609820         [ 0.0035,  0.6098],
0.003280,0.604363         [ 0.0033,  0.6043],
0.002711,0.606441         [ 0.0027,  0.6064],
0.003065,0.599403         [ 0.0031,  0.5994],
0.004386,0.600533         [ 0.0044,  0.6005],
0.004443,0.609845         [ 0.0044,  0.6098],
0.002879,0.600789         [ 0.0029,  0.6008],
0.005783,0.596618         [ 0.0058,  0.5966],
0.005589,0.601561         [ 0.0056,  0.6015],
0.005739,0.593500         [ 0.0057,  0.5935],
0.006239,0.596110         [ 0.0062,  0.5961],
0.005204,0.605715         [ 0.0052,  0.6057],
0.006330,0.603428         [ 0.0063,  0.6034],
0.004776,0.596250         [ 0.0048,  0.5962],
0.007388,0.584935         [ 0.0074,  0.5849],
0.006047,0.601648         [ 0.0060,  0.6016],
0.007334,0.602399         [ 0.0073,  0.6024],
0.008742,0.594365         [ 0.0087,  0.5943]

*/