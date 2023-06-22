/*
** File: Main.cpp
** Project: Ai2
** Created Date: Tuesday, June 20th 2023, 5:17:55 am
** Author: titanfrigel
** -----
** Last Modified: Thursday, June 22nd 2023, 6:38:55 pm
** Modified By: titanfrigel
*/

#include "NN.hpp"

int main(void)
{
    NN<double> nn(1, 3, 1, 2);

    int iter = 10000000;

    nn.set_learning_rate(0.01);

    for (int i = 0; i <= iter; ++i) {
        double nb = (double)rand()  / (double)RAND_MAX;
        nn.train({nb}, {sin(nb) * sin(nb)}, (i > iter - 10) ? true : false);
        if (i % 1000 == 0) {
            printf("i: %d\n", i);
            printf("Average error: %f\n", nn.get_average_error());
        }
    }

    return 0;
}