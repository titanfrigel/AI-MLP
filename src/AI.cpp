/*
** File: AI.cpp
** Project: Ai2
** Created Date: Tuesday, June 20th 2023, 6:26:35 am
** Author: titanfrigel
** -----
** Last Modified: Tuesday, June 20th 2023, 8:26:40 am
** Modified By: titanfrigel
*/

#include "AI.hpp"

void ai::AI::feed_forward(const std::vector<double>& input)
{
    for (int i = 0; i < input.size(); i++) {
        this->m_layers[0][i] = input[i];
    }

    for (int i = 1; i < this->m_layers.size(); i++) {
        for (int j = 0; j < this->m_layers[i].size(); j++) {
            this->m_layers[i][j] = 0;

            for (int k = 0; k < this->m_layers[i - 1].size(); k++) {
                this->m_layers[i][j] += this->m_layers[i - 1][k] * this->m_weights[i - 1][j][k];
            }

            this->m_layers[i][j] += this->m_biases[i - 1][j];
            this->m_layers[i][j] = m_activationFunction(this->m_layers[i][j]);
        }
    }
}

void ai::AI::back_propagation(const std::vector<double>& target)
{
    std::vector<std::vector<double>> output = this->m_layers;
    std::vector<std::vector<std::vector<double>>> weights = this->m_weights;
    std::vector<std::vector<double>> biases = this->m_biases;

}

static int get_max(std::vector<double> output)
{
    int max = 0;

    for (int i = 0; i < output.size(); i++) {
        if (output[i] > output[max]) {
            max = i;
        }
    }

    return max;
}

void ai::AI::train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& target, int epochs)
{
    double error = 0;
    int total_iter = 1;

    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < input.size(); j++, total_iter++) {
            this->feed_forward(input[j]);
            if (get_max(this->m_layers.back()) != get_max(target[j])) {
                error++;
            }
            if (i % 10 == 0 && j == 0) {
                printf("Epoch: %d, Error: %f\n", i, error / total_iter * 100);
                printf("Output: ");
                for (int k = 0; k < this->m_layers.back().size(); k++) {
                    printf("%.4f\t", this->m_layers.back()[k]);
                }
                printf("\nTarget: ");
                for (int k = 0; k < target[j].size(); k++) {
                    printf("%.4f\t", target[j][k]);
                }
                printf("\n");
            }
            this->back_propagation(target[j]);
        }
    }
}
