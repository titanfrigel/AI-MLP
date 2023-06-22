/*
** File: AI.hpp
** Project: Ai2
** Created Date: Tuesday, June 20th 2023, 6:26:48 am
** Author: titanfrigel
** -----
** Last Modified: Tuesday, June 20th 2023, 8:39:14 am
** Modified By: titanfrigel
*/

#ifndef AI_HPP
    #define AI_HPP

    #include "NN.hpp"
    #include <functional>
    #include <cmath>
    #include <cstdio>

namespace ai
{

    class AI : private nn::NN
    {
        private:
            std::function<double(double)> m_activationFunction;
            double m_learningRate;

        public:
            AI(int input, int hidden, int output, int hiddenLayers) : nn::NN(input, hidden, output, hiddenLayers),
            m_learningRate(1), m_activationFunction([](double x) { return 1 / (1 + exp(-x)); })
            { this->randomize(); }
            ~AI(void) = default;
            void set_activation_function(std::function<double(double)> activationFunction) { m_activationFunction = activationFunction; }
            void feed_forward(const std::vector<double>& input);
            void back_propagation(const std::vector<double>& target);
            std::vector<double> get_output(void) const { return this->m_layers.back(); }
            void set_learning_rate(double learningRate) { this->m_learningRate = learningRate; }
            void train(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& target, int epochs);
    };

} // namespace ai

#endif // !AI_HPP

