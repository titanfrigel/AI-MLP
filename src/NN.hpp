/*
** File: NN.hpp
** Project: Ai2
** Created Date: Tuesday, June 20th 2023, 5:18:59 am
** Author: titanfrigel
** -----
** Last Modified: Thursday, June 22nd 2023, 6:29:02 pm
** Modified By: titanfrigel
*/

#ifndef NN_HPP
    #define NN_HPP

#include "Matrix.hpp"
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

template <typename T>
class NN

{
    private:
        std::vector<Matrix<T>> m_weights;
        std::vector<Matrix<T>> m_biases;
        std::vector<Matrix<T>> m_layers;
        double m_learning_rate;
        double m_average_error;

    public:
        NN(size_t input, size_t hidden, size_t output, size_t hidden_layers) : m_weights(), m_biases(), m_layers(), m_learning_rate(0.01)
        {
            srand(time(NULL));

            this->m_weights.push_back(Matrix<T>(hidden, output));
            this->m_weights.back().randomize();
            this->m_biases.push_back(Matrix<T>(hidden, 1));
            this->m_biases.back().randomize();
            this->m_layers.push_back(Matrix<T>(input, 1));
            this->m_layers.back().randomize();

            for (size_t i = 1; i <= hidden_layers; ++i) {
                this->m_weights.push_back(Matrix<T>(hidden, hidden));
                this->m_weights.back().randomize();
                this->m_biases.push_back(Matrix<T>(hidden, 1));
                this->m_biases.back().randomize();
                this->m_layers.push_back(Matrix<T>(hidden, 1));
                this->m_layers.back().randomize();

            }

            this->m_weights.push_back(Matrix<T>(output, hidden));
            this->m_weights.back().randomize();
            this->m_biases.push_back(Matrix<T>(output, 1));
            this->m_biases.back().randomize();
            this->m_layers.push_back(Matrix<T>(hidden, 1));
            this->m_layers.back().randomize();
            this->m_layers.push_back(Matrix<T>(output, 1));
            this->m_layers.back().randomize();
        }
        ~NN(void) = default;

        void set_learning_rate(double learning_rate)
        {
            this->m_learning_rate = learning_rate;
        }

        void feed_forward(std::vector<T> inputs)
        {
            assert(inputs.size() == this->m_layers[0].get_rows());

            for (size_t i = 0; i < inputs.size(); ++i)
                this->m_layers[0](i, 0) = inputs[i];

            for (size_t i = 1; i < this->m_layers.size(); ++i) {
                this->m_layers[i] = this->m_weights[i - 1] * this->m_layers[i - 1];
                this->m_layers[i] += this->m_biases[i - 1];
                this->m_layers[i].apply_func([](T x) { return (1 / (1 + exp(-x))); });
            }
        }

        void back_propagation(std::vector<T> targets)
        {
            assert(targets.size() == this->m_layers[this->m_layers.size() - 1].get_rows());

            std::vector<Matrix<T>> errors(this->m_layers.size());
            std::vector<Matrix<T>> gradients(this->m_layers.size());

            errors[this->m_layers.size() - 1] = Matrix<T>(targets.size(), 1);
            for (size_t i = 0; i < targets.size(); ++i) {
                errors[this->m_layers.size() - 1](i, 0) = targets[i] - this->m_layers[this->m_layers.size() - 1](i, 0);
                m_average_error += errors[this->m_layers.size() - 1](i, 0) * errors[this->m_layers.size() - 1](i, 0);
            }
            m_average_error /= targets.size() + 1;

            for (size_t i = this->m_layers.size() - 1; i > 0; --i) {
                gradients[i] = this->m_layers[i];
                gradients[i].apply_func([](T x) { return (x * (1 - x)); });
                gradients[i] = gradients[i].hadamard(errors[i]);
                gradients[i] *= this->m_learning_rate;

                Matrix<T> delta = gradients[i] * this->m_layers[i - 1].transpose();
                this->m_weights[i - 1] += delta;
                this->m_biases[i - 1] += gradients[i];

                errors[i - 1] = this->m_weights[i - 1].transpose() * errors[i];
            }
        }

        void train(std::vector<T> inputs, std::vector<T> targets, bool print = false)
        {
            this->feed_forward(inputs);
            if (print) {
                printf("Got: \n");
                m_layers[m_layers.size() - 1].print();
                printf("Expected: \n");
                for (size_t i = 0; i < targets.size(); ++i)
                    printf("%.4f\n", targets[i]);
            }
            this->back_propagation(targets);
        }

        void print_output(void)
        {
            this->m_layers[this->m_layers.  size() - 1].print();
        }

        double get_average_error(void)
        {
            return (this->m_average_error);
        }

};


#endif // !NN_HPP