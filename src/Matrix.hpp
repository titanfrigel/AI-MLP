/*
** File: Matrix.hpp
** Project: AI-MLP
** Created Date: Tuesday, June 20th 2023, 8:28:23 am
** Author: titanfrigel
** -----
** Last Modified: Thursday, June 22nd 2023, 6:45:52 pm
** Modified By: titanfrigel
*/

#ifndef MATRIX_HPP
    #define MATRIX_HPP

    #include <cstddef>
    #include <vector>
    #include <functional>
    #include <iostream>
    #include <cassert>

template <typename T>
class Matrix
{
    private:
        std::vector<T> m_matrix;
        size_t m_rows;
        size_t m_cols;

    public:
        Matrix() : m_matrix(0), m_rows(0), m_cols(0) {}

        Matrix(size_t rows, size_t cols) : m_matrix(rows * cols), m_rows(rows), m_cols(cols)
        {
            assert(this->m_cols > 0 && this->m_rows > 0);
        }

        Matrix(const Matrix<T> &target) = default;

        ~Matrix() = default;

        Matrix<T> &operator=(const Matrix<T> &) = default;


        T &operator()(unsigned int row, unsigned int col)
        {
            return m_matrix[row * m_cols + col];
        }

        T operator()(unsigned int row, unsigned int col) const
        {
            return m_matrix[row * m_cols + col];
        }

        Matrix<T> &operator+=(Matrix<T> const &target) // ! Operator +=
        {
            assert(this->m_rows == target.m_rows && this->m_cols == target.m_cols);

            for (size_t row = 0; row < this->m_rows; row++)
                for (size_t col = 0; col < this->m_cols; col++)
                    (*this)(row, col) += target(row, col);
            return (*this);
        }

        Matrix<T> operator+(Matrix<T> const &target) const // Operator +
        {
            return Matrix<T>(*this) += target;
        }

        Matrix<T> &operator-=(Matrix<T> const &target) // ! Operator -=
        {
            assert(this->m_rows == target.m_rows && this->m_cols == target.m_cols);

            for (size_t row = 0; row < this->m_rows; row++)
                for (size_t col = 0; col < this->m_cols; col++)
                    (*this)(row, col) -= target(row, col);
            return (*this);
        }

        Matrix<T> operator-(Matrix<T> const &target) const // Operator -
        {
            return Matrix<T>(*this) -= target;
        }

        bool operator==(Matrix<T> const &target) const // ! Operator ==
        {
            for (size_t row = 0; row < this->m_rows; row++)
                for (size_t col = 0; col < this->m_cols; col++)
                    if ((*this)(row, col) != target(row, col))
                        return (false);
            return (true);
        }

        bool operator!=(Matrix<T> const &target) const // Operator !=
        {
            return !(*this == target);
        }

        Matrix<T> &operator*=(T target) // ! Operator *=
        {
            for (size_t row = 0; row < this->m_rows; row++)
                for (size_t col = 0; col < this->m_cols; col++)
                    (*this)(row, col) *= target;
            return (*this);
        }

        Matrix<T> operator*(T target) const // Operator *
        {
            return (Matrix<T>(*this) *= target);
        }

        Matrix<T> &operator*=(Matrix<T> const &target) // ! Operator *=
        {
            return (*this = *this * target);
        }

        Matrix<T> operator*(Matrix<T> const &target) const // Operator *
        {
            assert(this->m_cols == target.m_rows);

            Matrix<T> matrix(this->m_rows, target.m_cols);

            for (size_t row = 0; row < this->m_rows; row++)
                for (size_t col = 0; col < target.m_cols; col++)
                    for (size_t i = 0; i < this->m_cols; i++)
                        matrix(row, col) += (*this)(row, i) * target(i, col);
            return (matrix);
        }

        Matrix<T> &operator/=(T target) // ! Operator /=
        {
            for (size_t row = 0; row < this->m_rows; row++)
                for (size_t col = 0; col < this->m_cols; col++)
                    (*this)(row, col) /= target;
            return (*this);
        }

        Matrix<T> operator/(T target) const // Operator /
        {
            return (Matrix<T>(*this) /= target);
        }

        Matrix<T> transpose() const // ! Transpose
        {
            Matrix<T> matrix(this->m_cols, this->m_rows);

            for (size_t row = 0; row < this->m_rows; row++)
                for (size_t col = 0; col < this->m_cols; col++)
                    matrix(col, row) = (*this)(row, col);
            return (matrix);
        }

        Matrix<T> hadamard(Matrix<T> const &target)
        {
            assert(this->m_rows == target.m_rows && this->m_cols == target.m_cols);

            Matrix<T> matrix(this->m_rows, this->m_cols);

            for (size_t row = 0; row < this->m_rows; row++)
                for (size_t col = 0; col < this->m_cols; col++)
                    matrix(row, col) = (*this)(row, col) * target(row, col);
            return (matrix);
        }

        static Matrix<T> identity(size_t rows, size_t cols) // ! Identity
        {
            assert(rows == cols);

            Matrix<T> matrix(rows, cols);

            for (unsigned int i = 0; i < cols; i++)
                matrix(i, i) = 1.f;
            return (matrix);
        }

        Matrix<T> apply_func(std::function<T(T)> f)
        {
            for (size_t row = 0; row < this->m_rows; row++)
                for (size_t col = 0; col < this->m_cols; col++)
                    (*this)(row, col) = f((*this)(row, col));
            return (*this);
        }

        Matrix<T> fill(T x) const
        {
            for (size_t row = 0; row < this->m_rows; row++)
                for (size_t col = 0; col < this->m_cols; col++)
                    (*this)(row, col) = x;
            return (*this);
        }

        size_t get_cols() const
        {
            return (this->m_cols);
        }

        size_t get_rows() const
        {
            return (this->m_rows);
        }

        void print() const
        {
            for (size_t row = 0; row < this->m_rows; row++)
            {
                for (size_t col = 0; col < this->m_cols; col++)
                    printf("%.4f\t", (*this)(row, col));
                printf("\n");
            }
        }

        void randomize()
        {
            for (size_t row = 0; row < this->m_rows; row++)
                for (size_t col = 0; col < this->m_cols; col++)
                    (*this)(row, col) = (T)rand() / (T)RAND_MAX;
        }
};


#endif // ! MATRIX_HPP