from fractions import Fraction
from itertools import chain, takewhile


class Matrix:
    def __init__(self, matrix_iter):
        self.rows, self.cols = len(matrix_iter), len(matrix_iter[0])
        self.matrix = [*map(lambda row: [*map(Fraction, row)], matrix_iter)]

    def __getitem__(self, item):
        return self.matrix[item]

    def __setitem__(self, key, value):
        self.matrix[key] = value

    def __iter__(self):
        return self.matrix.__iter__()

    def __str__(self):
        return str('\n'.join(str([*map(str, row)]) for row in self.matrix))

    def __add__(self, other):
        return Matrix([[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(self.matrix, other.matrix)])

    def __sub__(self, other):
        return Matrix([[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(self.matrix, other.matrix)])

    def __mul__(self, other):
        return self.matrix_mult(other) if type(other) == Matrix else self.constant_mult(other)

    def __pow__(self, n):
        return self.inverse().matrix_power(-n) if n < 0 else self.matrix_power(n)

    def __eq__(self, other):
        return self.matrix == other.matrix

    def constant_mult(self, factor):
        return Matrix([[entry * factor for entry in row] for row in self.matrix])

    def matrix_mult(self, other):
        return Matrix([[sum(a * b for a, b in zip(row, col)) for col in zip(*other.matrix)] for row in self.matrix]) if self.cols == other.rows else None

    def matrix_power(self, n):
        curr_pow, result = self, 1
        while n:
            if n & 1:
                result = curr_pow * result
            curr_pow *= curr_pow
            n >>= 1
        return result

    def ref(self):
        ref_matrix, clearing_row = Matrix(self.matrix), 0
        for col in range(self.cols):
            for row in range(clearing_row, self.rows):
                if ref_matrix[row][col]:
                    ref_matrix[clearing_row], ref_matrix[row] = ref_matrix[row], ref_matrix[clearing_row]
                    break
            else:
                continue
            ref_matrix[clearing_row] = [entry * Fraction(1, ref_matrix[clearing_row][col]) for entry in ref_matrix[clearing_row]]
            for row_to_modify in chain(range(0, clearing_row), range(clearing_row + 1, self.rows)):
                ref_matrix[row_to_modify] = [entry1 - entry2 * ref_matrix[row_to_modify][col] for entry1, entry2 in zip(ref_matrix[row_to_modify], ref_matrix[clearing_row])]
            clearing_row += 1
        return ref_matrix

    def inverse(self):
        if self.rows != self.cols:
            return None
        identity = Matrix.identity_matrix(self.rows)
        matrix = Matrix([[*row1, *row2] for row1, row2 in zip(self.matrix, identity)]).ref()
        return Matrix([row[self.cols:] for row in matrix]) if Matrix([row[:self.cols] for row in matrix]) == identity else None

    def transpose(self):
        return Matrix([*zip(*self.matrix)])

    def det(self):
        if self.rows != self.cols:
            return None
        if self.rows == 1:
            return self.matrix[0][0]
        return sum(((-(r & 1) << 1) + 1) * self.matrix[r][0] * Matrix([*map(lambda row: row[1:], self.matrix[:r]), *map(lambda row: row[1:], self.matrix[r + 1:])]).det() for r in range(self.rows))

    def row_space_basis(self):
        return Basis(self, Basis.ROW)

    def column_space_basis(self):
        return Basis(self, Basis.COLUMN)

    def null_space_basis(self):
        return Basis(self, Basis.NULL)

    @staticmethod
    def identity_matrix(n):
        return Matrix([[*([0] * i), 1, *([0] * (n - i - 1))] for i in range(n)])


class Basis:
    ROW, COLUMN, NULL = 0, 1, 2

    def __init__(self, matrix, basis_type):
        match basis_type:
            case 0: self.basis, self.to_str = Basis.row_basis(matrix), self.row_basis_str
            case 1: self.basis, self.to_str = Basis.column_basis(matrix), self.col_basis_str
            case 2: self.basis, self.to_str = Basis.null_basis(matrix), self.col_basis_str

    def __str__(self):
        return self.to_str()

    def col_basis_str(self):
        spacing = max(max(len(str(entry)) for entry in vector) for vector in zip(*self.basis))
        return '\n'.join('  '.join(f'[{" " * (spacing - len(str(entry)))}{entry}]' for entry in vector) for vector in zip(*self.basis))

    def row_basis_str(self):
        return '\n'.join(f'{Matrix([row])}' for row in self.basis)

    @staticmethod
    def column_basis(matrix):
        return [*takewhile(any, matrix.transpose().ref())]

    @staticmethod
    def row_basis(matrix):
        return [*takewhile(any, matrix.ref())]

    @staticmethod
    def null_basis(matrix):
        columns, basis, row = matrix.transpose(), [], 0
        for c, col in enumerate(zip(*matrix.ref())):
            if row >= len(col) or col[row] == 0:
                basis.append(columns[c])
            else:
                row += 1
        return basis


def main():
    a = Matrix([[-3, 4, -5, 22],
                [-3, 3, 2, -10]])
    b = Matrix([[-9, 7, -4, -2],
                [-2, 3, -4, -3],
                [9, 8, -3, -6],
                [2, -7, -2, 1]])
    c = Matrix([[1, 4, 5, 69, 1],
                [9, 6, 2, 2, 3],
                [4, 5, 6, 11, 15],
                [14, 3, 4, 41, 23],
                [34, 52, 1, 16, 7]])
    print(c)
    print(c.det())


if __name__ == '__main__':
    main()
