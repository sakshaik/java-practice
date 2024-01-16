package com.test.matrix;

import java.util.ArrayList;

public class MatrixTest {

	public static void main(String[] args) {
		int[][] matrix = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
		int[][] matrix2 = { { 1, 2, 3, 4, 5, 6 }, { 7, 8, 9, 10, 11, 12 }, { 13, 14, 15, 16, 17, 18 } };

		MatrixTest test = new MatrixTest();

		System.out.println(MatrixTest.search(matrix, matrix.length, matrix[0].length, 5));
		System.out.println(MatrixTest.search(matrix, matrix.length, matrix[0].length, 10));

		System.out.println(test.findK(matrix, matrix.length, matrix[0].length, 5));
		System.out.println(test.findK(matrix, matrix.length, matrix[0].length, 10));

		System.out.println(MatrixTest.spirallyTraverse(matrix2, matrix2.length, matrix2[0].length));
		System.out.println(MatrixTest.boundaryTraversal(matrix2, matrix2.length, matrix2[0].length));

		MatrixTest.rotate(matrix);
		printMatrix(matrix);

	}

	static void rotate(int matrix[][]) {
		for (int i = 0; i < matrix.length; i++) {
			for (int j = i; j < matrix[i].length; j++) {
				int temp = matrix[j][i];
				matrix[j][i] = matrix[i][j];
				matrix[i][j] = temp;
			}
		}
		for (int i = 0; i < matrix[0].length; i++) {
			for (int j = 0, k = matrix[i].length - 1; j < k; j++, k--) {
				int temp = matrix[j][i];
				matrix[j][i] = matrix[k][i];
				matrix[k][i] = temp;
			}
		}
	}

	static ArrayList<Integer> spirallyTraverse(int matrix[][], int r, int c) {
		ArrayList<Integer> spiral = new ArrayList<Integer>();
		int startRow = 0, startColumn = 0;
		while (startRow < r && startColumn < c) {
			for (int i = startColumn; i < c; i++) {
				spiral.add(matrix[startRow][i]);
			}
			startRow++;
			for (int i = startRow; i < r; i++) {
				spiral.add(matrix[i][c - 1]);
			}
			c--;
			if (startRow < r) {
				for (int i = c - 1; i >= startColumn; i--) {
					spiral.add(matrix[r - 1][i]);
				}
				r--;
			}
			if (startColumn < c) {
				for (int i = r - 1; i >= startRow; i--) {
					spiral.add(matrix[i][startColumn]);
				}
				startColumn++;
			}
		}
		return spiral;
	}

	static ArrayList<Integer> boundaryTraversal(int matrix[][], int n, int m) {
		ArrayList<Integer> boundary = new ArrayList<Integer>();
		for (int i = 0; i < m; i++) {
			boundary.add(matrix[0][i]);
		}
		for (int i = 1; i < n; i++) {
			boundary.add(matrix[i][m - 1]);
		}
		if (n > 1) {
			for (int i = m - 2; i >= 0; i--) {
				boundary.add(matrix[n - 1][i]);
			}
		}
		if (m > 1) {
			for (int i = n - 2; i > 0; i--) {
				boundary.add(matrix[i][0]);
			}
		}
		return boundary;
	}

	int findK(int A[][], int n, int m, int k) {
		if (k > (n * m)) {
			return 0;
		}
		int count = 0;
		int startRow = 0, startColumn = 0;
		while (startRow < n && startColumn < m) {
			for (int i = startColumn; i < m; i++) {
				count++;
				if (count == k) {
					return A[startRow][i];
				}
			}
			startRow++;
			for (int i = startRow; i < n; i++) {
				count++;
				if (count == k) {
					return A[i][m - 1];
				}
			}
			m--;
			if (startRow < n) {
				for (int i = m - 1; i >= startColumn; i--) {
					count++;
					if (count == k) {
						return A[n - 1][i];
					}
				}
				n--;
			}
			if (startColumn < m) {
				for (int i = n - 1; i >= startRow; i--) {
					count++;
					if (count == k) {
						return A[i][startColumn];
					}
				}
				startColumn++;
			}
		}
		return 0;
	}

	static boolean search(int matrix[][], int n, int m, int x) {
		int i = 0, j = m - 1;
		while (i < n && j >= 0) {
			if (matrix[i][j] == x) {
				return true;
			}
			if (matrix[i][j] > x) {
				j--;
			} else {
				i++;
			}
		}
		return false;
	}

	static void printMatrix(int arr[][]) {
		for (int i = 0; i < arr.length; i++) {
			for (int j = 0; j < arr[0].length; j++)
				System.out.print(arr[i][j] + " ");
			System.out.println("");
		}
	}
}
